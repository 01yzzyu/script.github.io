#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config, **kwargs):
        super(LlavaMetaModel, self).__init__(config, **kwargs)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def token_merging(self, image_features, index_mask, scaling=1.0):
        B, N, D = image_features.shape
        device = image_features.device

        token_counts = index_mask.sum(dim=1)  # shape (B,)
        all_same = (token_counts == token_counts[0]).all()

        retained_tokens = []
        non_retained_tokens = []

        for b in range(B):
            retained_tokens.append(image_features[b][index_mask[b]])      # (T_b, D)
            non_retained_tokens.append(image_features[b][~index_mask[b]])  # (N - T_b, D)

        if all_same:
            # === ✅ 快路径：所有 retained token 数一致，可 batch stack ===
            T = token_counts[0].item()

            retained = torch.stack(retained_tokens, dim=0)       # (B, T, D)
            non_retained = torch.stack(non_retained_tokens, dim=0)  # (B, N-T, D)

            if non_retained.shape[1] == 0:
                return retained

            # 相似度计算
            cosine_sim = F.cosine_similarity(
                non_retained.unsqueeze(2),       # (B, N-T, 1, D)
                retained.unsqueeze(1),           # (B, 1, T, D)
                dim=-1                           # → (B, N-T, T)
            )
            nearest_idx = cosine_sim.argmax(dim=2)  # (B, N-T)

            # 初始化合并结果
            merged = retained * scaling
            merge_count = torch.ones(B, T, 1, device=device, dtype=torch.float32) * scaling

            # 执行 scatter_add_
            merged.scatter_add_(
                1,
                nearest_idx.unsqueeze(-1).expand(-1, -1, D),
                non_retained
            )
            merge_count.scatter_add_(
                1,
                nearest_idx.unsqueeze(-1),
                torch.ones_like(non_retained[:, :, :1], dtype=torch.float32)
            )

            merged /= merge_count
            return merged

        else:
            # === 🐢 慢路径：每个样本保留 token 数不同 ===
            merged_batch = []

            for b in range(B):
                retained = retained_tokens[b]        # (T_b, D)
                non_retained = non_retained_tokens[b]  # (N - T_b, D)

                if non_retained.shape[0] == 0:
                    merged_batch.append(retained)
                    continue

                sim = F.cosine_similarity(
                    non_retained.unsqueeze(1),  # (N-T, 1, D)
                    retained.unsqueeze(0),      # (1, T, D)
                    dim=-1                     # → (N-T, T)
                )
                nearest_idx = sim.argmax(dim=1)  # (N-T,)

                merged = retained * scaling
                count = torch.ones(retained.shape[0], 1, device=device, dtype=torch.float32) * scaling

                for i in range(non_retained.shape[0]):
                    j = nearest_idx[i].item()
                    merged[j] += non_retained[i]
                    count[j] += 1

                merged /= count
                merged_batch.append(merged)

            # pad 成统一长度
            merged_padded = torch.nn.utils.rnn.pad_sequence(merged_batch, batch_first=True)
            return merged_padded

    def encode_images(self, images, texts=None):
        # Step 0: 提取图像特征、图像嵌入、文本嵌入
        image_features, image_embeds, text_embeds = self.get_model().get_vision_tower()(images, texts=texts)

        B, N, C = image_features.shape
        device = image_features.device

        # Step 1: 映射图像特征至多模态空间（作为 patch feature / key）
        image_features = self.get_model().mm_projector(image_features)  # (B, N, D)
        key_features = F.normalize(image_features, dim=-1)  # (B, N, D)

        # Step 2: 图像 patch-patch 相似度
        similarity = torch.bmm(key_features, key_features.transpose(1, 2))  # (B, N, N)

        # === Step 3: 结构剪枝  ===
        sim_threshold = 0.2
        K = 4
        gamma = 1.0

        valid_mask = similarity >= sim_threshold              
        valid_counts = valid_mask.sum(dim=-1)                
        row_mask = valid_counts >= K                       

        # === 冗余评分 score_candidate：结构连接越多、相似性越强，得分越高 ===
        score_candidate = (similarity * valid_mask).sum(dim=-1) / valid_counts.clamp(min=1)
        score_candidate = valid_counts * torch.exp(gamma * (score_candidate - sim_threshold))
        score_alternative = similarity.sum(dim=-1) / similarity.size(-1)

        # === 结构 mask 的权重引导 final_scores（可调）
        final_scores = torch.where(
            row_mask,                      # 若结构足够连通
            score_candidate,              # 用主力得分
            0.5 * score_candidate + 0.5 * score_alternative  # 否则稍微降权
        )

        keep_r = self.visual_token_num
        sorted_indices = final_scores.argsort(dim=-1, descending=True)
        keep_idx = sorted_indices[:, :keep_r]

        index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
        index_masks.scatter_(1, keep_idx, True)

        # === Step 4: 文本引导剪枝（语义 relevance） ===
        with torch.no_grad():
            M = text_embeds.shape[0]
            text_embeds_expanded = text_embeds.unsqueeze(0).expand(B, -1, -1)  # (B, M, D)

            text_relevance = torch.bmm(text_embeds_expanded, image_embeds.transpose(1, 2))  # (B, M, N)
            text_relevance = text_relevance.mean(dim=-1)  # (B, M)

            text_relevance = (text_relevance - text_relevance.min(dim=1, keepdim=True)[0]) / (
                text_relevance.max(dim=1, keepdim=True)[0] - text_relevance.min(dim=1, keepdim=True)[0] + 1e-6
            )

            T = max(1, int(0.5 * M))
            topk_text_indices = torch.topk(text_relevance, T, dim=-1).indices  # (B, T)

            selected_text_embeds = torch.stack([
                text_embeds[topk_text_indices[b]] for b in range(B)
            ], dim=0)  # (B, T, D)

            # === Step 5: 语义引导 patch relevance ===
            relevance = torch.bmm(image_embeds, selected_text_embeds.transpose(1, 2))  # (B, N, T)
            relevance = (-relevance).mean(dim=-1)  # (B, N)
            relevance = (relevance - relevance.min(dim=1, keepdim=True)[0]) / (
                relevance.max(dim=1, keepdim=True)[0] - relevance.min(dim=1, keepdim=True)[0] + 1e-6
            )

        # === Step 6: 核矩阵（结构 × 语义）
        kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)  # (B, N, N)

        # === Step 7: Gram-Schmidt 正交选择
        cis = torch.zeros((self.visual_token_num, B, N), device=device)
        di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()  # (B, N)
        select_idx = torch.empty((self.visual_token_num, B), dtype=torch.long, device=device)

        for i in range(self.visual_token_num):
            j = torch.argmax(di2s, dim=-1)  # (B,)
            select_idx[i] = j

            eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) / (
                torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1) + 1e-6
            )
            cis[i, :, :] = eis
            di2s -= torch.square(eis)
            di2s[torch.arange(B), j] = -float('inf')

        index_masks_gs = torch.zeros(B, N, dtype=torch.bool, device=device)
        index_masks_gs.scatter_(1, torch.sort(select_idx.t()).values, True)

        # === Step 7: DPP-based greedy MAP inference ===
        selected_idx = torch.empty((self.visual_token_num, B), dtype=torch.long, device=device)
        d_scores = torch.diagonal(kernel, dim1=1, dim2=2).clone()  # (B, N)
        mask = torch.zeros_like(d_scores, dtype=torch.bool)  # (B, N)

        for i in range(self.visual_token_num):
            # Step 7.1: Select token with max unexplained variance
            idx = torch.argmax(d_scores.masked_fill(mask, float('-inf')), dim=-1)  # (B,)
            selected_idx[i] = idx
            mask.scatter_(1, idx.unsqueeze(1), True)  # mark as selected

            # Step 7.2: Update residual variance for remaining tokens
            for b in range(B):
                j = idx[b]  # selected index for batch b
                if d_scores[b, j] <= 0:
                    continue

                # Compute projection term
                proj = kernel[b, :, j].clone()  # (N,)
                for t in range(i):
                    prev = selected_idx[t, b]
                    coeff = kernel[b, j, prev] / (d_scores[b, prev] + 1e-6)
                    proj -= coeff * kernel[b, :, prev]

                # Normalize by current variance
                proj = proj / (torch.sqrt(d_scores[b, j]) + 1e-6)

                # Update residuals
                d_scores[b] -= proj ** 2
                d_scores[b, j] = -float('inf')  # mask selected token

        # === Step 8: 取交集 + 补全到固定 token 数 self.visual_token_num ===
        # intersection_mask = index_masks & index_masks_gs  # (B, N)
        # num_selected = intersection_mask.sum(dim=1)       # (B,)
        # final_mask = intersection_mask.clone()
        index_masks_dpp = torch.zeros(B, N, dtype=torch.bool, device=device)
        index_masks_dpp.scatter_(1, torch.sort(selected_idx.t()).values, True)

        intersection_mask = index_masks & index_masks_dpp  # DPP 替代 GS
        num_selected = intersection_mask.sum(dim=1)       # (B,)
        final_mask = intersection_mask.clone()

        for b in range(B):
            if num_selected[b] < self.visual_token_num:
                dpp_available = index_masks_dpp[b] & (~intersection_mask[b])
                candidate_indices = torch.nonzero(dpp_available, as_tuple=False).squeeze(1)
                if candidate_indices.numel() == 0:
                    continue
                candidate_relevance = relevance[b, candidate_indices]
                sorted_indices = candidate_indices[torch.argsort(candidate_relevance, descending=True)]
                pad_indices = sorted_indices[:self.visual_token_num - num_selected[b]]
                final_mask[b, pad_indices] = True



        # === Step 9: 决策使用哪一种 index_masks
        # return image_features, index_masks      # 使用冗余剪枝
        # return image_features, index_masks_gs  # 使用 Gram-Schmidt
        return image_features, final_mask      # 结构 + 语义交集 + 补全
        # return image_features, index_masks_dpp   # ✅ 使用 DPP MAP 近似推理


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, texts=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, index_masks = self.encode_images(concat_images, texts=texts)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            index_masks = torch.split(index_masks, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            mm_patch_merge_type = mm_patch_merge_type.replace('_unpad', '')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                index_masks = [x.flatten(0, 1) for x in index_masks]
                # image_features = [self.token_merging(x.unsqueeze(0), m.unsqueeze(0)).squeeze(0) for x, m in zip(image_features, index_masks)]
                image_features = [x[m] for x, m in zip(image_features, index_masks)]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, (image_feature, index_mask) in enumerate(zip(image_features, index_masks)):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        base_index_mask = index_mask[0]
                        index_mask = index_mask[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            index_mask = index_mask.view(num_patch_height, num_patch_width, height, width)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous().unsqueeze(0)
                            index_mask = index_mask.flatten(1, 2).flatten(2, 3)
                            index_mask = unpad_image(index_mask, image_sizes[image_idx])
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(*index_mask.shape[:-1], 1, dtype=torch.bool).to(index_mask.device)
                            ), dim=-1)
                            index_mask = index_mask.flatten(1, 2).squeeze(0)
                            image_feature = image_feature[index_mask]
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous()
                            index_mask = index_mask.flatten(0, 3)
                            image_feature = image_feature[index_mask]
                        base_image_feature = base_image_feature[base_index_mask]
                        image_feature = torch.cat((base_image_feature, image_feature))
                    else:
                        image_feature = image_feature[0]
                        index_mask = index_mask[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(1, dtype=torch.bool).to(index_mask.device)
                            ), dim=0)
                        image_feature = image_feature[index_mask]
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features, index_masks = self.encode_images(images, texts=texts)
            # image_features = self.token_merging(image_features, index_masks)
            image_features = image_features[index_masks].unsqueeze(0)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_features[0].shape[0]

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
