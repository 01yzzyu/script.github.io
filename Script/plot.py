import argparse
import json
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import colormaps

from transformers import (
    CLIPVisionConfig, CLIPImageProcessor, CLIPVisionModel, CLIPVisionModelWithProjection,
    CLIPTokenizerFast, CLIPTextModelWithProjection,
)

# ----------------- 基础工具 -----------------
def sanitize_filename(text):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', text.strip())[:100]

def pair_key(a: str, b: str) -> str:
    """统一键名（字典序）"""
    return "|".join(sorted([a, b]))

# ----------------- 配置与模型塔 -----------------
class Args:
    mm_vision_select_layer = -2
    mm_vision_select_feature = "patch"
    unfreeze_mm_vision_tower = False

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def load_text_tower(self, device_map=None):
        vision_tower_with_projection = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name).to(self.device)
        self.vision_tower.visual_projection = vision_tower_with_projection.visual_projection.to(self.device)
        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(self.vision_tower_name)
        self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.text_tower.requires_grad_(False)
        self.max_position_embeddings = self.text_tower.config.max_position_embeddings

    def feature_select(self, image_forward_outs, output_attentions=False):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if output_attentions:
            image_attentions = image_forward_outs.attentions[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
            if output_attentions:
                image_attentions = image_attentions[:, :, 0, 1:]
        elif self.select_feature == 'cls_patch':
            pass
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        if output_attentions:
            return image_features, image_attentions
        return image_features

    @torch.no_grad()
    def forward(self, images, texts=None, output_attentions=False):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
            return image_features

        if output_attentions:
            def save_cls_attn_hook(module, input, output):
                _, attn_weights = output
                module.cls_attn_output = attn_weights[:, :, 0, 1:].detach()  # [CLS]→patch
            attn_module = self.vision_tower.vision_model.encoder.layers[self.select_layer].self_attn
            handle = attn_module.register_forward_hook(save_cls_attn_hook)

        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=output_attentions
        )

        image_outputs = self.feature_select(image_forward_outs, output_attentions=output_attentions)
        if output_attentions:
            image_features_only = image_outputs[0]
            image_attentions = image_outputs[1]
            image_k_proj_output = attn_module.cls_attn_output
            cls_token_feature = image_forward_outs.hidden_states[self.select_layer][:, :1]
            image_features = (
                image_features_only.to(images.dtype),
                image_attentions.to(images.dtype),
                image_k_proj_output.to(images.dtype),
                cls_token_feature.to(images.dtype),
            )
            handle.remove()
        else:
            image_features_only = image_outputs
            image_features = image_outputs.to(images.dtype)

        if texts is None:
            return image_features

        text_inputs = self.text_tokenizer(
            text=texts, return_tensors="pt", padding="longest", truncation=True
        )
        text_segment = (text_inputs.input_ids.shape[1] - 1) // self.max_position_embeddings + 1
        text_padding = self.max_position_embeddings * text_segment - text_inputs.input_ids.shape[1]
        text_inputs = {
            k: torch.cat([v, v.new_zeros((v.shape[0], text_padding))], dim=1)
            .reshape(-1, self.max_position_embeddings)
            .to(self.device)
            for k, v in text_inputs.items()
        }
        text_embeds = self.text_tower(**text_inputs).text_embeds  # (B,D)

        image_embeds = self.vision_tower.vision_model.post_layernorm(image_features_only)
        image_embeds = self.vision_tower.visual_projection(image_embeds.float())  # (B,N,D)

        return (image_features, image_embeds, text_embeds)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size

    @property
    def patch_size(self):
        return self.config.patch_size

# ----------------- 剪枝器 -----------------
class script:
    def __init__(self, model_dir, visual_token_num=64):
        self.args = Args()
        self.vision_tower = CLIPVisionTower(model_dir, self.args)
        self.vision_tower.load_model()
        self.vision_tower.load_text_tower()
        self.processor = self.vision_tower.image_processor
        self.visual_token_num = visual_token_num
        self.grid_size = self.vision_tower.num_patches_per_side
        self.image_size = self.vision_tower.image_size
        self.patch_size = self.vision_tower.patch_size

    def prune(self, image, question):
        image_tensor = self.processor(image, return_tensors="pt")["pixel_values"].to(self.vision_tower.device)

        # 1) forward with attentions
        (image_features, image_embeds, text_embeds) = self.vision_tower(
            image_tensor, texts=[question], output_attentions=True
        )
        
        # unpack
        feat_only, attn, _, _ = image_features  # feat_only: (B,N,D), attn: (B,h,N)
        B, N, D = feat_only.shape
        device = feat_only.device

        # 2) structure-based
        key_features = F.normalize(feat_only, dim=-1)
        similarity = torch.bmm(key_features, key_features.transpose(1, 2))  # (B,N,N)

        sim_threshold = 0.2
        K = 4
        gamma = 1.0

        valid_mask = similarity >= sim_threshold
        valid_counts = valid_mask.sum(dim=-1)
        row_mask = valid_counts >= K

        score_candidate = (similarity * valid_mask).sum(dim=-1) / valid_counts.clamp(min=1)
        score_candidate = valid_counts * torch.exp(gamma * (score_candidate - sim_threshold))
        score_alternative = similarity.sum(dim=-1) / similarity.size(-1)
        final_scores_struct = torch.where(row_mask, score_candidate, 0.5 * score_candidate + 0.5 * score_alternative)

        # sorted_indices = final_scores_struct.argsort(dim=-1, descending=True)
        # keep_idx = sorted_indices[:, :self.visual_token_num]
        # mask_struct = torch.zeros(B, N, dtype=torch.bool, device=device)
        # mask_struct.scatter_(1, keep_idx, True)

        sorted_indices = final_scores_struct.argsort(dim=-1, descending=False)
        keep_idx = sorted_indices[:, :self.visual_token_num]
        mask_struct = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask_struct.scatter_(1, keep_idx, True)

        # 3) text relevance
        text_embeds_exp = text_embeds.unsqueeze(0).expand(B, -1, -1)
        text_relevance = torch.bmm(text_embeds_exp, image_embeds.transpose(1, 2)).mean(dim=1)
        text_relevance = (text_relevance - text_relevance.min(dim=1, keepdim=True)[0]) / (
            text_relevance.max(dim=1, keepdim=True)[0] - text_relevance.min(dim=1, keepdim=True)[0] + 1e-6
        )

        selected_text_embeds = text_embeds.unsqueeze(1).expand(B, 1, -1)
        relevance = torch.bmm(image_embeds, selected_text_embeds.transpose(1, 2)).mean(dim=-1)
        relevance = (relevance - relevance.min(dim=1, keepdim=True)[0]) / (
            relevance.max(dim=1, keepdim=True)[0] - relevance.min(dim=1, keepdim=True)[0] + 1e-6
        )

        # 4) GS-DPP
        kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
        cis = torch.zeros((self.visual_token_num, B, N), device=device)
        di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()
        select_idx = torch.empty((self.visual_token_num, B), dtype=torch.long, device=device)

        for i in range(self.visual_token_num):
            j = torch.argmax(di2s, dim=-1)
            select_idx[i] = j
            eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) / (
                torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1) + 1e-6)
            cis[i] = eis
            di2s -= torch.square(eis)
            di2s[torch.arange(B), j] = -float('inf')

        mask_gs_dpp = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask_gs_dpp.scatter_(1, torch.sort(select_idx.t()).values, True)

        # 5) Top-K relevance (辅助)
        topk_idx = relevance.argsort(descending=True)[:, :self.visual_token_num]
        mask_gs = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask_gs.scatter_(1, topk_idx, True)

        # 6) final = intersection + pad
        intersection_mask = mask_struct & mask_gs_dpp
        final_mask = intersection_mask.clone()
        num_selected = intersection_mask.sum(dim=1)
        for b in range(B):
            if num_selected[b] < self.visual_token_num:
                pad_candidates = mask_gs[b] & ~intersection_mask[b]
                candidates = torch.nonzero(pad_candidates, as_tuple=False).squeeze(1)
                if candidates.numel() > 0:
                    pad = candidates[torch.argsort(relevance[b, candidates], descending=True)][:self.visual_token_num - num_selected[b]]
                    final_mask[b, pad] = True

        # 7) attention top-K
        attn_scores = attn.mean(dim=1).squeeze(0)  # (N,)
        topk_attn = attn_scores.topk(self.visual_token_num).indices
        mask_attn = torch.zeros_like(attn_scores, dtype=torch.bool)
        mask_attn[topk_attn] = True

        # 8) diversity top-K
        features = F.normalize(image_embeds, dim=-1).squeeze(0)  # (N,D)
        full_repr = features.mean(dim=0, keepdim=True)
        div_scores = 1 - F.cosine_similarity(features, full_repr)  # (N,)
        topk_div = div_scores.topk(self.visual_token_num).indices
        mask_divprune = torch.zeros_like(div_scores, dtype=torch.bool)
        mask_divprune[topk_div] = True

        return (
            mask_struct[0], mask_gs[0], final_mask[0],
            final_scores_struct[0], relevance[0], final_scores_struct[0],  # 保留占位
            mask_gs_dpp[0], mask_attn, mask_divprune,
            attn_scores, div_scores
        )

    # --------- 可视化 ----------
    def visualize_mask(self, image_path, mask, output_path):
        img = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.imshow(img)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                if not mask[idx]:
                    rect = plt.Rectangle((j * self.patch_size, i * self.patch_size),
                                         self.patch_size, self.patch_size,
                                         linewidth=0, edgecolor=None, facecolor='white', alpha=0.65)
                    ax.add_patch(rect)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def visualize_scores(self, image_path, scores, output_path, title="Score Map"):
        import matplotlib.colors as mcolors
        img = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.imshow(img)

        scores = scores.detach().cpu().numpy()
        vmin = np.percentile(scores, 5)
        vmax = np.percentile(scores, 95)
        if abs(vmax - vmin) < 1e-6:
            vmin = scores.min()
            vmax = scores.max() + 1e-3

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colormap = colormaps["jet"]
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                color = colormap(norm(scores[idx]))
                rect = plt.Rectangle(
                    (j * self.patch_size, i * self.patch_size),
                    self.patch_size, self.patch_size,
                    linewidth=0, edgecolor=None, facecolor=color, alpha=0.5
                )
                ax.add_patch(rect)

        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

# ----------------- 两两重叠计算 -----------------
def compute_pairwise_overlaps(mask_dict: dict, visual_token_num: int) -> dict:
    """
    输入:
      mask_dict: {name: (N,) bool tensor}
      visual_token_num: K
    输出:
      { "a|b": {"count": int, "ratio": float, "exact_same": bool}, ... }
    """
    names = sorted(mask_dict.keys())
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ma, mb = mask_dict[a], mask_dict[b]
            overlap = (ma & mb).sum().item()
            exact_same = bool(torch.equal(ma, mb))
            out[pair_key(a, b)] = {
                "count": int(overlap),
                "ratio": float(overlap) / float(visual_token_num),
                "exact_same": exact_same
            }
    return out

# ----------------- 多进程 Worker -----------------
def run_gpu_worker(worker_id, gpu_id, args, visual_token_num):
    torch.cuda.set_device(gpu_id)
    torch.set_num_threads(1)

    print(f"[GPU {gpu_id}] Init model with K={visual_token_num} ...")
    pruner = script(args.model_dir, visual_token_num=visual_token_num)

    with open(args.question_file, "r") as f:
        samples = [json.loads(line) for line in f]

    total_workers = args.num_gpus * args.num_workers_per_gpu
    split_samples = [samples[i] for i in range(len(samples)) if i % total_workers == worker_id]

    vt_out_dir = f"{args.output_dir}_{visual_token_num}"
    os.makedirs(vt_out_dir, exist_ok=True)

    worker_jsonl = os.path.join(vt_out_dir, f"overlap_worker{worker_id}.jsonl")
    with open(worker_jsonl, "w", encoding="utf-8") as fout:
        for sample in tqdm(split_samples, desc=f"GPU {gpu_id} [K={visual_token_num}]"):
            image_name = sample.get("image", None)
            try:
                question = sample["text"].replace("\nAnswer the question using a single word or phrase.", "").strip()
                image_path = os.path.join(args.image_root, image_name)
                image = Image.open(image_path).convert("RGB")

                (mask_struct, mask_gs, final_mask,
                 struct_scores, gs_scores, final_scores,
                 mask_gs_dpp, mask_attn, mask_divprune,
                 attn_scores, div_scores) = pruner.prune(image, question)

                mask_dict = {
                    "struct":    mask_struct,
                    "gs_topk":   mask_gs,
                    "final":     final_mask,
                    "gs_dpp":    mask_gs_dpp,
                    "attn":      mask_attn,
                    "divprune":  mask_divprune,
                }

                pairwise = compute_pairwise_overlaps(mask_dict, visual_token_num)

                rec = {
                    "image": image_name,
                    "question": question,
                    "visual_token_num": visual_token_num,
                    "pairs": pairwise
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"[GPU {gpu_id}] Error on {image_name}: {e}")

    print(f"[GPU {gpu_id}] Worker {worker_id} wrote {worker_jsonl}")

# ----------------- 汇总 JSON -----------------
def summarize_jsonl(vt_out_dir: str, visual_token_num: int):
    """
    读取 vt_out_dir 下所有 overlap_worker*.jsonl，聚合每个 pair：
      - avg_count
      - avg_ratio
      - exact_match_rate  (完全重叠比率)
    写入 pairwise_overlap_summary.json
    """
    files = [os.path.join(vt_out_dir, f) for f in os.listdir(vt_out_dir)
             if f.startswith("overlap_worker") and f.endswith(".jsonl")]

    agg = {}   # pair -> {"sum_count": float, "sum_ratio": float, "sum_exact": int}
    total = 0  # 样本计数

    for path in files:
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pairs = rec["pairs"]
                # 初始化键
                if not agg and pairs:
                    for k in pairs.keys():
                        agg[k] = {"sum_count": 0.0, "sum_ratio": 0.0, "sum_exact": 0}
                # 累加
                for k, v in pairs.items():
                    if k not in agg:
                        agg[k] = {"sum_count": 0.0, "sum_ratio": 0.0, "sum_exact": 0}
                    agg[k]["sum_count"] += float(v["count"])
                    agg[k]["sum_ratio"] += float(v["ratio"])
                    agg[k]["sum_exact"] += 1 if v.get("exact_same", False) else 0
                total += 1

    summary = {
        "visual_token_num": visual_token_num,
        "num_samples": total,
        "pairs": {}
    }
    if total > 0:
        for k, v in sorted(agg.items()):
            summary["pairs"][k] = {
                "avg_count": v["sum_count"] / total,
                "avg_ratio": v["sum_ratio"] / total,
                "exact_match_rate": v["sum_exact"] / total
            }

    out_path = os.path.join(vt_out_dir, "pairwise_overlap_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[SUMMARY] Wrote {out_path}")

# ----------------- 单图调试：也落盘 JSON -----------------
def run_single_image_pruning(
    model_dir: str,
    image_path: str,
    question: str,
    output_dir: str = "./single_image_output",
    visual_token_num: int = 64
):
    pruner = script(model_dir, visual_token_num=visual_token_num)
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")

    (mask_struct, mask_gs, final_mask,
     struct_scores, gs_scores, final_scores,
     mask_gs_dpp, mask_attn, mask_divprune,
     attn_scores, div_scores) = pruner.prune(image, question)

    mask_dict = {
        "struct":    mask_struct,
        "gs_topk":   mask_gs,
        "final":     final_mask,
        "gs_dpp":    mask_gs_dpp,
        "attn":      mask_attn,
        "divprune":  mask_divprune,
    }
    pairwise = compute_pairwise_overlaps(mask_dict, visual_token_num)

    single_json = {
        "image": os.path.basename(image_path),
        "question": question,
        "visual_token_num": visual_token_num,
        "pairs": pairwise
    }
    out_path = os.path.join(output_dir, "single_image_pairwise_overlap.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(single_json, f, ensure_ascii=False, indent=2)
    print(f"[Single Image] Pairwise overlap (with exact_same) saved -> {out_path}")

# ----------------- 主程序 -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="openai/clip-vit-large-patch14-336")
    # parser.add_argument("--image_root", type=str, default="./playground/data/eval/textvqa/train_images")
    # parser.add_argument("--question_file", type=str, default="./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl")
    # parser.add_argument("--output_dir", type=str, default="output_textvqa1")

    parser.add_argument("--image_root", type=str, default="./playground/data/eval/vqav2/test2015")
    parser.add_argument("--question_file", type=str, default="./playground/data/eval/vqav2/llava_vqav2_mscoco_test2015.jsonl")
    parser.add_argument("--output_dir", type=str, default="output_vqav21")

    # Scienceqa
    # parser.add_argument("--image_root", type=str, default="./playground/data/eval/scienceqa/images/test")
    # parser.add_argument("--question_file", type=str, default="./playground/data/eval/scienceqa/llava_test_CQM-I—transfer.jsonl")
    # parser.add_argument("--output_dir", type=str, default="output_scienceqa")
 
    # mme
    # parser.add_argument("--image_root", type=str, default="./playground/data/eval/MME/MME_Benchmark_release_version")
    # parser.add_argument("--question_file", type=str, default="./playground/data/eval/MME/llava_mme.jsonl")
    # parser.add_argument("--output_dir", type=str, default="output_mme1")

    # vizwiz
    # parser.add_argument("--image_root", type=str, default="./playground/data/eval/vizwiz/test")
    # parser.add_argument("--question_file", type=str, default="./playground/data/eval/vizwiz/llava_test.jsonl")
    # parser.add_argument("--output_dir", type=str, default="output_vizwiz")
    
    # gqa
    # parser.add_argument("--image_root", type=str, default="./playground/data/eval/gqa/data/images")
    # parser.add_argument("--question_file", type=str, default="./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl")
    # parser.add_argument("--output_dir", type=str, default="output_gqa1")

    # pope
    # parser.add_argument("--image_root", type=str, default="./playground/data/eval/pope/val2014")
    # parser.add_argument("--question_file", type=str, default="./playground/data/eval/pope/llava_pope_test.jsonl")
    # parser.add_argument("--output_dir", type=str, default="output_pope1")

    # # mm-vet
    # parser.add_argument("--image_root", type=str, default="./playground/data/eval/mm-vet/images")
    # parser.add_argument("--question_file", type=str, default="./playground/data/eval/mm-vet/llava-mm-vet.jsonl")
    # parser.add_argument("--output_dir", type=str, default="output_mm-vet")

    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--visual_token_nums", type=int, nargs="+", default=[16, 32, 192, 64, 128])
    parser.add_argument("--num_workers_per_gpu", type=int, default=16, help="parallel workers per GPU")
    args = parser.parse_args()

    ctx = torch.multiprocessing.get_context("spawn")

    for visual_token_num in args.visual_token_nums:
        processes = []
        total_workers = args.num_gpus * args.num_workers_per_gpu
        for worker_id in range(total_workers):
            gpu_id = worker_id % args.num_gpus
            p = ctx.Process(
                target=run_gpu_worker,
                args=(worker_id, gpu_id, args, visual_token_num)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # 汇总该 K 的所有 worker jsonl -> summary json（包含 exact_match_rate）
        vt_out_dir = f"{args.output_dir}_{visual_token_num}"
        summarize_jsonl(vt_out_dir, visual_token_num)

if __name__ == "__main__":
    main()
