import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from itertools import combinations
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA
from scipy.stats import entropy
from tqdm import tqdm
import matplotlib.patches as patches
import concurrent.futures  # 引入并发库
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.transform import resize


# === CONFIG ===
model_name = "openai/clip-vit-large-patch14-336"
image_dir = "./playground/data/eval/pope/val2014"
output_dir = "./outputs1"
os.makedirs(output_dir, exist_ok=True)

resize_size = 336  
grid_size = 24
redundancy_threshold = 0.2
max_images = 10000
max_workers = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === LOAD MODEL ===
print("Loading CLIP model...")
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# === PLOT UTILITY ===

def plot_patch_heatmap_on_image(image, heatmap, save_path, cmap='jet', alpha=0.5):
    fig, ax = plt.subplots(figsize=(6, 6))
    if image is not None:
        ax.imshow(image)
    h, w = image.size if image is not None else (heatmap.shape[0], heatmap.shape[1])
    grid_h, grid_w = heatmap.shape
    patch_h, patch_w = h // grid_w, w // grid_h

    # 🔧 动态范围增强：拉伸分布到 5%-95%
    scores = heatmap.flatten()
    vmin = np.percentile(scores, 5)
    vmax = np.percentile(scores, 95)
    if abs(vmax - vmin) < 1e-6:
        vmin = scores.min()
        vmax = scores.max() + 1e-3

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_func = plt.get_cmap(cmap)

    for y in range(grid_h):
        for x in range(grid_w):
            val = heatmap[y, x]
            color = cmap_func(norm(val))
            rect = patches.Rectangle(
                (x * patch_w, y * patch_h),
                patch_w, patch_h,
                linewidth=0,
                edgecolor=None,
                facecolor=color,
                alpha=alpha
            )
            ax.add_patch(rect)

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


# === 处理图像时确保数据也移至GPU ===
def process_single_image(img_data):
    img_idx, img = img_data
    
    # 处理图像并将输入数据移至GPU
    inputs = processor(images=img, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入数据移至GPU

    with torch.no_grad():
        outputs = model.vision_model(pixel_values=inputs["pixel_values"])  # 模型已在GPU上
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

    tokens = patch_tokens[0].cpu().numpy()  # 计算后的tokens移回CPU
    sim_matrix = cosine_similarity(tokens)
    coords = [(i // grid_size, i % grid_size) for i in range(tokens.shape[0])]

    # 1. Distance vs similarity
    distance_groups = defaultdict(list)
    for i, j in combinations(range(tokens.shape[0]), 2):
        y1, x1 = coords[i]
        y2, x2 = coords[j]
        dist = abs(y1 - y2) + abs(x1 - x2)
        similarity = sim_matrix[i, j]
        distance_groups[dist].append(similarity)
    avg_similarity = {d: np.mean(sims) for d, sims in distance_groups.items()}

    # 2. Redundancy mask & local/random similarity
    mask = np.zeros((grid_size, grid_size))
    redundancy_score_map = np.zeros((grid_size, grid_size))  # 新增：连续冗余得分
    img_local_sims = []
    img_random_sims = []

    for i in range(tokens.shape[0]):
        yi, xi = coords[i]
        local_sims_per_patch = []
        random_sims_per_patch = []

        for j in range(tokens.shape[0]):
            if i == j:
                continue
            yj, xj = coords[j]
            dist = abs(yi - yj) + abs(xi - xj)
            sim = sim_matrix[i, j]
            if dist <= 1:
                local_sims_per_patch.append(sim)
            elif np.random.rand() < 0.01:
                random_sims_per_patch.append(sim)

        if local_sims_per_patch:
            avg_sim = np.mean(local_sims_per_patch)
            redundancy_score_map[yi, xi] = avg_sim  # ⬅️ 新增
            if avg_sim > redundancy_threshold:
                mask[yi, xi] = 1

            img_local_sims.append(avg_sim)
        if random_sims_per_patch:
            img_random_sims.append(np.mean(random_sims_per_patch))

    # 返回一个包含所有结果的字典
    return {
        "avg_similarity": avg_similarity,
        "redundancy_mask": mask,
        "redundancy_score_map": redundancy_score_map,  # ⬅️ 新增
        "local_sims": img_local_sims,
        "random_sims": img_random_sims,
        "tokens": tokens,
        "is_first": img_idx == 0,
    }



def compute_token_local_entropy_map(tokens, grid_size=24, radius=1):
    token_map = tokens.reshape(grid_size, grid_size, -1)
    entropy_map = np.zeros((grid_size, grid_size))

    for y in range(grid_size):
        for x in range(grid_size):
            neighbors = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid_size and 0 <= nx < grid_size:
                        neighbors.append(token_map[ny, nx])
            neighbors = np.array(neighbors)
            if len(neighbors) < 2:
                entropy_val = 0.0
            else:
                try:
                    reduced = PCA(n_components=1).fit_transform(neighbors)
                    hist, _ = np.histogram(reduced[:, 0], bins=20, density=True)
                    entropy_val = entropy(hist + 1e-8)
                except:
                    entropy_val = 0.0
            entropy_map[y, x] = entropy_val

    entropy_map = entropy_map / np.max(entropy_map + 1e-8)
    return entropy_map



def run_with_threshold(threshold):
    global redundancy_threshold
    redundancy_threshold = threshold
    suffix = f"thresh_{threshold:.1f}"
    current_output_dir = os.path.join(output_dir, suffix)
    os.makedirs(current_output_dir, exist_ok=True)

    # === LOAD IMAGES ===
    print("Loading images...")
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
    if max_images:
        image_paths = image_paths[:max_images]
    images = [Image.open(p).convert("RGB").resize((resize_size, resize_size)) for p in image_paths]
    images_with_indices = list(enumerate(images))

    # === STATS CONTAINERS ===
    all_avg_sims = []
    redundancy_counts = np.zeros((grid_size, grid_size))
    local_sim_list = []
    random_sim_list = []
    token_grid = defaultdict(list)
    all_tokens_list = []
    example_img = None
    example_tokens = None
    token_local_entropy_accumulator = np.zeros((grid_size, grid_size))
    local_pixel_entropy_accumulator = np.zeros((grid_size, grid_size))

    # === PARALLEL PROCESSING ===
    print("Processing images in parallel...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(process_single_image, images_with_indices)
        for result in tqdm(results_iterator, total=len(images), desc="Processing Images"):
            results.append(result)

    # === AGGREGATION ===
    print("Aggregating results...")
    for result in results:
        all_avg_sims.append(result["avg_similarity"])
        redundancy_counts += result["redundancy_mask"]
        local_sim_list.extend(result["local_sims"])
        random_sim_list.extend(result["random_sims"])
        tokens = result["tokens"]
        all_tokens_list.append(tokens)

        token_entropy_map = compute_token_local_entropy_map(result["tokens"], grid_size=grid_size, radius=1)
        token_local_entropy_accumulator += token_entropy_map

        # 如果是第一张图，保存下来用于可视化叠加
        if result["is_first"]:
            token_entropy_example = token_entropy_map
            example_tokens = tokens
            example_img = images[0]

        # === Compute local entropy map using skimage (pixel-level)
        gray_img = rgb2gray(np.array(images[results.index(result)]))
        gray_img_ubyte = img_as_ubyte(gray_img)
        entropy_map_gray = sk_entropy(gray_img_ubyte, disk(5))  # radius = 5 for local window
        entropy_resized = resize(entropy_map_gray, (grid_size, grid_size), mode='reflect', anti_aliasing=True)
        entropy_resized = entropy_resized / np.max(entropy_resized + 1e-8)  # normalize
        local_pixel_entropy_accumulator += entropy_resized


        coords = [(i // grid_size, i % grid_size) for i in range(tokens.shape[0])]
        for i, token in enumerate(tokens):
            y, x = coords[i]
            token_grid[(y, x)].append(token)

    # === PLOTTING ===

    # 1. Similarity vs Distance
    max_distance = max(max(d.keys()) for d in all_avg_sims)
    avg_curve = np.zeros(max_distance + 1)
    counts = np.zeros(max_distance + 1)
    for sim_dict in all_avg_sims:
        for d, val in sim_dict.items():
            avg_curve[d] += val
            counts[d] += 1
    avg_curve = avg_curve / np.maximum(counts, 1)
    x_vals, y_vals = list(range(1, len(avg_curve))), avg_curve[1:]
    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, y_vals, marker='o', markersize=4, linewidth=2, color="#B7A0D1")
    plt.title("Average Cosine Similarity vs. Spatial Distance", fontsize=19)
    plt.xlabel("Manhattan Distance", fontsize=19)
    plt.ylabel("Average Cosine Similarity", fontsize=19)
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(current_output_dir, "avg_similarity_vs_distance.png"), dpi=300)
    plt.close()

    # 2. KDE Plot
    plt.figure(figsize=(6, 5))
    sns.kdeplot(local_sim_list, label="Local Neighbors", fill=True, color="#A6C8D2")
    sns.kdeplot(random_sim_list, label="Random Patches", fill=True, color="#B1A3BF")
    plt.grid(False)
    plt.title("Similarity Distribution: Local vs Random", fontsize=19)
    plt.xlabel("Cosine Similarity", fontsize=19)
    plt.ylabel("Density", fontsize=19)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(current_output_dir, "local_vs_random_similarity.png"))
    plt.close()

    # 3. Entropy Map
    entropy_map = np.zeros((grid_size, grid_size))
    for (y, x), vecs in tqdm(token_grid.items(), desc="Computing Entropy"):
        vecs = np.array(vecs)
        if vecs.shape[0] < 2:
            entropy_val = 0.0
        else:
            n_comp = min(10, vecs.shape[0], vecs.shape[1])
            try:
                pca = PCA(n_components=n_comp)
                reduced = pca.fit_transform(vecs)
                hist, _ = np.histogram(reduced[:, 0], bins=30, density=True)
                entropy_val = entropy(hist + 1e-8)
            except Exception:
                entropy_val = 0.0
        entropy_map[y, x] = entropy_val
    entropy_map = entropy_map / np.max(entropy_map + 1e-8)

    # 4. Redundancy & Entropy Overlays for First Image
    avg_redundancy = redundancy_counts / len(images)
    example_img.save(os.path.join(current_output_dir, "example_raw_image.png"))

    plot_patch_heatmap_on_image(
        example_img,
        results[0]["redundancy_score_map"],
        os.path.join(current_output_dir, "redundancy_overlay.png"),
        cmap="jet",
        alpha=0.5
    )

    example_entropy_map = np.zeros((grid_size, grid_size))
    coords = [(i // grid_size, i % grid_size) for i in range(example_tokens.shape[0])]
    for i, token in enumerate(example_tokens):
        y, x = coords[i]
        local_tokens = []
        for j, other_token in enumerate(example_tokens):
            if i == j:
                continue
            yj, xj = coords[j]
            dist = abs(y - yj) + abs(x - xj)
            if dist <= 1:
                local_tokens.append(other_token)
        if len(local_tokens) >= 2:
            vecs = np.array(local_tokens)
            try:
                n_comp = min(5, vecs.shape[0], vecs.shape[1])
                pca = PCA(n_components=n_comp)
                reduced = pca.fit_transform(vecs)
                hist, _ = np.histogram(reduced[:, 0], bins=20, density=True)
                entropy_val = entropy(hist + 1e-8)
            except Exception:
                entropy_val = 0.0
        else:
            entropy_val = 0.0
        example_entropy_map[y, x] = entropy_val
    example_entropy_map = example_entropy_map / np.max(example_entropy_map + 1e-8)

    plot_patch_heatmap_on_image(
        example_img,
        example_entropy_map,
        os.path.join(current_output_dir, "entropy_overlay.png"),
        cmap="jet",
        alpha=0.5
    )

    # 5. Final heatmaps without overlay
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_redundancy, cmap="jet", interpolation="nearest")
    plt.axis('off')
    # plt.title("Redundancy Heatmap (All Images)", fontsize=19)
    plt.tight_layout()
    plt.savefig(os.path.join(current_output_dir, "redundancy_heatmap_all_images.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(entropy_map, cmap="jet", interpolation="nearest")
    plt.axis('off')
    # plt.title("Entropy Heatmap (All Images)", fontsize=19)
    plt.tight_layout()
    plt.savefig(os.path.join(current_output_dir, "entropy_heatmap_all_images.png"), dpi=300)
    plt.close()

    # 6. All tokens PCA cluster map
    all_tokens_combined = np.vstack(all_tokens_list)
    print(f"Performing PCA on {all_tokens_combined.shape[0]} tokens...")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_tokens_combined)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5, s=2, color="#A3B1BF")
    plt.title("Cluster Map of Tokens from All Images", fontsize=27)
    plt.xlabel("PCA Component 1", fontsize=28)
    plt.ylabel("PCA Component 2", fontsize=28)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(current_output_dir, "all_images_token_cluster_map.png"), dpi=300)
    plt.close()

    # Save heatmap overlay on raw image
    plot_patch_heatmap_on_image(
        example_img,
        entropy_resized,
        os.path.join(current_output_dir, "pixel_entropy_overlay.png"),
        cmap="jet",
        alpha=0.5
    )

    # Save standalone pixel entropy heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(entropy_resized, cmap="jet", interpolation="nearest")
    plt.axis('off')
    plt.title("Local Pixel Entropy (Example)", fontsize=19)
    plt.tight_layout()
    plt.savefig(os.path.join(current_output_dir, "pixel_entropy_heatmap_example.png"), dpi=300)
    plt.close()

    avg_pixel_entropy = local_pixel_entropy_accumulator / len(images)

    plt.figure(figsize=(6, 6))
    plt.imshow(avg_pixel_entropy, cmap="jet", interpolation="nearest")
    plt.axis('off')
    plt.title("Pixel Entropy Heatmap (All Images)", fontsize=19)
    plt.tight_layout()
    plt.savefig(os.path.join(current_output_dir, "pixel_entropy_heatmap_all_images.png"), dpi=300)
    plt.close()

    # === Token Local Entropy Heatmap: overlay on raw image
    plot_patch_heatmap_on_image(
        example_img,
        token_entropy_example,
        os.path.join(current_output_dir, "token_local_entropy_overlay.png"),
        cmap="jet",
        alpha=0.5
    )

    # === Token Local Entropy Heatmap: averaged over all images
    avg_token_local_entropy = token_local_entropy_accumulator / len(images)
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_token_local_entropy, cmap="jet", interpolation="nearest")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(current_output_dir, "token_local_entropy_heatmap_all_images.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


    print(f"=== Done with threshold {threshold:.1f} ===")


if __name__ == "__main__":
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        run_with_threshold(thresh)
