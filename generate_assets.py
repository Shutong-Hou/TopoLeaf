import numpy as np
import cv2
import pickle
import os
import json
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

FEATURES_DIR = "features_cache"
RESULTS_DIR = "results"
CACHE_DIR = "benchmark_cache"
ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

BENCHMARK_SPECIES = ["Apple", "Tomato", "Grape", "Strawberry", "Corn_(maize)"]
IMAGE_SIZE = 448

# ============================================
# 图1: 跨物种 AUROC 矩阵热力图
# ============================================
print("生成跨物种矩阵热力图...")
matrix = {}
for bench in BENCHMARK_SPECIES:
    path = os.path.join(RESULTS_DIR, f"results_{bench}.json")
    if os.path.exists(path):
        with open(path) as f:
            matrix[bench] = json.load(f)

species_order = ["Cherry_(including_sour)", "Corn_(maize)", "Grape", "Peach",
                 "Pepper,_bell", "Potato", "Strawberry", "Tomato"]
bench_list = [b for b in BENCHMARK_SPECIES if b in matrix]

data = np.zeros((len(species_order), len(bench_list)))
for i, sp in enumerate(species_order):
    for j, b in enumerate(bench_list):
        if sp in matrix[b]:
            data[i, j] = matrix[b][sp]['geo']
        else:
            data[i, j] = np.nan

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(data, cmap='YlGn', vmin=0, vmax=1, aspect='auto')

# 在每个格子上标注数值
for i in range(len(species_order)):
    for j in range(len(bench_list)):
        if not np.isnan(data[i, j]):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

# 短标签（去掉 _(maize) 等）
short_species = [s.replace("_(including_sour)", "").replace("_(maize)", "").replace(",_bell", "") for s in species_order]

ax.set_xticks(range(len(bench_list)))
ax.set_xticklabels(bench_list, rotation=30, ha='right', fontsize=10)
ax.set_yticks(range(len(species_order)))
ax.set_yticklabels(short_species, fontsize=10)
ax.set_title("Cross-Species Zero-Shot Anomaly Detection (AUROC)\nBenchmark → Test Species", fontsize=13, fontweight='bold')
ax.set_xlabel("Benchmark Species", fontsize=11)
ax.set_ylabel("Test Species", fontsize=11)
plt.colorbar(im, ax=ax, label='Geometric AUROC', shrink=0.85)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "cross_species_matrix.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  -> cross_species_matrix.png")

# ============================================
# 图2: 健康 vs 病害热力图对比（用第一对可用数据）
# ============================================
print("生成热力图对比...")

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "facebookresearch_dinov2_main")
model = torch.hub.load(local, 'dinov2_vits14', source='local', pretrained=True).to(DEVICE)
model.eval()

# 加载 Apple 基准
with open(os.path.join(CACHE_DIR, "Apple.pkl"), 'rb') as f:
    bench_data = pickle.load(f)
knn = bench_data["knn"]
sel_dims = bench_data["sel_dims"]
mean_loc = bench_data["mean_loc"]
std_loc = bench_data["std_loc"]

# 加载一张健康 Apple 和一张病害 Apple
with open(os.path.join(FEATURES_DIR, "bench_Apple.pkl"), 'rb') as f:
    apple_feats = pickle.load(f)

healthy_paths = list(apple_feats.keys())
np.random.seed(42)
healthy_example = healthy_paths[0]  # 健康示例

# 加载病害示例
with open(os.path.join(FEATURES_DIR, "test_Apple.pkl"), 'rb') as f:
    test_apple = pickle.load(f)
disease_paths = [p for p in test_apple.keys() if "healthy" not in os.path.basename(p).lower()][:5]
disease_example = disease_paths[0] if disease_paths else healthy_example

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = torch.from_numpy(img).permute(2,0,1).float()/255.0
    tensor = (tensor - torch.tensor([0.485,0.456,0.406]).view(3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = model.forward_features(tensor)['x_norm_patchtokens']
    return feats.squeeze(0).cpu().numpy(), img

def generate_heatmap(feats, orig_img):
    fsel = feats[:, sel_dims]
    dists, _ = knn.kneighbors(fsel)
    dev = (dists.mean(axis=1) - mean_loc) / std_loc
    hmap = dev.reshape(int(np.sqrt(len(dev))), -1)
    hmap = np.clip(hmap, 0, None)
    hmap = cv2.resize(hmap, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    hmap = cv2.GaussianBlur(hmap, (21,21), 0)
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    colored = cv2.applyColorMap((hmap*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_img, 0.4, colored, 0.6, 0)
    return overlay, hmap

feat_h, img_h = extract_features(healthy_example)
feat_d, img_d = extract_features(disease_example)

if feat_h is not None and feat_d is not None:
    over_h, hmap_h = generate_heatmap(feat_h, img_h)
    over_d, hmap_d = generate_heatmap(feat_d, img_d)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # 健康行
    axes[0, 0].imshow(img_h)
    axes[0, 0].set_title("Healthy Leaf", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(hmap_h, cmap='inferno')
    axes[0, 1].set_title("Healthy Heatmap", fontsize=12)
    axes[0, 1].axis('off')
    axes[0, 2].imshow(over_h)
    axes[0, 2].set_title("Healthy Overlay", fontsize=12)
    axes[0, 2].axis('off')
    # 病害行
    axes[1, 0].imshow(img_d)
    axes[1, 0].set_title("Disease Leaf", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(hmap_d, cmap='inferno')
    axes[1, 1].set_title("Disease Heatmap", fontsize=12)
    axes[1, 1].axis('off')
    axes[1, 2].imshow(over_d)
    axes[1, 2].set_title("Disease Overlay", fontsize=12)
    axes[1, 2].axis('off')

    fig.suptitle("TopoLeaf: Anomaly Heatmap Comparison", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "heatmap_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  -> heatmap_comparison.png")

# ============================================
# 图3: UMAP 特征空间定位
# ============================================
print("生成 UMAP 特征空间图...")
umap_emb = bench_data["umap_embedding"]
geo_healthy = bench_data["healthy_scores_geo"]

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], c=geo_healthy, cmap='viridis', s=20, alpha=0.8, edgecolors='white', linewidth=0.3)

# 如果有病害示例特征，标出位置
if feat_d is not None:
    reducer = bench_data.get("umap_reducer") or bench_data.get("pca_reducer")
    if reducer is not None:
        img_feat_d = feat_d[:, sel_dims].mean(axis=0).reshape(1, -1)
        try:
            new_pt = reducer.transform(img_feat_d)
            ax.scatter(new_pt[0,0], new_pt[0,1], c='red', marker='*', s=400, edgecolors='white', linewidth=2, label='Disease Example')
            ax.legend(fontsize=10)
        except:
            pass

plt.colorbar(sc, ax=ax, label='Healthy Geo Score (σ)', shrink=0.85)
ax.set_title("Feature Space (UMAP): Apple Healthy Benchmark", fontsize=13, fontweight='bold')
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "umap_space.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  -> umap_space.png")

# ============================================
# 图4: 邻居缩略图展示
# ============================================
print("生成邻居展示图...")
from sklearn.neighbors import NearestNeighbors

# 用健康基准的所有图像级特征
img_feats = bench_data["image_level_feats"]
img_paths = bench_data["image_paths"]

# 选一张健康示例作为查询
query_idx = 0
query_feat = img_feats[query_idx]
dists = np.linalg.norm(img_feats - query_feat, axis=1)
sorted_idx = np.argsort(dists)[:6]  # 包括自己

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, idx in enumerate(sorted_idx[1:]):  # 排除自己
    img = cv2.imread(img_paths[idx])
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(f"#{i+1} (dist={dists[idx]:.3f})", fontsize=9)
    axes[i].axis('off')

fig.suptitle("Most Similar Healthy Leaves to Query", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "neighbor_gallery.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  -> neighbor_gallery.png")

print(f"\n全部素材已生成至 {ASSETS_DIR}/")