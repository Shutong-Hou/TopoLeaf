import torch
import numpy as np
import cv2
import os
import glob
from sklearn.neighbors import NearestNeighbors
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# --- 配置 ---
HEALTHY_DIR = "data/healthy"
DISEASE_DIR = "data/disease"
DINO_BACKBONE = "dinov2_vits14"
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANCE_PERCENTILE = 50
K_NEIGHBORS = 20
CLIP_LOW = 2.0    # dev低于此值视为正常背景
CLIP_HIGH = 8.0   # dev高于此值饱和显示

print(f"使用设备: {DEVICE}")

# --- 1. 加载 DINOv2 ---
print(f"加载 DINOv2 模型 ({DINO_BACKBONE})...")
dinov2 = torch.hub.load('facebookresearch/dinov2', DINO_BACKBONE).to(DEVICE)
dinov2.eval()

def extract_patch_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features_dict = dinov2.forward_features(img_tensor)
        features = features_dict['x_norm_patchtokens']
    return features.squeeze(0).cpu().numpy()

# --- 2. 构建健康基准 ---
print(f"\n构建健康基准库，从 {HEALTHY_DIR} 读取...")
healthy_paths = sorted(glob.glob(os.path.join(HEALTHY_DIR, "*.jpg")) + glob.glob(os.path.join(HEALTHY_DIR, "*.png")))
all_healthy_features = []
for path in healthy_paths:
    print(f"  处理: {os.path.basename(path)}")
    feat = extract_patch_features(path)
    all_healthy_features.append(feat)

healthy_reference = np.concatenate(all_healthy_features, axis=0)
print(f"健康基准总 patch 数: {healthy_reference.shape[0]}")

# --- 3. 稳定维度选择 ---
variances = np.var(healthy_reference, axis=0)
threshold_var = np.percentile(variances, VARIANCE_PERCENTILE)
selected_dims = np.where(variances <= threshold_var)[0]
print(f"保留稳定维度: {len(selected_dims)}")

healthy_selected = healthy_reference[:, selected_dims]

knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine')
knn.fit(healthy_selected)

healthy_local_dists = []
for feat in all_healthy_features:
    feat_sel = feat[:, selected_dims]
    dists, _ = knn.kneighbors(feat_sel)
    local_dists = dists.mean(axis=1)
    healthy_local_dists.append(local_dists)
healthy_local_dists = np.concatenate(healthy_local_dists)
mean_local = healthy_local_dists.mean()
std_local = healthy_local_dists.std()
print(f"健康局部离群评分: mean={mean_local:.4f}, std={std_local:.4f}")

# --- 4. 生成修复后的热力图 ---
def compute_patch_dev(img_path):
    feat = extract_patch_features(img_path)
    feat_sel = feat[:, selected_dims]
    local_dists, _ = knn.kneighbors(feat_sel)
    local_dists = local_dists.mean(axis=1)
    patch_dev = (local_dists - mean_local) / std_local
    return patch_dev

# 对每张病害图像生成热力图
disease_paths = sorted(glob.glob(os.path.join(DISEASE_DIR, "*.jpg")) + glob.glob(os.path.join(DISEASE_DIR, "*.png")))
print(f"\n生成 {len(disease_paths)} 张病害图像的热力图...")

for idx, d_path in enumerate(disease_paths):
    patch_dev = compute_patch_dev(d_path)
    num_patches = int(np.sqrt(len(patch_dev)))
    heatmap_raw = patch_dev.reshape(num_patches, num_patches)
    
    # 裁剪到[CLIP_LOW, CLIP_HIGH]并归一化
    heatmap_clipped = np.clip(heatmap_raw, CLIP_LOW, CLIP_HIGH)
    heatmap_norm = (heatmap_clipped - CLIP_LOW) / (CLIP_HIGH - CLIP_LOW)
    
    # 上采样 + 高斯平滑
    heatmap_resized = cv2.resize(heatmap_norm, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    heatmap_smooth = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
    
    # 读取原图
    original_img = cv2.imread(d_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE))
    
    # 热力图着色 - 使用inferno，更能突出异常
    heatmap_colored = cv2.applyColorMap((heatmap_smooth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # 叠加：只在异常区域覆盖热力图
    alpha_mask = heatmap_smooth[:, :, np.newaxis] * 0.7
    overlay = (original_img * (1 - alpha_mask) + heatmap_colored * alpha_mask).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original: {os.path.basename(d_path)}")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_smooth, cmap='inferno')
    axes[1].set_title("Anomaly Heatmap (KNN + Stable Dims)")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (anomaly-weighted)")
    axes[2].axis('off')
    
    plt.tight_layout()
    out_name = f"heatmap_{os.path.splitext(os.path.basename(d_path))[0]}.png"
    plt.savefig(out_name)
    plt.close()
    print(f"  已保存: {out_name}")

print("\n所有热力图生成完毕。")