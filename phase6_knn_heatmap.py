import torch
import numpy as np
import cv2
import os
import glob
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
import scipy
import matplotlib.pyplot as plt

# --- 配置 ---
HEALTHY_DIR = "data/healthy"
DISEASE_DIR = "data/disease"
DINO_BACKBONE = "dinov2_vits14"
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANCE_PERCENTILE = 50
K_NEIGHBORS = 20  # 局部上下文窗口大小

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
print(f"原始维度: 384, 保留稳定维度: {len(selected_dims)} (方差 <= {threshold_var:.4f})")

healthy_selected = healthy_reference[:, selected_dims]
healthy_centroid = np.mean(healthy_selected, axis=0, keepdims=True)

# 构建健康patch的局部邻域模型
knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine')
knn.fit(healthy_selected)

# 健康样本自身局部异常评分（用于校准阈值）
healthy_local_dists = []
for feat in all_healthy_features:
    feat_sel = feat[:, selected_dims]
    dists, _ = knn.kneighbors(feat_sel)
    # 到K个邻居的平均余弦距离
    local_dists = dists.mean(axis=1)
    healthy_local_dists.append(local_dists)
healthy_local_dists = np.concatenate(healthy_local_dists)
mean_local = healthy_local_dists.mean()
std_local = healthy_local_dists.std()
print(f"健康局部离群评分: mean={mean_local:.4f}, std={std_local:.4f}")

# --- 4. 测试所有图像 ---
def compute_anomaly_scores(img_path):
    feat = extract_patch_features(img_path)
    feat_sel = feat[:, selected_dims]
    # 全局评分：到健康质心的余弦距离
    global_dists = scipy.spatial.distance.cdist(feat_sel, healthy_centroid, metric='cosine').flatten()
    # 局部评分：到健康邻居的平均距离
    local_dists, _ = knn.kneighbors(feat_sel)
    local_dists = local_dists.mean(axis=1)
    # 最终异常评分：局部评分（用偏离正常局部变异程度的sigma值）
    patch_dev = (local_dists - mean_local) / std_local
    return global_dists, local_dists, patch_dev

test_paths, test_labels, test_names = [], [], []
for path in sorted(glob.glob(os.path.join(HEALTHY_DIR, "*.jpg")) + glob.glob(os.path.join(HEALTHY_DIR, "*.png"))):
    test_paths.append(path)
    test_labels.append(0)
    test_names.append(os.path.basename(path))
for path in sorted(glob.glob(os.path.join(DISEASE_DIR, "*.jpg")) + glob.glob(os.path.join(DISEASE_DIR, "*.png"))):
    test_paths.append(path)
    test_labels.append(1)
    test_names.append(os.path.basename(path))

print(f"\n测试 {len(test_paths)} 张图像...")
scores_dev = []
for i, (path, label) in enumerate(zip(test_paths, test_labels)):
    _, local_dists, patch_dev = compute_anomaly_scores(path)
    # 图像级异常评分：取patch dev的P90
    img_dev = np.percentile(patch_dev, 90)
    scores_dev.append(img_dev)
    print(f"  [{i+1}/{len(test_paths)}] {os.path.basename(path)} label={label} | img_dev={img_dev:.2f}σ")

scores_dev = np.array(scores_dev)
labels = np.array(test_labels)

if len(set(labels)) == 2:
    au_roc = roc_auc_score(labels, scores_dev)
    healthy_devs = scores_dev[labels == 0]
    threshold = healthy_devs.max() if len(healthy_devs) > 0 else 0
    preds = (scores_dev > threshold).astype(int)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    print(f"\n========== 量化指标 ==========")
    print(f"AUROC: {au_roc:.4f}")
    print(f"阈值 (健康集最大dev={threshold:.2f}σ) 的 F1: {f1:.4f}, Accuracy: {acc:.4f}")
else:
    print("测试集只有一类样本")

# --- 5. 生成局部异常热力图 ---
disease_example = next((p for p, l in zip(test_paths, test_labels) if l == 1), None)
if disease_example:
    print(f"\n生成局部异常热力图: {os.path.basename(disease_example)}")
    _, _, patch_dev = compute_anomaly_scores(disease_example)
    num_patches = int(np.sqrt(len(patch_dev)))
    heatmap = patch_dev.reshape(num_patches, num_patches)
    # 截断极端值让热力图更平滑
    heatmap = np.clip(heatmap, -2, 6)
    heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    original_img = cv2.imread(disease_example)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE))

    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.5, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet'); axes[1].set_title("Local Anomaly Heatmap (KNN)"); axes[1].axis('off')
    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig("topoleaf_phase6_knn_heatmap.png")
    print("热力图已保存: topoleaf_phase6_knn_heatmap.png")