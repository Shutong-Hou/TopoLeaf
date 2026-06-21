import torch
import numpy as np
import cv2
import os
import glob
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import scipy
from ripser import ripser
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# --- 配置 ---
HEALTHY_DIR = "data/healthy"
DISEASE_DIR = "data/disease"
DINO_BACKBONE = "dinov2_vits14"
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PCA_VARIANCE = 0.95
N_PERM = 300  # 拓扑计算采样

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

# --- 2. 构建健康基准库 ---
print(f"\n构建健康基准库，从 {HEALTHY_DIR} 读取...")
healthy_paths = sorted(glob.glob(os.path.join(HEALTHY_DIR, "*.jpg")) + glob.glob(os.path.join(HEALTHY_DIR, "*.png")))
if len(healthy_paths) == 0:
    raise FileNotFoundError(f"未在 {HEALTHY_DIR} 中找到任何图像。请放入健康叶片图片。")

all_healthy_features = []
for path in healthy_paths:
    print(f"  处理: {os.path.basename(path)}")
    feat = extract_patch_features(path)
    all_healthy_features.append(feat)

healthy_reference = np.concatenate(all_healthy_features, axis=0)
print(f"健康基准总 patch 数: {healthy_reference.shape[0]} (来自 {len(healthy_paths)} 张图像)")

# PCA 净化
pca = PCA(n_components=PCA_VARIANCE)
healthy_pca = pca.fit_transform(healthy_reference)
print(f"PCA 降维后维度: {healthy_pca.shape[1]}")

healthy_centroid = np.mean(healthy_pca, axis=0, keepdims=True)

# 健康样本自身距离分布
healthy_dists = scipy.spatial.distance.cdist(healthy_pca, healthy_centroid, metric='cosine').flatten()
mean_healthy = healthy_dists.mean()
std_healthy = healthy_dists.std()
print(f"健康参考余弦距离: mean={mean_healthy:.4f}, std={std_healthy:.4f}")

# --- 3. 测试所有图像并计算评分 ---
def compute_anomaly_score(img_path):
    feat = extract_patch_features(img_path)
    feat_pca = pca.transform(feat)
    dists = scipy.spatial.distance.cdist(feat_pca, healthy_centroid, metric='cosine').flatten()
    p90 = np.percentile(dists, 90)
    mean_dist = dists.mean()
    # 偏离健康基准的标准差倍数
    dev = (p90 - mean_healthy) / std_healthy if std_healthy > 0 else p90 - mean_healthy
    return p90, dev, dists, feat

# 收集所有测试图像（健康 + 病害）
test_paths = []
test_labels = []  # 0=健康, 1=病害
test_names = []

for path in sorted(glob.glob(os.path.join(HEALTHY_DIR, "*.jpg")) + glob.glob(os.path.join(HEALTHY_DIR, "*.png"))):
    test_paths.append(path)
    test_labels.append(0)
    test_names.append(os.path.basename(path))

for path in sorted(glob.glob(os.path.join(DISEASE_DIR, "*.jpg")) + glob.glob(os.path.join(DISEASE_DIR, "*.png"))):
    test_paths.append(path)
    test_labels.append(1)
    test_names.append(os.path.basename(path))

print(f"\n开始测试 {len(test_paths)} 张图像...")
scores_p90 = []
scores_dev = []
for i, (path, label) in enumerate(zip(test_paths, test_labels)):
    p90, dev, _, _ = compute_anomaly_score(path)
    scores_p90.append(p90)
    scores_dev.append(dev)
    print(f"  [{i+1}/{len(test_paths)}] {os.path.basename(path)} label={label} | P90={p90:.4f} | dev={dev:.2f}σ")

# --- 4. 计算分类指标 (使用dev作为异常分数) ---
scores_dev = np.array(scores_dev)
labels = np.array(test_labels)

if len(set(labels)) == 2:
    au_roc = roc_auc_score(labels, scores_dev)
    # 使用健康集的最大dev作为简单阈值
    threshold_dev = 3.0  # 3σ阈值
    preds = (scores_dev > threshold_dev).astype(int)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    print(f"\n========== 量化指标 ==========")
    print(f"AUROC: {au_roc:.4f}")
    print(f"使用阈值 (dev > 3σ) 的 F1: {f1:.4f}")
    print(f"使用阈值 (dev > 3σ) 的 Accuracy: {acc:.4f}")
else:
    print("\n警告: 测试集中只有一类样本，无法计算AUROC/F1。")

# --- 5. 生成单个样本的热力图示例 (选择第一张病害图像) ---
disease_example = None
for path, label in zip(test_paths, test_labels):
    if label == 1:
        disease_example = path
        break
if disease_example:
    print(f"\n生成示例热力图: {os.path.basename(disease_example)}")
    _, _, dists, feat_raw = compute_anomaly_score(disease_example)
    num_patches = int(np.sqrt(len(dists)))
    heatmap = dists.reshape(num_patches, num_patches)
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
    axes[1].imshow(heatmap, cmap='jet'); axes[1].set_title("Anomaly Heatmap"); axes[1].axis('off')
    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig("topoleaf_phase4_example.png")
    print("示例热力图已保存: topoleaf_phase4_example.png")