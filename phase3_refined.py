import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import scipy
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

# --- 配置 ---
HEALTHY_IMG_PATHS = ["healthy.jpg"]
TEST_IMG_PATH = "disease.jpg"
DINO_BACKBONE = "dinov2_vits14"
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")

# ============================================
# 1. 加载 DINOv2
# ============================================
print(f"正在加载 DINOv2 模型 ({DINO_BACKBONE})...")
dinov2 = torch.hub.load('facebookresearch/dinov2', DINO_BACKBONE).to(DEVICE)
dinov2.eval()
print("模型加载完毕。")

# ============================================
# 2. 特征提取函数
# ============================================
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
        patch_features = features_dict['x_norm_patchtokens']

    return patch_features.squeeze(0).cpu().numpy()

# ============================================
# 3. 构建健康基准库
# ============================================
print("\n========== 构建健康基准库 ==========")
all_healthy_features = []
for path in HEALTHY_IMG_PATHS:
    print(f"处理: {path}")
    feat = extract_patch_features(path)
    all_healthy_features.append(feat)

healthy_reference = np.concatenate(all_healthy_features, axis=0)
print(f"健康基准特征矩阵形状: {healthy_reference.shape}")

# ============================================
# 4. PCA 降维净化 (保留95%方差)
# ============================================
print("\n========== PCA 特征空间净化 ==========")
pca = PCA(n_components=0.95)  # 保留95%方差
healthy_pca = pca.fit_transform(healthy_reference)
print(f"PCA 降维后维度: {healthy_pca.shape[1]} (原始384)")
print(f"前5个主成分解释方差比: {pca.explained_variance_ratio_[:5]}")

# 在PCA空间计算质心
healthy_centroid_pca = np.mean(healthy_pca, axis=0, keepdims=True)

# 健康样本自身到质心的距离
healthy_dists_pca = scipy.spatial.distance.cdist(
    healthy_pca, healthy_centroid_pca, metric='cosine'
).flatten()
print(f"健康样本 PCA余弦距离: mean={healthy_dists_pca.mean():.4f}, "
      f"std={healthy_dists_pca.std():.4f}, max={healthy_dists_pca.max():.4f}")

# 用于异常决策的阈值 (健康样本的均值 + 3*std)
threshold = healthy_dists_pca.mean() + 3 * healthy_dists_pca.std()
print(f"异常阈值 (mean+3σ): {threshold:.4f}")

# ============================================
# 5. 测试图像推理
# ============================================
print(f"\n========== 推理: {TEST_IMG_PATH} ==========")
test_features_original = extract_patch_features(TEST_IMG_PATH)
# 投影到PCA空间
test_features_pca = pca.transform(test_features_original)
print(f"测试特征 PCA 形状: {test_features_pca.shape}")

# 计算每个patch到质心的余弦距离
test_dists_pca = scipy.spatial.distance.cdist(
    test_features_pca, healthy_centroid_pca, metric='cosine'
).flatten()

# 异常评分：均值和90分位数
anomaly_score_mean = test_dists_pca.mean()
anomaly_score_p90 = np.percentile(test_dists_pca, 90)
print(f"测试图像异常评分 (均值): {anomaly_score_mean:.4f}")
print(f"测试图像异常评分 (P90): {anomaly_score_p90:.4f}")
print(f"  距离范围: min={test_dists_pca.min():.4f}, max={test_dists_pca.max():.4f}")

# ============================================
# 6. 生成净化后的异常热力图
# ============================================
num_patches_per_side = int(np.sqrt(test_features_pca.shape[0]))
heatmap_raw = test_dists_pca.reshape(num_patches_per_side, num_patches_per_side)

# 上采样到原始图像尺寸
heatmap_resized = cv2.resize(heatmap_raw, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
# 高斯平滑减少块状效应
heatmap_smooth = cv2.GaussianBlur(heatmap_resized, (21, 21), 0)

# 归一化到[0,1]
heatmap_normalized = (heatmap_smooth - heatmap_smooth.min()) / (heatmap_smooth.max() - heatmap_smooth.min() + 1e-8)

# 读取原图
original_img = cv2.imread(TEST_IMG_PATH)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
original_img = cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE))

# 热力图着色
heatmap_colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

# 叠加
overlay = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.5, 0)

fig_heatmap, axes_heatmap = plt.subplots(1, 3, figsize=(18, 6))
axes_heatmap[0].imshow(original_img)
axes_heatmap[0].set_title("Original Image")
axes_heatmap[0].axis('off')

axes_heatmap[1].imshow(heatmap_normalized, cmap='jet')
axes_heatmap[1].set_title("Anomaly Heatmap (PCA + Smoothing)")
axes_heatmap[1].axis('off')

axes_heatmap[2].imshow(overlay)
axes_heatmap[2].set_title("Overlay")
axes_heatmap[2].axis('off')

plt.tight_layout()
plt.savefig("topoleaf_refined_heatmap.png")
print("\n净化热力图已保存为 'topoleaf_refined_heatmap.png'")

# ============================================
# 7. 拓扑距离 (辅助指标，使用原始特征)
# ============================================
print("\n计算拓扑指纹 (原始特征)...")
healthy_topo = ripser(healthy_reference, maxdim=1, n_perm=300)['dgms']
test_topo = ripser(test_features_original, maxdim=1, n_perm=300)['dgms']

fig_topo, axes_topo = plt.subplots(1, 2, figsize=(12, 5))
fig_topo.suptitle("Topological Fingerprint: Healthy Reference vs Test", fontsize=16)

plot_diagrams(healthy_topo, show=False, ax=axes_topo[0])
axes_topo[0].set_title("Healthy Reference")

plot_diagrams(test_topo, show=False, ax=axes_topo[1])
axes_topo[1].set_title("Test Image")

plt.tight_layout()
plt.savefig("topoleaf_topo_comparison_v2.png")
print("拓扑对比图已保存为 'topoleaf_topo_comparison_v2.png'")

def compute_wasserstein_distance(dgm1, dgm2, p=2):
    dgm1_finite = dgm1[np.isfinite(dgm1[:, 1])]
    dgm2_finite = dgm2[np.isfinite(dgm2[:, 1])]

    n1, n2 = len(dgm1_finite), len(dgm2_finite)

    if n1 == 0 and n2 == 0:
        return 0.0
    if n1 == 0:
        diag_dist = (dgm2_finite[:, 1] - dgm2_finite[:, 0]) / np.sqrt(2)
        return np.sum(diag_dist ** p) ** (1/p)
    if n2 == 0:
        diag_dist = (dgm1_finite[:, 1] - dgm1_finite[:, 0]) / np.sqrt(2)
        return np.sum(diag_dist ** p) ** (1/p)

    cost_matrix = scipy.spatial.distance.cdist(dgm1_finite, dgm2_finite, metric='euclidean')
    diag1 = (dgm1_finite[:, 1] - dgm1_finite[:, 0]) / np.sqrt(2)
    diag2 = (dgm2_finite[:, 1] - dgm2_finite[:, 0]) / np.sqrt(2)

    big_cost = np.zeros((n1 + n2, n2 + n1))
    big_cost[:n1, :n2] = cost_matrix
    for i in range(n1):
        big_cost[i, n2 + i] = diag1[i]
    for j in range(n2):
        big_cost[n1 + j, j] = diag2[j]

    row_ind, col_ind = linear_sum_assignment(big_cost)
    total_cost = big_cost[row_ind, col_ind].sum()
    return total_cost ** (1/p)

wass_h0 = compute_wasserstein_distance(healthy_topo[0], test_topo[0], p=2)
wass_h1 = compute_wasserstein_distance(healthy_topo[1], test_topo[1], p=2)
print(f"\n拓扑距离: H0={wass_h0:.4f}, H1={wass_h1:.4f}")

# ============================================
# 8. 融合异常评分
# ============================================
alpha = 0.7
topo_score = (wass_h0 + wass_h1) / 2
combined_score = alpha * anomaly_score_p90 + (1 - alpha) * topo_score
print(f"\n融合异常评分 (α={alpha}): {combined_score:.4f}")
print(f"  几何分量 (P90): {anomaly_score_p90:.4f}")
print(f"  拓扑分量: {topo_score:.4f}")
print(f"  几何分量 (均值): {anomaly_score_mean:.4f}")
print(f"  健康阈值 (PCA均值+3σ): {threshold:.4f}")
print(f"  测试图像是否异常 (基于P90): {'是' if anomaly_score_p90 > threshold else '否'}")