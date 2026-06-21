import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import scipy
from scipy.optimize import linear_sum_assignment

HEALTHY_IMG_PATH = "healthy.jpg"
DISEASE_IMG_PATH = "disease.jpg"
DINO_BACKBONE = "dinov2_vits14"
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")

print(f"正在加载 DINOv2 模型 ({DINO_BACKBONE})...")
dinov2 = torch.hub.load('facebookresearch/dinov2', DINO_BACKBONE).to(DEVICE)
dinov2.eval()
print("模型加载完毕。")

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

print("\n提取健康叶片特征...")
healthy_features = extract_patch_features(HEALTHY_IMG_PATH)
print(f"健康叶片特征矩阵形状: {healthy_features.shape}")

print("\n提取病害叶片特征...")
disease_features = extract_patch_features(DISEASE_IMG_PATH)
print(f"病害叶片特征矩阵形状: {disease_features.shape}")

# ============================================
# 诊断：查看特征点云的尺度
# ============================================
from sklearn.decomposition import PCA

print("\n========== 特征点云诊断 ==========")

for name, feat in [("健康叶片", healthy_features), ("病害叶片", disease_features)]:
    # 特征值的范围
    print(f"\n{name}:")
    print(f"  特征矩阵 min/max: {feat.min():.4f} / {feat.max():.4f}")
    print(f"  特征矩阵均值/标准差: {feat.mean():.4f} / {feat.std():.4f}")
    
    # 点对距离统计
    sample_indices = np.random.choice(len(feat), min(500, len(feat)), replace=False)
    sample_feat = feat[sample_indices]
    dists = scipy.spatial.distance.pdist(sample_feat, metric='euclidean')
    print(f"  点对欧氏距离 (采样500点): min={dists.min():.4f}, median={np.median(dists):.4f}, max={dists.max():.4f}")

# PCA 降维可视化，看看点云的大致结构
pca = PCA(n_components=2)
healthy_pca = pca.fit_transform(healthy_features)
disease_pca = pca.fit_transform(disease_features)

fig_pca, axes_pca = plt.subplots(1, 2, figsize=(12, 5))
axes_pca[0].scatter(healthy_pca[:, 0], healthy_pca[:, 1], s=1, alpha=0.5)
axes_pca[0].set_title("Healthy - PCA projection")
axes_pca[1].scatter(disease_pca[:, 0], disease_pca[:, 1], s=1, alpha=0.5)
axes_pca[1].set_title("Disease - PCA projection")
plt.tight_layout()
plt.savefig("topoleaf_pca_projection.png")
print("\nPCA 投影图已保存为 'topoleaf_pca_projection.png'")

# ============================================
# 计算持续同调（这次不限制可视化范围）
# ============================================
print("\n开始计算持续同调...")
print("计算健康叶片的持续同调...")
healthy_diagrams = ripser(healthy_features, maxdim=1, n_perm=None)['dgms']
print(f"  健康叶片 H0 点数: {len(healthy_diagrams[0])}, H1 点数: {len(healthy_diagrams[1])}")

print("计算病害叶片的持续同调...")
disease_diagrams = ripser(disease_features, maxdim=1, n_perm=None)['dgms']
print(f"  病害叶片 H0 点数: {len(disease_diagrams[0])}, H1 点数: {len(disease_diagrams[1])}")

# ============================================
# 可视化（自动缩放范围）
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("TopoLeaf Phase 1: Topological Fingerprint Comparison", fontsize=16)

plot_diagrams(healthy_diagrams, show=False, ax=axes[0])
axes[0].set_title("Healthy Leaf - Persistence Diagram")

plot_diagrams(disease_diagrams, show=False, ax=axes[1])
axes[1].set_title("Disease Leaf - Persistence Diagram")

plt.tight_layout()
plt.savefig("topoleaf_phase1_comparison.png")
print("可视化结果已保存为 'topoleaf_phase1_comparison.png'。")

# ============================================
# 计算 Wasserstein 距离
# ============================================
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

print("\n计算 Wasserstein 距离...")
wass_h0 = compute_wasserstein_distance(healthy_diagrams[0], disease_diagrams[0], p=2)
wass_h1 = compute_wasserstein_distance(healthy_diagrams[1], disease_diagrams[1], p=2)
print(f"H0 (连通性) 的 Wasserstein 距离: {wass_h0:.4f}")
print(f"H1 (空洞/环) 的 Wasserstein 距离: {wass_h1:.4f}")