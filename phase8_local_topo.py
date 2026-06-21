import torch
import numpy as np
import cv2
import os
import glob
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from ripser import ripser
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# --- 配置 ---
HEALTHY_DIR = "data/healthy"
DISEASE_DIR = "data/disease"
DINO_BACKBONE = "dinov2_vits14"
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANCE_PERCENTILE = 50
K_NEIGHBORS = 30          # 用于局部拓扑分析的邻居数
TOPO_K_NEIGHBORS = 15     # 局部点云中的邻居数（用于持续同调）

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

# --- 4. 构建健康基准的KNN索引 ---
knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine')
knn.fit(healthy_selected)

# --- 5. 定义局部拓扑异常评分函数 ---
def local_topology_anomaly(query_patch, knn_model, topo_k=TOPO_K_NEIGHBORS):
    """
    对单个查询patch计算局部拓扑异常评分。
    1. 找到它的topo_k个最近邻健康patch
    2. 在邻居点云上计算0维持续同调
    3. 加入查询patch后重新计算，度量拓扑结构的变化
    返回：拓扑异常分数（越大越异常）
    """
    query_patch = query_patch.reshape(1, -1)
    dists, indices = knn_model.kneighbors(query_patch, n_neighbors=topo_k + 1)
    
    # 取topo_k个最近邻（排除自己如果重合）
    neighbor_indices = indices[0][:topo_k]
    neighbor_points = healthy_selected[neighbor_indices]
    
    # 邻居点云的0维持续同调
    if len(neighbor_points) < 3:
        return 0.0
    
    try:
        dgm_neighbors = ripser(neighbor_points, maxdim=0, n_perm=None)['dgms'][0]
    except:
        return 0.0
    
    # 加入查询patch后的点云
    augmented_points = np.vstack([neighbor_points, query_patch])
    try:
        dgm_augmented = ripser(augmented_points, maxdim=0, n_perm=None)['dgms'][0]
    except:
        return 0.0
    
    # 度量拓扑变化：0维特征的持久性变化
    # 只考虑有限death的点
    finite_neighbors = dgm_neighbors[np.isfinite(dgm_neighbors[:, 1])]
    finite_augmented = dgm_augmented[np.isfinite(dgm_augmented[:, 1])]
    
    if len(finite_neighbors) == 0 and len(finite_augmented) == 0:
        return 0.0
    
    # 计算最大death值的变化
    max_death_neighbors = finite_neighbors[:, 1].max() if len(finite_neighbors) > 0 else 0
    max_death_augmented = finite_augmented[:, 1].max() if len(finite_augmented) > 0 else 0
    
    # 计算平均持久性的变化
    persistence_neighbors = (finite_neighbors[:, 1] - finite_neighbors[:, 0]).mean() if len(finite_neighbors) > 0 else 0
    persistence_augmented = (finite_augmented[:, 1] - finite_augmented[:, 0]).mean() if len(finite_augmented) > 0 else 0
    
    # 综合变化量
    delta_max = abs(max_death_augmented - max_death_neighbors)
    delta_persistence = abs(persistence_augmented - persistence_neighbors)
    
    # 查询patch的连通性异常：如果它推迟了所有点连通的时间（增大最大death）
    # 或者改变了平均持久性，说明它是局部拓扑离群点
    topo_score = delta_max + delta_persistence * 5  # 加权组合
    
    return float(topo_score)


# --- 6. 批量计算图像级的局部拓扑异常评分 ---
def compute_image_topology_score(img_path, sampling_ratio=0.15):
    """
    对一张图像，采样部分patch计算局部拓扑异常评分。
    采样是为了控制计算时间（全量1024个patch太慢）。
    返回采样patch的拓扑异常分数的统计值。
    """
    feat = extract_patch_features(img_path)
    feat_sel = feat[:, selected_dims]
    
    n_patches = feat_sel.shape[0]
    n_sample = max(20, int(n_patches * sampling_ratio))
    sample_indices = np.random.choice(n_patches, n_sample, replace=False)
    sample_patches = feat_sel[sample_indices]
    
    topo_scores = []
    for patch in sample_patches:
        score = local_topology_anomaly(patch, knn)
        topo_scores.append(score)
    
    topo_scores = np.array(topo_scores)
    return {
        'mean': topo_scores.mean(),
        'p90': np.percentile(topo_scores, 90),
        'max': topo_scores.max(),
        'std': topo_scores.std()
    }


# --- 7. 同时计算KNN几何评分（用于对比）---
def compute_image_geometric_score(img_path):
    """
    Phase 6的KNN几何异常评分，用于和拓扑评分对比。
    """
    feat = extract_patch_features(img_path)
    feat_sel = feat[:, selected_dims]
    
    local_dists, _ = knn.kneighbors(feat_sel)
    local_dists = local_dists.mean(axis=1)
    
    # 健康基准的局部距离统计
    healthy_local_dists = []
    for h_feat in all_healthy_features:
        h_sel = h_feat[:, selected_dims]
        h_dists, _ = knn.kneighbors(h_sel)
        h_local = h_dists.mean(axis=1)
        healthy_local_dists.append(h_local)
    healthy_local_dists = np.concatenate(healthy_local_dists)
    mean_local = healthy_local_dists.mean()
    std_local = healthy_local_dists.std()
    
    patch_dev = (local_dists - mean_local) / std_local
    return {
        'mean': patch_dev.mean(),
        'p90': np.percentile(patch_dev, 90),
        'max': patch_dev.max()
    }


# --- 8. 在所有测试图像上对比两种评分 ---
test_paths, test_labels, test_names = [], [], []
for path in sorted(glob.glob(os.path.join(HEALTHY_DIR, "*.jpg")) + glob.glob(os.path.join(HEALTHY_DIR, "*.png"))):
    test_paths.append(path)
    test_labels.append(0)
    test_names.append(os.path.basename(path))
for path in sorted(glob.glob(os.path.join(DISEASE_DIR, "*.jpg")) + glob.glob(os.path.join(DISEASE_DIR, "*.png"))):
    test_paths.append(path)
    test_labels.append(1)
    test_names.append(os.path.basename(path))

print(f"\n========== 对比几何评分 vs 拓扑评分 ==========")
print(f"测试 {len(test_paths)} 张图像...\n")
print(f"{'图像':<30} {'标签':<6} {'几何P90':<12} {'拓扑P90':<12} {'拓扑均值':<12} {'拓扑最大':<12}")
print("-" * 85)

geo_scores = []
topo_scores_p90 = []
topo_scores_mean = []

for i, (path, label) in enumerate(zip(test_paths, test_labels)):
    name = os.path.basename(path)
    
    # 几何评分
    geo = compute_image_geometric_score(path)
    geo_scores.append(geo['p90'])
    
    # 拓扑评分
    topo = compute_image_topology_score(path, sampling_ratio=0.15)
    topo_scores_p90.append(topo['p90'])
    topo_scores_mean.append(topo['mean'])
    
    print(f"{name:<30} {label:<6} {geo['p90']:<12.2f} {topo['p90']:<12.4f} {topo['mean']:<12.4f} {topo['max']:<12.4f}")

# --- 9. 计算两种评分的分类性能 ---
labels = np.array(test_labels)
geo_arr = np.array(geo_scores)
topo_arr = np.array(topo_scores_p90)

print("\n========== 分类性能对比 ==========")

if len(set(labels)) == 2:
    # 几何评分
    geo_auroc = roc_auc_score(labels, geo_arr)
    geo_threshold = geo_arr[labels == 0].max()
    geo_preds = (geo_arr > geo_threshold).astype(int)
    geo_f1 = f1_score(labels, geo_preds)
    
    # 拓扑评分
    topo_auroc = roc_auc_score(labels, topo_arr)
    topo_threshold = topo_arr[labels == 0].max()
    topo_preds = (topo_arr > topo_threshold).astype(int)
    topo_f1 = f1_score(labels, topo_preds)
    
    print(f"几何评分 (KNN P90):     AUROC={geo_auroc:.4f}, F1={geo_f1:.4f}")
    print(f"拓扑评分 (局部拓扑P90): AUROC={topo_auroc:.4f}, F1={topo_f1:.4f}")
    
    # 融合评分
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    geo_norm = scaler.fit_transform(geo_arr.reshape(-1, 1)).flatten()
    topo_norm = scaler.fit_transform(topo_arr.reshape(-1, 1)).flatten()
    
    fusion = geo_norm * 0.5 + topo_norm * 0.5
    fusion_auroc = roc_auc_score(labels, fusion)
    fusion_threshold = fusion[labels == 0].max()
    fusion_preds = (fusion > fusion_threshold).astype(int)
    fusion_f1 = f1_score(labels, fusion_preds)
    print(f"融合评分 (几何+拓扑):   AUROC={fusion_auroc:.4f}, F1={fusion_f1:.4f}")

# --- 10. 保存对比图 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 散点图：几何 vs 拓扑
axes[0].scatter(geo_arr[labels == 0], topo_arr[labels == 0], c='blue', label='健康', s=60, alpha=0.8)
axes[0].scatter(geo_arr[labels == 1], topo_arr[labels == 1], c='red', label='病害', s=60, alpha=0.8)
axes[0].set_xlabel("几何异常评分 (KNN P90)")
axes[0].set_ylabel("拓扑异常评分 (局部拓扑 P90)")
axes[0].set_title("几何评分 vs 拓扑评分")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 箱线图：几何评分
axes[1].boxplot([geo_arr[labels == 0], geo_arr[labels == 1]], labels=['健康', '病害'])
axes[1].set_title("几何评分分布")
axes[1].set_ylabel("KNN P90 (σ)")

# 箱线图：拓扑评分
axes[2].boxplot([topo_arr[labels == 0], topo_arr[labels == 1]], labels=['健康', '病害'])
axes[2].set_title("拓扑评分分布")
axes[2].set_ylabel("局部拓扑 P90")

plt.tight_layout()
plt.savefig("topoleaf_phase8_geometry_vs_topology.png")
print("\n对比图已保存: topoleaf_phase8_geometry_vs_topology.png")