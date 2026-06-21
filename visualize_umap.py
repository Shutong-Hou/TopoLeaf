import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# 尝试导入 umap-learn，如果没有安装则回退到 PCA
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("警告: 未安装 umap-learn，将使用 PCA 替代。")
    print("建议安装: pip install umap-learn")

# ============================================
# 配置
# ============================================
FEATURES_DIR = "features_cache"
RESULTS_DIR = "results"
VIZ_DIR = "visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# 要可视化的物种（Apple基准 + 代表性测试物种）
BENCHMARK_SPECIES = "Apple"
TEST_SPECIES = ["Strawberry", "Corn_(maize)", "Tomato", "Grape", "Peach"]

# 每物种采样图像数
MAX_IMAGES_PER_SPECIES = 50

# ============================================
# 1. 加载特征
# ============================================
def load_cached_features(cache_name):
    cache_path = os.path.join(FEATURES_DIR, f"{cache_name}.pkl")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存不存在: {cache_path}")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def get_image_level_features(features_dict, img_paths, max_images=None):
    """从patch特征计算图像级平均特征"""
    if max_images and len(img_paths) > max_images:
        np.random.seed(42)
        img_paths = np.random.choice(img_paths, max_images, replace=False).tolist()
    
    img_feats = []
    labels = []
    for p in img_paths:
        if p in features_dict:
            img_feats.append(features_dict[p].mean(axis=0))
            labels.append(p)
    return np.array(img_feats), labels

# ============================================
# 2. 收集所有物种的特征
# ============================================
print("加载特征缓存...")

# 加载Apple基准健康特征（用于维度选择）
bench_cache = load_cached_features(f"bench_{BENCHMARK_SPECIES}")
bench_paths = list(bench_cache.keys())
bench_feats = np.array([bench_cache[p].mean(axis=0) for p in bench_paths])

# 稳定维度选择
variances = np.var(np.concatenate([bench_cache[p] for p in bench_paths[:20]], axis=0), axis=0)
thr = np.percentile(variances, 50)
sel_dims = np.where(variances <= thr)[0]
print(f"稳定维度: {len(sel_dims)}")

# 收集所有物种的图像级特征
all_features = []
all_labels = []
all_species = []

# 基准健康
bench_sel = bench_feats[:, sel_dims]
all_features.append(bench_sel)
all_labels.extend([f"{BENCHMARK_SPECIES}_healthy"] * len(bench_sel))
all_species.extend([BENCHMARK_SPECIES] * len(bench_sel))

# 测试物种
for test_sp in TEST_SPECIES:
    try:
        test_cache = load_cached_features(f"test_{test_sp}")
        test_paths_all = list(test_cache.keys())
        
        # 分离健康和病害
        healthy_paths = [p for p in test_paths_all if "healthy" in os.path.basename(p).lower()]
        disease_paths = [p for p in test_paths_all if "healthy" not in os.path.basename(p).lower()]
        
        # 采样
        healthy_feats, _ = get_image_level_features(test_cache, healthy_paths, MAX_IMAGES_PER_SPECIES)
        disease_feats, _ = get_image_level_features(test_cache, disease_paths, MAX_IMAGES_PER_SPECIES)
        
        if len(healthy_feats) > 0:
            h_sel = healthy_feats[:, sel_dims]
            all_features.append(h_sel)
            all_labels.extend([f"{test_sp}_healthy"] * len(h_sel))
            all_species.extend([test_sp] * len(h_sel))
        
        if len(disease_feats) > 0:
            d_sel = disease_feats[:, sel_dims]
            all_features.append(d_sel)
            all_labels.extend([f"{test_sp}_disease"] * len(d_sel))
            all_species.extend([test_sp] * len(d_sel))
            
        print(f"  {test_sp}: H={len(healthy_feats)}, D={len(disease_feats)}")
    except FileNotFoundError:
        print(f"  跳过 {test_sp}: 缓存不存在")
    except Exception as e:
        print(f"  跳过 {test_sp}: {e}")

all_features = np.concatenate(all_features, axis=0)
print(f"总样本数: {len(all_features)}")

# ============================================
# 3. 降维可视化
# ============================================
print("降维...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(all_features)

if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(features_scaled)
    method_name = "UMAP"
else:
    reducer = PCA(n_components=2, random_state=42)
    embedding = reducer.fit_transform(features_scaled)
    method_name = "PCA"

# ============================================
# 4. 可视化
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 颜色方案：每个物种一个颜色
species_list = [BENCHMARK_SPECIES] + [s for s in TEST_SPECIES if s in all_species]
colors = plt.cm.tab10(np.linspace(0, 1, len(species_list)))
species_colors = dict(zip(species_list, colors))

# 图1：按物种着色，健康=圆点，病害=叉号
ax = axes[0]
for sp in species_list:
    mask_sp = np.array([s == sp for s in all_species])
    if not mask_sp.any():
        continue
    
    # 健康
    mask_h = mask_sp & np.array(["healthy" in l for l in all_labels])
    if mask_h.any():
        ax.scatter(embedding[mask_h, 0], embedding[mask_h, 1], 
                   c=[species_colors[sp]], marker='o', s=30, alpha=0.7,
                   label=f"{sp} (healthy)", edgecolors='white', linewidth=0.3)
    
    # 病害
    mask_d = mask_sp & np.array(["disease" in l for l in all_labels])
    if mask_d.any():
        ax.scatter(embedding[mask_d, 0], embedding[mask_d, 1],
                   c=[species_colors[sp]], marker='x', s=40, alpha=0.7,
                   label=f"{sp} (disease)", linewidth=1.0)

ax.set_title(f"Feature Space ({method_name}): Species Distribution", fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, framealpha=0.9)
ax.set_xlabel(f"{method_name} 1")
ax.set_ylabel(f"{method_name} 2")

# 图2：健康/病害热力分区
ax2 = axes[1]
# 绘制所有点
mask_healthy = np.array(["healthy" in l for l in all_labels])
mask_disease = np.array(["disease" in l for l in all_labels])

ax2.scatter(embedding[mask_healthy, 0], embedding[mask_healthy, 1],
            c='#3498db', marker='o', s=25, alpha=0.6, label='Healthy')
ax2.scatter(embedding[mask_disease, 0], embedding[mask_disease, 1],
            c='#e74c3c', marker='x', s=35, alpha=0.6, label='Disease')

# 绘制Apple健康区域的凸包
if sum(mask_healthy) > 3:
    from scipy.spatial import ConvexHull
    apple_healthy_mask = np.array([f"{BENCHMARK_SPECIES}_healthy" == l for l in all_labels])
    if apple_healthy_mask.sum() >= 3:
        points = embedding[apple_healthy_mask]
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax2.plot(points[simplex, 0], points[simplex, 1], 'b-', alpha=0.3, linewidth=1)
        ax2.fill(points[hull.vertices, 0], points[hull.vertices, 1], 
                 alpha=0.08, color='blue')

ax2.set_title(f"Healthy vs Disease ({BENCHMARK_SPECIES} benchmark)", fontsize=14)
ax2.legend(fontsize=10)
ax2.set_xlabel(f"{method_name} 1")
ax2.set_ylabel(f"{method_name} 2")

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "feature_space_umap.png"), dpi=200, bbox_inches='tight')
print(f"保存: {VIZ_DIR}/feature_space_umap.png")

# ============================================
# 5. 物种间统计距离 vs AUROC 分析
# ============================================
print("\n物种间统计距离分析...")
from scipy.spatial.distance import cdist

# 加载Apple基准健康特征
apple_healthy_feats = bench_feats[:, sel_dims]
apple_centroid = apple_healthy_feats.mean(axis=0)

# 加载各物种健康特征的中心
species_centroids = {BENCHMARK_SPECIES: apple_centroid}
for test_sp in TEST_SPECIES:
    try:
        test_cache = load_cached_features(f"test_{test_sp}")
        test_paths_all = list(test_cache.keys())
        healthy_paths = [p for p in test_paths_all if "healthy" in os.path.basename(p).lower()]
        if len(healthy_paths) > 0:
            h_feats, _ = get_image_level_features(test_cache, healthy_paths, MAX_IMAGES_PER_SPECIES)
            species_centroids[test_sp] = h_feats[:, sel_dims].mean(axis=0)
    except:
        pass

# 计算各物种健康中心到Apple健康中心的余弦距离
print(f"\n物种健康中心到 {BENCHMARK_SPECIES} 健康中心的余弦距离:")
for sp, centroid in species_centroids.items():
    if sp == BENCHMARK_SPECIES:
        continue
    dist = cdist(centroid.reshape(1, -1), apple_centroid.reshape(1, -1), metric='cosine')[0][0]
    
    # 查找对应的AUROC
    try:
        with open(os.path.join(RESULTS_DIR, f"results_{BENCHMARK_SPECIES}.json"), 'r') as f:
            apple_results = json.load(f)
        if sp in apple_results:
            geo_auc = apple_results[sp]['geo']
            print(f"  {sp}: 余弦距离={dist:.4f}, 几何AUROC={geo_auc:.4f}")
        else:
            print(f"  {sp}: 余弦距离={dist:.4f}, AUROC=N/A")
    except:
        print(f"  {sp}: 余弦距离={dist:.4f}")

print(f"\n所有可视化结果保存于: {VIZ_DIR}/")