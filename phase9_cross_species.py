import torch
import numpy as np
import cv2
import os
import glob
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from ripser import ripser
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================
# 配置
# ============================================
PLANTVILLAGE_DIR = "data/PlantVillage/raw/color"  # 直接指向 color 子目录
DINO_BACKBONE = "dinov2_vits14"
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANCE_PERCENTILE = 50
K_NEIGHBORS = 30
TOPO_K_NEIGHBORS = 15
TOPO_SAMPLING_RATIO = 0.10
MAX_BENCHMARK_IMAGES = 30
MAX_TEST_PER_CONDITION = 50

BENCHMARK_SPECIES = ["Apple"]
TEST_SPECIES = ["Corn_(maize)", "Tomato", "Grape", "Potato", "Peach", "Strawberry", "Blueberry", "Cherry", "Pepper,_bell", "Raspberry", "Soybean", "Squash"]

print(f"使用设备: {DEVICE}")
print(f"PlantVillage 路径: {PLANTVILLAGE_DIR}")

# ============================================
# 1. 加载 DINOv2
# ============================================
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

# ============================================
# 2. 扫描数据集（适配 PlantVillage raw/color 结构）
# ============================================
def scan_plantvillage_color(base_dir):
    """
    扫描 raw/color 目录。
    结构: raw/color/Apple___healthy/xxx.jpg
    返回: {species: {condition: [path_list]}}
    """
    dataset = {}
    base = Path(base_dir)
    
    if not base.exists():
        raise FileNotFoundError(f"目录不存在: {base_dir}")
    
    # 扫描所有子文件夹
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    
    for subdir in subdirs:
        folder_name = subdir.name
        
        # 解析物种和状态：Apple___healthy -> (Apple, healthy)
        if "___" in folder_name:
            species, condition = folder_name.split("___", 1)
        else:
            species = folder_name
            condition = "unknown"
        
        # 收集该文件夹下所有图像
        images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.JPG")) + \
                 list(subdir.glob("*.png")) + list(subdir.glob("*.PNG"))
        
        if len(images) == 0:
            continue
        
        if species not in dataset:
            dataset[species] = {}
        dataset[species][condition] = [str(img) for img in images]
    
    return dataset

print("\n扫描 PlantVillage 数据集...")
dataset = scan_plantvillage_color(PLANTVILLAGE_DIR)

# 打印统计
print("\n数据集统计:")
for species in sorted(dataset.keys()):
    conditions = dataset[species]
    total = sum(len(v) for v in conditions.values())
    print(f"  {species}: {total} 张图像 ({len(conditions)} 种状态)")
    for cond, paths in sorted(conditions.items()):
        print(f"    - {cond}: {len(paths)} 张")

# ============================================
# 3. 构建健康基准库
# ============================================
print(f"\n构建健康基准库 (物种: {BENCHMARK_SPECIES})...")
healthy_benchmark_paths = []

for species in BENCHMARK_SPECIES:
    if species in dataset:
        for condition, paths in dataset[species].items():
            if "healthy" in condition.lower():
                healthy_benchmark_paths.extend(paths)
                print(f"  使用 {species}/{condition}: {len(paths)} 张")

if len(healthy_benchmark_paths) == 0:
    print("  未找到指定物种的健康图像，回退到所有物种的健康图像...")
    for species in dataset:
        for condition, paths in dataset[species].items():
            if "healthy" in condition.lower():
                healthy_benchmark_paths.extend(paths)

print(f"健康基准总图像数: {len(healthy_benchmark_paths)}")

if len(healthy_benchmark_paths) > MAX_BENCHMARK_IMAGES:
    np.random.seed(42)
    indices = np.random.choice(len(healthy_benchmark_paths), MAX_BENCHMARK_IMAGES, replace=False)
    healthy_benchmark_paths = [healthy_benchmark_paths[i] for i in indices]
    print(f"  限制为 {MAX_BENCHMARK_IMAGES} 张")

# 提取健康基准特征
print("提取健康基准特征...")
all_healthy_features = []
for i, path in enumerate(healthy_benchmark_paths):
    if (i + 1) % 10 == 0:
        print(f"  进度: {i+1}/{len(healthy_benchmark_paths)}")
    try:
        feat = extract_patch_features(path)
        all_healthy_features.append(feat)
    except Exception as e:
        print(f"  跳过 {path}: {e}")

healthy_reference = np.concatenate(all_healthy_features, axis=0)
print(f"健康基准总 patch 数: {healthy_reference.shape[0]}")

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
    healthy_local_dists.append(dists.mean(axis=1))
healthy_local_dists = np.concatenate(healthy_local_dists)
mean_local = healthy_local_dists.mean()
std_local = healthy_local_dists.std()
print(f"健康局部距离: mean={mean_local:.4f}, std={std_local:.4f}")

# ============================================
# 4. 评分函数
# ============================================
def compute_geometric_score(img_path):
    feat = extract_patch_features(img_path)
    feat_sel = feat[:, selected_dims]
    local_dists, _ = knn.kneighbors(feat_sel)
    local_dists = local_dists.mean(axis=1)
    patch_dev = (local_dists - mean_local) / std_local
    return {
        'mean': float(patch_dev.mean()),
        'p90': float(np.percentile(patch_dev, 90)),
        'max': float(patch_dev.max())
    }

def compute_topology_score(img_path, sampling_ratio=TOPO_SAMPLING_RATIO):
    feat = extract_patch_features(img_path)
    feat_sel = feat[:, selected_dims]
    n_patches = feat_sel.shape[0]
    n_sample = max(15, int(n_patches * sampling_ratio))
    sample_indices = np.random.choice(n_patches, n_sample, replace=False)
    sample_patches = feat_sel[sample_indices]
    
    topo_scores = []
    for patch in sample_patches:
        query = patch.reshape(1, -1)
        dists, indices = knn.kneighbors(query, n_neighbors=TOPO_K_NEIGHBORS + 1)
        neighbor_indices = indices[0][:TOPO_K_NEIGHBORS]
        neighbor_points = healthy_selected[neighbor_indices]
        
        if len(neighbor_points) < 3:
            topo_scores.append(0.0)
            continue
        try:
            dgm_n = ripser(neighbor_points, maxdim=0, n_perm=None)['dgms'][0]
            augmented = np.vstack([neighbor_points, query])
            dgm_a = ripser(augmented, maxdim=0, n_perm=None)['dgms'][0]
            finite_n = dgm_n[np.isfinite(dgm_n[:, 1])]
            finite_a = dgm_a[np.isfinite(dgm_a[:, 1])]
            max_n = finite_n[:, 1].max() if len(finite_n) > 0 else 0
            max_a = finite_a[:, 1].max() if len(finite_a) > 0 else 0
            pers_n = (finite_n[:, 1] - finite_n[:, 0]).mean() if len(finite_n) > 0 else 0
            pers_a = (finite_a[:, 1] - finite_a[:, 0]).mean() if len(finite_a) > 0 else 0
            score = abs(max_a - max_n) + abs(pers_a - pers_n) * 5
            topo_scores.append(float(score))
        except:
            topo_scores.append(0.0)
    
    topo_scores = np.array(topo_scores)
    return {
        'mean': float(topo_scores.mean()),
        'p90': float(np.percentile(topo_scores, 90)),
        'max': float(topo_scores.max())
    }

# ============================================
# 5. 跨物种实验
# ============================================
print(f"\n{'='*60}")
print(f"跨物种零样本异常检测实验")
print(f"基准物种: {BENCHMARK_SPECIES}")
print(f"测试物种: {TEST_SPECIES}")
print(f"{'='*60}")

all_results = {}

for test_species in TEST_SPECIES:
    if test_species not in dataset:
        print(f"\n跳过 {test_species}: 数据集中不存在")
        continue
    
    print(f"\n--- 测试物种: {test_species} ---")
    
    test_paths = []
    test_labels = []
    
    for condition, paths in dataset[test_species].items():
        sample_paths = paths[:MAX_TEST_PER_CONDITION] if len(paths) > MAX_TEST_PER_CONDITION else paths
        for path in sample_paths:
            test_paths.append(path)
            if "healthy" in condition.lower():
                test_labels.append(0)
            else:
                test_labels.append(1)
    
    if len(test_paths) == 0 or len(set(test_labels)) < 2:
        print(f"  跳过: 样本不足或只有一类")
        continue
    
    print(f"  测试图像: {len(test_paths)} (健康: {test_labels.count(0)}, 病害: {test_labels.count(1)})")
    
    geo_scores = []
    topo_scores = []
    
    for i, (path, label) in enumerate(zip(test_paths, test_labels)):
        if (i + 1) % 50 == 0:
            print(f"    进度: {i+1}/{len(test_paths)}")
        try:
            geo = compute_geometric_score(path)
            geo_scores.append(geo['p90'])
            topo = compute_topology_score(path)
            topo_scores.append(topo['p90'])
        except Exception as e:
            geo_scores.append(np.mean(geo_scores) if geo_scores else 0)
            topo_scores.append(np.mean(topo_scores) if topo_scores else 0)
    
    geo_arr = np.array(geo_scores)
    topo_arr = np.array(topo_scores)
    labels_arr = np.array(test_labels[:len(geo_arr)])
    
    geo_auroc = roc_auc_score(labels_arr, geo_arr)
    topo_auroc = roc_auc_score(labels_arr, topo_arr)
    
    scaler = StandardScaler()
    geo_norm = scaler.fit_transform(geo_arr.reshape(-1, 1)).flatten()
    topo_norm = scaler.fit_transform(topo_arr.reshape(-1, 1)).flatten()
    fusion = geo_norm * 0.5 + topo_norm * 0.5
    fusion_auroc = roc_auc_score(labels_arr, fusion)
    
    print(f"  几何 AUROC: {geo_auroc:.4f}")
    print(f"  拓扑 AUROC: {topo_auroc:.4f}")
    print(f"  融合 AUROC: {fusion_auroc:.4f}")
    
    all_results[test_species] = {
        'geo_auroc': geo_auroc,
        'topo_auroc': topo_auroc,
        'fusion_auroc': fusion_auroc,
        'n_test': len(geo_arr),
        'n_healthy': int((labels_arr == 0).sum()),
        'n_disease': int((labels_arr == 1).sum())
    }

# ============================================
# 6. 汇总
# ============================================
print(f"\n{'='*60}")
print("跨物种泛化实验汇总报告")
print(f"基准物种: {BENCHMARK_SPECIES}")
print(f"{'='*60}")
print(f"{'物种':<25} {'样本':<8} {'几何AUROC':<12} {'拓扑AUROC':<12} {'融合AUROC':<12}")
print("-" * 75)

for species, res in all_results.items():
    print(f"{species:<25} {res['n_test']:<8} {res['geo_auroc']:.4f}        {res['topo_auroc']:.4f}        {res['fusion_auroc']:.4f}")

if all_results:
    avg_geo = np.mean([r['geo_auroc'] for r in all_results.values()])
    avg_topo = np.mean([r['topo_auroc'] for r in all_results.values()])
    avg_fusion = np.mean([r['fusion_auroc'] for r in all_results.values()])
    print("-" * 75)
    print(f"{'平均':<25} {'':<8} {avg_geo:.4f}        {avg_topo:.4f}        {avg_fusion:.4f}")

# ============================================
# 7. 可视化
# ============================================
if all_results:
    species_names = list(all_results.keys())
    geo_aurocs = [all_results[s]['geo_auroc'] for s in species_names]
    topo_aurocs = [all_results[s]['topo_auroc'] for s in species_names]
    fusion_aurocs = [all_results[s]['fusion_auroc'] for s in species_names]
    
    x = np.arange(len(species_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, geo_aurocs, width, label='几何评分', color='#3498db')
    ax.bar(x, topo_aurocs, width, label='拓扑评分', color='#e74c3c')
    ax.bar(x + width, fusion_aurocs, width, label='融合评分', color='#2ecc71')
    
    ax.set_xlabel('测试物种')
    ax.set_ylabel('AUROC')
    ax.set_title(f'跨物种零样本异常检测 (基准: {", ".join(BENCHMARK_SPECIES)})')
    ax.set_xticks(x)
    ax.set_xticklabels(species_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("topoleaf_cross_species_results.png")
    print("\n跨物种结果图已保存: topoleaf_cross_species_results.png")