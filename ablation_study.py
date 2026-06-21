import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

FEATURES_DIR = "features_cache"
RESULTS_DIR = "ablation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 测试组合
ABLATION_PAIRS = [
    ("Apple", "Strawberry"),
    ("Apple", "Corn_(maize)"),
    ("Corn_(maize)", "Apple"),
]

# 消融参数网格
K_VALUES = [5, 10, 20, 30, 50]
DIM_PERCENTILES = [10, 30, 50, 70, 90]
BENCHMARK_SIZES = [10, 30, 50, 100]

# 固定默认值
DEFAULT_K = 30
DEFAULT_DIM = 50
DEFAULT_BENCH_SIZE = 100

def load_cached_features(cache_name):
    cache_path = os.path.join(FEATURES_DIR, f"{cache_name}.pkl")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def compute_geometric_auroc(bench_paths, bench_feats_dict, test_paths, test_feats_dict, 
                            test_labels, dim_percentile, k_neighbors):
    """向量化计算几何评分AUROC"""
    # 构建基准
    all_healthy = [bench_feats_dict[p] for p in bench_paths]
    healthy_ref = np.concatenate(all_healthy, axis=0)
    
    variances = np.var(healthy_ref, axis=0)
    thr = np.percentile(variances, dim_percentile)
    sel_dims = np.where(variances <= thr)[0]
    healthy_sel = healthy_ref[:, sel_dims]
    
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
    knn.fit(healthy_sel)
    
    h_dists = [knn.kneighbors(f[:, sel_dims])[0].mean(axis=1) for f in all_healthy]
    h_dists = np.concatenate(h_dists)
    mean_loc, std_loc = h_dists.mean(), h_dists.std()
    
    # 向量化测试评分
    all_patches, img_counts = [], []
    for p in test_paths:
        feat = test_feats_dict.get(p)
        if feat is not None:
            all_patches.append(feat[:, sel_dims])
            img_counts.append(len(feat))
        else:
            all_patches.append(np.zeros((1024, len(sel_dims))))
            img_counts.append(1024)
    
    all_patches_stacked = np.concatenate(all_patches, axis=0)
    all_dists, _ = knn.kneighbors(all_patches_stacked)
    all_devs = (all_dists.mean(axis=1) - mean_loc) / std_loc
    
    geo_scores, start = [], 0
    for count in img_counts:
        geo_scores.append(np.percentile(all_devs[start:start + count], 90))
        start += count
    
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(test_labels, geo_scores)

def run_ablation(bench_sp, test_sp):
    print(f"\n{'='*60}")
    print(f"消融实验: {bench_sp} -> {test_sp}")
    
    # 加载特征
    bench_feats = load_cached_features(f"bench_{bench_sp}")
    test_feats = load_cached_features(f"test_{test_sp}")
    
    all_bench_paths = list(bench_feats.keys())
    all_test_paths = list(test_feats.keys())
    
    # 标签
    disease_kw = ["scab", "rot", "rust", "blight", "spot", "mildew", 
                  "virus", "scorch", "mold", "mite", "curl", "mosaic", "esca"]
    
    test_labels = []
    valid_test_paths = []
    for p in all_test_paths:
        pl = p.lower()
        if "healthy" in pl:
            valid_test_paths.append(p); test_labels.append(0)
        elif any(d in pl for d in disease_kw):
            valid_test_paths.append(p); test_labels.append(1)
    
    print(f"  测试样本: {len(valid_test_paths)} (H:{test_labels.count(0)} D:{test_labels.count(1)})")
    
    results = {'k_curve': [], 'dim_curve': [], 'size_curve': []}
    
    # === 1. K值敏感性 ===
    print("  K值消融...")
    np.random.seed(42)
    bench_paths_k = np.random.choice(all_bench_paths, DEFAULT_BENCH_SIZE, replace=False).tolist()
    for k in K_VALUES:
        auc = compute_geometric_auroc(bench_paths_k, bench_feats, valid_test_paths, 
                                      test_feats, test_labels, DEFAULT_DIM, k)
        results['k_curve'].append({'k': k, 'auc': auc})
        print(f"    K={k}: AUROC={auc:.4f}")
    
    # === 2. 维度比例消融 ===
    print("  维度消融...")
    np.random.seed(42)
    bench_paths_d = np.random.choice(all_bench_paths, DEFAULT_BENCH_SIZE, replace=False).tolist()
    for dp in DIM_PERCENTILES:
        auc = compute_geometric_auroc(bench_paths_d, bench_feats, valid_test_paths,
                                      test_feats, test_labels, dp, DEFAULT_K)
        results['dim_curve'].append({'dim_pct': dp, 'auc': auc})
        print(f"    dim={dp}%: AUROC={auc:.4f}")
    
    # === 3. 基准大小消融 ===
    print("  基准大小消融...")
    for size in BENCHMARK_SIZES:
        aucs = []
        # 重复3次取平均
        for seed in [42, 43, 44]:
            np.random.seed(seed)
            bench_paths_s = np.random.choice(all_bench_paths, size, replace=False).tolist()
            auc = compute_geometric_auroc(bench_paths_s, bench_feats, valid_test_paths,
                                          test_feats, test_labels, DEFAULT_DIM, DEFAULT_K)
            aucs.append(auc)
        avg_auc = np.mean(aucs)
        results['size_curve'].append({'size': size, 'auc': avg_auc, 'std': np.std(aucs)})
        print(f"    size={size}: AUROC={avg_auc:.4f}±{np.std(aucs):.4f}")
    
    return results

if __name__ == '__main__':
    all_ablation = {}
    for bs, ts in ABLATION_PAIRS:
        try:
            res = run_ablation(bs, ts)
            all_ablation[f"{bs}_{ts}"] = res
        except Exception as e:
            print(f"错误 {bs}->{ts}: {e}")
    
    # 保存结果
    with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
        # 转换numpy类型
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        json.dump(convert(all_ablation), f, indent=2)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    linestyles = ['-', '--', '-.']
    
    for i, (pair_name, res) in enumerate(all_ablation.items()):
        # K值
        ks = [r['k'] for r in res['k_curve']]
        aucs = [r['auc'] for r in res['k_curve']]
        axes[0].plot(ks, aucs, color=colors[i], linestyle=linestyles[i], 
                    marker='o', label=pair_name, linewidth=2)
        
        # 维度
        dims = [r['dim_pct'] for r in res['dim_curve']]
        aucs_d = [r['auc'] for r in res['dim_curve']]
        axes[1].plot(dims, aucs_d, color=colors[i], linestyle=linestyles[i],
                    marker='s', label=pair_name, linewidth=2)
        
        # 基准大小
        sizes = [r['size'] for r in res['size_curve']]
        aucs_s = [r['auc'] for r in res['size_curve']]
        stds = [r['std'] for r in res['size_curve']]
        axes[2].errorbar(sizes, aucs_s, yerr=stds, color=colors[i], linestyle=linestyles[i],
                        marker='^', label=pair_name, linewidth=2, capsize=4)
    
    axes[0].set_xlabel('K (Neighbors)'); axes[0].set_ylabel('AUROC')
    axes[0].set_title('K Sensitivity'); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Stable Dim Percentile'); axes[1].set_ylabel('AUROC')
    axes[1].set_title('Dimension Sensitivity'); axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Benchmark Images'); axes[2].set_ylabel('AUROC')
    axes[2].set_title('Benchmark Size Sensitivity'); axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_curves.png"), dpi=200)
    print(f"\n消融曲线保存: {RESULTS_DIR}/ablation_curves.png")