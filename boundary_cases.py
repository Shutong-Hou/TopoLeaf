import numpy as np
import pickle
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
import warnings
warnings.filterwarnings("ignore")

FEATURES_DIR = "features_cache"
FAILURE_DIR = "failure_analysis"
os.makedirs(FAILURE_DIR, exist_ok=True)

KEY_PAIRS = [
    ("Apple", "Corn_(maize)"),
    ("Apple", "Strawberry"),
    ("Corn_(maize)", "Apple"),
    ("Tomato", "Corn_(maize)"),
]

K_NEIGHBORS = 30
TOPO_K_NEIGHBORS = 15
TOPO_SAMPLING_RATIO = 0.03
VARIANCE_PERCENTILE = 50

def load_cached_features(cache_name):
    cache_path = os.path.join(FEATURES_DIR, f"{cache_name}.pkl")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def analyze_boundary_cases(bench_sp, test_sp):
    print(f"\n{'='*60}")
    print(f"边界案例分析: {bench_sp} -> {test_sp}")
    
    bench_feats = load_cached_features(f"bench_{bench_sp}")
    bench_paths = list(bench_feats.keys())
    all_healthy = [bench_feats[p] for p in bench_paths]
    healthy_ref = np.concatenate(all_healthy, axis=0)
    
    variances = np.var(healthy_ref, axis=0)
    thr = np.percentile(variances, VARIANCE_PERCENTILE)
    sel_dims = np.where(variances <= thr)[0]
    healthy_sel = healthy_ref[:, sel_dims]
    
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine')
    knn.fit(healthy_sel)
    
    h_dists = [knn.kneighbors(f[:, sel_dims])[0].mean(axis=1) for f in all_healthy]
    h_dists = np.concatenate(h_dists)
    mean_loc, std_loc = h_dists.mean(), h_dists.std()
    
    test_feats = load_cached_features(f"test_{test_sp}")
    
    disease_kw = ["scab", "rot", "rust", "blight", "spot", "mildew", 
                  "virus", "scorch", "mold", "mite", "curl", "mosaic", "esca"]
    
    valid_paths, labels = [], []
    for p in test_feats.keys():
        pl = p.lower()
        if "healthy" in pl:
            valid_paths.append(p); labels.append(0)
        elif any(d in pl for d in disease_kw):
            valid_paths.append(p); labels.append(1)
    
    if len(set(labels)) < 2:
        print("  标签不足"); return None
    
    print(f"  样本: {len(valid_paths)} (H:{labels.count(0)} D:{labels.count(1)})")
    
    # 向量化几何评分
    all_patches, img_patch_counts = [], []
    for p in valid_paths:
        feat = test_feats.get(p)
        if feat is not None:
            all_patches.append(feat[:, sel_dims])
            img_patch_counts.append(len(feat))
        else:
            all_patches.append(np.zeros((1024, len(sel_dims))))
            img_patch_counts.append(1024)
    
    all_patches_stacked = np.concatenate(all_patches, axis=0)
    all_dists, _ = knn.kneighbors(all_patches_stacked)
    all_devs = (all_dists.mean(axis=1) - mean_loc) / std_loc
    
    geo_scores, start = [], 0
    for count in img_patch_counts:
        geo_scores.append(np.percentile(all_devs[start:start + count], 90))
        start += count
    geo_scores = np.array(geo_scores)
    labels_arr = np.array(labels)
    
    # 直接取：健康中评分最高的4张 + 病害中评分最低的4张
    healthy_idx = np.where(labels_arr == 0)[0]
    disease_idx = np.where(labels_arr == 1)[0]
    
    n_show = min(4, len(healthy_idx), len(disease_idx))
    
    fp_selected = healthy_idx[np.argsort(-geo_scores[healthy_idx])][:n_show]
    fn_selected = disease_idx[np.argsort(geo_scores[disease_idx])][:n_show]
    
    print(f"  最难健康样本 Geo: {[f'{geo_scores[i]:.1f}' for i in fp_selected]}")
    print(f"  最难病害样本 Geo: {[f'{geo_scores[i]:.1f}' for i in fn_selected]}")
    
    # 可视化
    fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 9))
    fig.suptitle(f"Boundary Cases: {bench_sp} -> {test_sp}", fontsize=14)
    
    for i, idx in enumerate(fp_selected):
        img = cv2.imread(valid_paths[idx])
        if img is not None:
            axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Hardest Healthy #{i+1}\nGeo={geo_scores[idx]:.2f}σ", fontsize=9)
        axes[0, i].axis('off')
    
    for i, idx in enumerate(fn_selected):
        img = cv2.imread(valid_paths[idx])
        if img is not None:
            axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"Hardest Disease #{i+1}\nGeo={geo_scores[idx]:.2f}σ", fontsize=9)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(FAILURE_DIR, f"boundary_{bench_sp}_{test_sp}.png")
    plt.savefig(save_path, dpi=150)
    print(f"  保存: {save_path}")
    plt.close()
    
    return {'bench': bench_sp, 'test': test_sp, 'n': len(valid_paths),
            'hardest_healthy': [float(geo_scores[i]) for i in fp_selected],
            'hardest_disease': [float(geo_scores[i]) for i in fn_selected]}

if __name__ == '__main__':
    all_stats = []
    for bs, ts in KEY_PAIRS:
        try:
            s = analyze_boundary_cases(bs, ts)
            if s: all_stats.append(s)
        except Exception as e:
            print(f"错误 {bs}->{ts}: {e}")
    
    print(f"\n{'='*60}")
    print("边界案例汇总")
    print(f"{'='*60}")
    for s in all_stats:
        print(f"{s['bench']}->{s['test']}:")
        print(f"  最难健康: {[f'{v:.2f}σ' for v in s['hardest_healthy']]}")
        print(f"  最难病害: {[f'{v:.2f}σ' for v in s['hardest_disease']]}")
    print(f"\n图像保存至: {FAILURE_DIR}/")