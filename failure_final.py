import numpy as np
import pickle
import os
import json
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
from pathlib import Path
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
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存不存在: {cache_path}")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def analyze_failures(bench_sp, test_sp):
    print(f"\n{'='*60}")
    print(f"分析: {bench_sp} -> {test_sp}")
    
    # 加载基准
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
    
    # 加载测试
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
        print("  无法区分标签，跳过"); return None
    
    print(f"  样本: {len(valid_paths)} (H:{labels.count(0)} D:{labels.count(1)})")
    
    # === 向量化几何评分（一次KNN查询完成所有图像）===
    print("  计算几何评分（向量化）...")
    
    # 堆叠所有图像的patch特征，记录每张图像的patch数
    all_patches = []
    img_patch_counts = []
    for p in valid_paths:
        feat = test_feats.get(p)
        if feat is not None:
            fsel = feat[:, sel_dims]
            all_patches.append(fsel)
            img_patch_counts.append(len(fsel))
        else:
            all_patches.append(np.zeros((1024, len(sel_dims))))
            img_patch_counts.append(1024)
    
    all_patches_stacked = np.concatenate(all_patches, axis=0)
    
    # 一次性查询所有patch的KNN距离
    all_dists, _ = knn.kneighbors(all_patches_stacked)
    all_dists_mean = all_dists.mean(axis=1)
    all_devs = (all_dists_mean - mean_loc) / std_loc
    
    # 按图像分组取P90
    geo_scores = []
    start = 0
    for count in img_patch_counts:
        img_devs = all_devs[start:start + count]
        geo_scores.append(np.percentile(img_devs, 90))
        start += count
    
    geo_scores = np.array(geo_scores)
    labels_arr = np.array(labels)
    
    # 阈值
    geo_threshold = geo_scores[labels_arr == 0].max() if sum(labels_arr == 0) > 0 else np.median(geo_scores)
    
    # 选FP和FN候选
    fp_candidates = np.where((labels_arr == 0) & (geo_scores > geo_threshold))[0]
    fn_candidates = np.where((labels_arr == 1) & (geo_scores < geo_threshold))[0]
    
    n_show = min(4, len(fp_candidates), len(fn_candidates))
    if n_show == 0:
        print("  失败案例不足"); return None
    
    fp_selected = fp_candidates[np.argsort(-geo_scores[fp_candidates])][:n_show]
    fn_selected = fn_candidates[np.argsort(geo_scores[fn_candidates])][:n_show]
    
    # === 只对候选算拓扑评分 ===
    print(f"  对{len(fp_selected)}FP + {len(fn_selected)}FN 计算拓扑...")
    
    def fast_topo(feat_sel):
        n_p = len(feat_sel)
        n_s = max(8, int(n_p * TOPO_SAMPLING_RATIO))
        idx = np.random.choice(n_p, n_s, replace=False)
        sc = []
        for patch in feat_sel[idx]:
            q = patch.reshape(1, -1)
            _, ind = knn.kneighbors(q, n_neighbors=TOPO_K_NEIGHBORS + 1)
            nb = healthy_sel[ind[0][:TOPO_K_NEIGHBORS]]
            if len(nb) < 3: sc.append(0.0); continue
            try:
                dn = ripser(nb, maxdim=0, n_perm=None)['dgms'][0]
                da = ripser(np.vstack([nb, q]), maxdim=0, n_perm=None)['dgms'][0]
                fn_arr = dn[np.isfinite(dn[:, 1])]; fa_arr = da[np.isfinite(da[:, 1])]
                mn = fn_arr[:, 1].max() if len(fn_arr) else 0
                ma = fa_arr[:, 1].max() if len(fa_arr) else 0
                pn = (fn_arr[:, 1] - fn_arr[:, 0]).mean() if len(fn_arr) else 0
                pa = (fa_arr[:, 1] - fa_arr[:, 0]).mean() if len(fa_arr) else 0
                sc.append(float(abs(ma - mn) + abs(pa - pn) * 5))
            except: sc.append(0.0)
        return np.percentile(sc, 90) if sc else 0.0
    
    topo_vals = {}
    for idx in np.concatenate([fp_selected, fn_selected]):
        feat = test_feats.get(valid_paths[idx])
        if feat is not None:
            topo_vals[idx] = fast_topo(feat[:, sel_dims])
    
    # 可视化
    fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))
    fig.suptitle(f"Failures: {bench_sp} -> {test_sp}", fontsize=14)
    
    for i, idx in enumerate(fp_selected):
        img = cv2.imread(valid_paths[idx])
        if img is not None:
            axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tv = topo_vals.get(idx, 0)
        axes[0, i].set_title(f"FP(Healthy) Geo={geo_scores[idx]:.1f}σ Topo={tv:.2f}", fontsize=8)
        axes[0, i].axis('off')
    
    for i, idx in enumerate(fn_selected):
        img = cv2.imread(valid_paths[idx])
        if img is not None:
            axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tv = topo_vals.get(idx, 0)
        axes[1, i].set_title(f"FN(Disease) Geo={geo_scores[idx]:.1f}σ Topo={tv:.2f}", fontsize=8)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(FAILURE_DIR, f"failures_{bench_sp}_{test_sp}.png")
    plt.savefig(save_path, dpi=150)
    print(f"  保存: {save_path}")
    plt.close()
    
    return {'bench': bench_sp, 'test': test_sp, 'n': len(valid_paths),
            'fp': len(fp_candidates), 'fn': len(fn_candidates)}

if __name__ == '__main__':
    all_stats = []
    for bs, ts in KEY_PAIRS:
        try:
            s = analyze_failures(bs, ts)
            if s: all_stats.append(s)
        except Exception as e:
            print(f"错误 {bs}->{ts}: {e}")
    
    print(f"\n汇总:")
    for s in all_stats:
        print(f"  {s['bench']}->{s['test']}: 总{s['n']} FP={s['fp']} FN={s['fn']}")
    print(f"\n图像保存至: {FAILURE_DIR}/")