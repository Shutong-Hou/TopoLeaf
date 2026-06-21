import numpy as np
import pickle
import os
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from ripser import ripser
import warnings
warnings.filterwarnings("ignore")

FEATURES_DIR = "features_cache"
RESULTS_DIR = "fusion_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FUSION_PAIRS = [
    ("Apple", "Strawberry"),
    ("Apple", "Corn_(maize)"),
    ("Corn_(maize)", "Apple"),
    ("Apple", "Tomato"),
    ("Corn_(maize)", "Tomato"),
]

K_NEIGHBORS = 30
TOPO_K_NEIGHBORS = 15
TOPO_SAMPLING_RATIO = 0.03
VARIANCE_PERCENTILE = 50

def load_cached_features(cache_name):
    cache_path = os.path.join(FEATURES_DIR, f"{cache_name}.pkl")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def compute_fusion(bench_sp, test_sp):
    print(f"\n{'='*60}")
    print(f"融合实验: {bench_sp} -> {test_sp}")
    
    bench_feats = load_cached_features(f"bench_{bench_sp}")
    all_bench_paths = list(bench_feats.keys())
    all_healthy = [bench_feats[p] for p in all_bench_paths]
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
    
    print(f"  样本: {len(valid_paths)} (H:{labels.count(0)} D:{labels.count(1)})")
    
    # 计算几何评分和拓扑评分
    geo_scores, topo_scores = [], []
    
    for p in valid_paths:
        feat = test_feats.get(p)
        if feat is None:
            geo_scores.append(0); topo_scores.append(0); continue
        
        fsel = feat[:, sel_dims]
        
        # 几何
        ld, _ = knn.kneighbors(fsel)
        geo_scores.append(np.percentile((ld.mean(axis=1) - mean_loc) / std_loc, 90))
        
        # 拓扑
        n_p = fsel.shape[0]
        n_s = max(8, int(n_p * TOPO_SAMPLING_RATIO))
        idx = np.random.choice(n_p, n_s, replace=False)
        sc = []
        for patch in fsel[idx]:
            q = patch.reshape(1, -1)
            _, ind = knn.kneighbors(q, n_neighbors=TOPO_K_NEIGHBORS + 1)
            nb = healthy_sel[ind[0][:TOPO_K_NEIGHBORS]]
            if len(nb) < 3: sc.append(0.0); continue
            try:
                dn = ripser(nb, maxdim=0, n_perm=None)['dgms'][0]
                da = ripser(np.vstack([nb, q]), maxdim=0, n_perm=None)['dgms'][0]
                fn = dn[np.isfinite(dn[:, 1])]; fa = da[np.isfinite(da[:, 1])]
                mn = fn[:, 1].max() if len(fn) else 0
                ma = fa[:, 1].max() if len(fa) else 0
                pn = (fn[:, 1] - fn[:, 0]).mean() if len(fn) else 0
                pa = (fa[:, 1] - fa[:, 0]).mean() if len(fa) else 0
                sc.append(float(abs(ma - mn) + abs(pa - pn) * 5))
            except:
                sc.append(0.0)
        topo_scores.append(np.percentile(sc, 90) if sc else 0.0)
    
    geo_scores = np.array(geo_scores)
    topo_scores = np.array(topo_scores)
    labels_arr = np.array(labels)
    
    # 标准化
    scaler = StandardScaler()
    geo_norm = scaler.fit_transform(geo_scores.reshape(-1, 1)).flatten()
    topo_norm = scaler.fit_transform(topo_scores.reshape(-1, 1)).flatten()
    
    results = {}
    
    # 策略1: 纯几何
    results['geo_only'] = roc_auc_score(labels_arr, geo_scores)
    
    # 策略2: 纯拓扑
    results['topo_only'] = roc_auc_score(labels_arr, topo_scores)
    
    # 策略3: 等权融合
    fusion_equal = geo_norm * 0.5 + topo_norm * 0.5
    results['fusion_equal'] = roc_auc_score(labels_arr, fusion_equal)
    
    # 策略4: 距离门控融合
    # 几何评分落在[健康集90分位, 健康集最大值]之间时启用拓扑
    if sum(labels_arr == 0) > 0:
        healthy_geo = geo_norm[labels_arr == 0]
        low_gate = np.percentile(healthy_geo, 90)
        high_gate = healthy_geo.max()
    else:
        low_gate = np.percentile(geo_norm, 50)
        high_gate = np.percentile(geo_norm, 90)
    
    alpha_gated = np.where(
        (geo_norm >= low_gate) & (geo_norm <= high_gate),
        0.3,  # 灰色地带：拓扑占30%
        0.1   # 清晰地带：拓扑占10%
    )
    fusion_gated = geo_norm * (1 - alpha_gated) + topo_norm * alpha_gated
    results['fusion_gated'] = roc_auc_score(labels_arr, fusion_gated)
    
    # 策略5: 置信度加权融合
    # 几何评分置信度 = 1 - |geo - 健康均值| / 健康范围
    if sum(labels_arr == 0) > 0:
        healthy_geo_raw = geo_scores[labels_arr == 0]
        geo_center = healthy_geo_raw.mean()
        geo_range = healthy_geo_raw.max() - healthy_geo_raw.min()
        if geo_range > 0:
            geo_confidence = 1 - np.minimum(np.abs(geo_scores - geo_center) / geo_range, 1)
        else:
            geo_confidence = np.ones_like(geo_scores)
    else:
        geo_confidence = np.ones_like(geo_scores)
    
    alpha_confident = 0.5 * (1 - geo_confidence)  # 几何越不确定，拓扑权重越大
    fusion_confident = geo_norm * (1 - alpha_confident) + topo_norm * alpha_confident
    results['fusion_confident'] = roc_auc_score(labels_arr, fusion_confident)
    
    for k, v in results.items():
        print(f"  {k:<20s}: {v:.4f}")
    
    return results

if __name__ == '__main__':
    all_results = {}
    for bs, ts in FUSION_PAIRS:
        try:
            res = compute_fusion(bs, ts)
            all_results[f"{bs}_{ts}"] = res
        except Exception as e:
            print(f"错误 {bs}->{ts}: {e}")
    
    with open(os.path.join(RESULTS_DIR, "fusion_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 汇总
    strategies = ['geo_only', 'topo_only', 'fusion_equal', 'fusion_gated', 'fusion_confident']
    strategy_names = ['Geo Only', 'Topo Only', 'Fusion (Equal)', 'Fusion (Gated)', 'Fusion (Confident)']
    
    print(f"\n{'='*60}")
    print("自适应融合汇总")
    print(f"{'='*60}")
    for si, strat in enumerate(strategies):
        aucs = [v[strat] for v in all_results.values()]
        if aucs:
            print(f"  {strategy_names[si]:<20s}: {np.mean(aucs):.4f}")
    
    print(f"\n结果保存至: {RESULTS_DIR}/fusion_results.json")