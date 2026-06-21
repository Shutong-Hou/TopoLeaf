import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

FEATURES_DIR = "features_cache"
RESULTS_DIR = "baseline_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BENCHMARK_SPECIES = ["Apple", "Tomato", "Grape", "Strawberry", "Corn_(maize)"]
TEST_SPECIES = ["Apple", "Cherry_(including_sour)", "Corn_(maize)", "Grape", 
                "Peach", "Pepper,_bell", "Potato", "Strawberry", "Tomato"]

DEFAULT_DIM = 50
DEFAULT_K = 30

def load_cached_features(cache_name):
    cache_path = os.path.join(FEATURES_DIR, f"{cache_name}.pkl")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def compute_baselines(bench_sp, test_sp, bench_feats, test_feats, 
                      valid_test_paths, test_labels, sel_dims, knn_model, 
                      mean_loc, std_loc, bench_img_features):
    """对一个组合计算所有基线方法的AUROC"""
    
    # 准备测试图像级特征
    test_img_features = []
    for p in valid_test_paths:
        feat = test_feats.get(p)
        if feat is not None:
            test_img_features.append(feat[:, sel_dims].mean(axis=0))
        else:
            test_img_features.append(np.zeros(len(sel_dims)))
    test_img_features = np.array(test_img_features)
    
    results = {}
    
    # === 1. 我们的几何评分 ===
    all_patches, img_counts = [], []
    for p in valid_test_paths:
        feat = test_feats.get(p)
        if feat is not None:
            all_patches.append(feat[:, sel_dims])
            img_counts.append(len(feat))
        else:
            all_patches.append(np.zeros((1024, len(sel_dims))))
            img_counts.append(1024)
    
    all_patches_stacked = np.concatenate(all_patches, axis=0)
    all_dists, _ = knn_model.kneighbors(all_patches_stacked)
    all_devs = (all_dists.mean(axis=1) - mean_loc) / std_loc
    
    geo_scores, start = [], 0
    for count in img_counts:
        geo_scores.append(np.percentile(all_devs[start:start + count], 90))
        start += count
    results['geo'] = roc_auc_score(test_labels, geo_scores)
    
    # === 2. 马氏距离 ===
    try:
        bench_center = bench_img_features.mean(axis=0)
        bench_cov = np.cov(bench_img_features.T)
        bench_cov_inv = np.linalg.pinv(bench_cov)
        mahalanobis = np.array([np.sqrt((x - bench_center).T @ bench_cov_inv @ (x - bench_center)) 
                                for x in test_img_features])
        results['mahalanobis'] = roc_auc_score(test_labels, mahalanobis)
    except:
        results['mahalanobis'] = 0.5
    
    # === 3. 余弦KNN ===
    cosine_dists = cdist(test_img_features, bench_img_features, metric='cosine')
    cosine_scores = cosine_dists.min(axis=1)
    results['cosine_knn'] = roc_auc_score(test_labels, cosine_scores)
    
    # === 4. Isolation Forest ===
    if_model = IsolationForest(contamination=0.1, random_state=42)
    if_model.fit(bench_img_features)
    if_scores = -if_model.decision_function(test_img_features)
    results['isolation_forest'] = roc_auc_score(test_labels, if_scores)
    
    # === 5. One-Class SVM ===
    ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    ocsvm.fit(bench_img_features)
    ocsvm_scores = -ocsvm.decision_function(test_img_features)
    results['ocsvm'] = roc_auc_score(test_labels, ocsvm_scores)
    
    # === 6. LOF ===
    lof = LocalOutlierFactor(n_neighbors=min(20, len(bench_img_features)-1), 
                             contamination=0.1, novelty=True)
    lof.fit(bench_img_features)
    lof_scores = -lof.decision_function(test_img_features)
    results['lof'] = roc_auc_score(test_labels, lof_scores)
    
    return results

def run_benchmark(bench_sp):
    print(f"\n{'='*60}")
    print(f"基准: {bench_sp}")
    
    bench_feats = load_cached_features(f"bench_{bench_sp}")
    all_bench_paths = list(bench_feats.keys())
    
    # 构建基准特征
    all_healthy = [bench_feats[p] for p in all_bench_paths]
    healthy_ref = np.concatenate(all_healthy, axis=0)
    
    variances = np.var(healthy_ref, axis=0)
    thr = np.percentile(variances, DEFAULT_DIM)
    sel_dims = np.where(variances <= thr)[0]
    healthy_sel = healthy_ref[:, sel_dims]
    
    knn = NearestNeighbors(n_neighbors=DEFAULT_K, metric='cosine')
    knn.fit(healthy_sel)
    
    h_dists = [knn.kneighbors(f[:, sel_dims])[0].mean(axis=1) for f in all_healthy]
    h_dists = np.concatenate(h_dists)
    mean_loc, std_loc = h_dists.mean(), h_dists.std()
    
    bench_img_features = np.array([f[:, sel_dims].mean(axis=0) for f in all_healthy])
    
    all_results = {}
    
    for test_sp in TEST_SPECIES:
        if test_sp == bench_sp:
            continue
        
        try:
            test_feats = load_cached_features(f"test_{test_sp}")
        except FileNotFoundError:
            continue
        
        # 标签
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
            continue
        
        print(f"  {test_sp} ({len(valid_paths)}张)")
        
        res = compute_baselines(bench_sp, test_sp, bench_feats, test_feats,
                               valid_paths, labels, sel_dims, knn, 
                               mean_loc, std_loc, bench_img_features)
        all_results[test_sp] = res
    
    return all_results

if __name__ == '__main__':
    all_baselines = {}
    for bs in BENCHMARK_SPECIES:
        res = run_benchmark(bs)
        all_baselines[bs] = res
        with open(os.path.join(RESULTS_DIR, f"baselines_{bs}.json"), "w") as f:
            json.dump(res, f, indent=2)
    
    with open(os.path.join(RESULTS_DIR, "all_baselines.json"), "w") as f:
        json.dump(all_baselines, f, indent=2)
    
    # 汇总
    methods = ['geo', 'mahalanobis', 'cosine_knn', 'isolation_forest', 'ocsvm', 'lof']
    method_names = ['Geo (Ours)', 'Mahalanobis', 'Cosine KNN', 'Isolation Forest', 'One-Class SVM', 'LOF']
    
    print(f"\n{'='*80}")
    print("基线对比汇总：平均AUROC")
    print(f"{'='*80}")
    
    for mi, method in enumerate(methods):
        aucs = []
        for bs in BENCHMARK_SPECIES:
            for ts in all_baselines.get(bs, {}):
                aucs.append(all_baselines[bs][ts].get(method, 0))
        if aucs:
            print(f"  {method_names[mi]:<20s}: {np.mean(aucs):.4f}")
    
    print(f"\n结果保存至: {RESULTS_DIR}/")