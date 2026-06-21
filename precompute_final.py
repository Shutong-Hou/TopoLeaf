import pickle, os, numpy as np
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
import warnings
warnings.filterwarnings("ignore")

FEATURES_DIR = "features_cache"
CACHE_DIR   = "benchmark_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

BENCHMARK_SPECIES = ["Apple", "Tomato", "Grape", "Strawberry", "Corn_(maize)"]
VARIANCE_PERCENTILE = 50
K_NEIGHBORS = 30
TOPO_K = 15
TOPO_SAMPLE_RATIO = 0.01      # 极低采样，加速预计算

try:
    import umap
    HAS_UMAP = True
except:
    HAS_UMAP = False
    print("未安装 umap，将使用 PCA 代替")

for bench_sp in BENCHMARK_SPECIES:
    print(f"处理 {bench_sp} …")

    # 加载基准特征
    with open(os.path.join(FEATURES_DIR, f"bench_{bench_sp}.pkl"), 'rb') as f:
        bench_feats = pickle.load(f)

    all_healthy = list(bench_feats.values())
    image_paths = list(bench_feats.keys())
    healthy_ref = np.concatenate(all_healthy, axis=0)

    # 稳定维度
    variances = np.var(healthy_ref, axis=0)
    thr = np.percentile(variances, VARIANCE_PERCENTILE)
    sel_dims = np.where(variances <= thr)[0]
    healthy_sel = healthy_ref[:, sel_dims]

    # KNN
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine')
    knn.fit(healthy_sel)

    # 几何评分分布
    all_dists, _ = knn.kneighbors(healthy_sel)
    all_mean_dists = all_dists.mean(axis=1)
    mean_loc = all_mean_dists.mean()
    std_loc = all_mean_dists.std()

    geo_scores = []
    start = 0
    for feat in all_healthy:
        n = feat.shape[0]
        devs = (all_mean_dists[start:start+n] - mean_loc) / std_loc
        geo_scores.append(np.percentile(devs, 90))
        start += n
    geo_scores = np.array(geo_scores)

    # 图像级平均特征（用于 UMAP / 邻居查询）
    img_feats = np.array([f[:, sel_dims].mean(axis=0) for f in all_healthy])

    # UMAP 投影
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_emb = reducer.fit_transform(img_feats)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        umap_emb = reducer.fit_transform(img_feats)

    # 拓扑评分分布（轻量采样）
    topo_scores = []
    for feat in all_healthy:
        fsel = feat[:, sel_dims]
        n_p = fsel.shape[0]
        n_s = max(5, int(n_p * TOPO_SAMPLE_RATIO))   # 约10个patch
        idx = np.random.choice(n_p, n_s, replace=False)
        sc = []
        for patch in fsel[idx]:
            q = patch.reshape(1, -1)
            _, ind = knn.kneighbors(q, n_neighbors=TOPO_K+1)
            nb = healthy_sel[ind[0][:TOPO_K]]
            if len(nb) < 3:
                sc.append(0.0); continue
            try:
                dn = ripser(nb, maxdim=0, n_perm=None)['dgms'][0]
                da = ripser(np.vstack([nb, q]), maxdim=0, n_perm=None)['dgms'][0]
                fn = dn[np.isfinite(dn[:,1])]; fa = da[np.isfinite(da[:,1])]
                mn = fn[:,1].max() if len(fn) else 0
                ma = fa[:,1].max() if len(fa) else 0
                pn = (fn[:,1]-fn[:,0]).mean() if len(fn) else 0
                pa = (fa[:,1]-fa[:,0]).mean() if len(fa) else 0
                sc.append(float(abs(ma-mn)+abs(pa-pn)*5))
            except:
                sc.append(0.0)
        topo_scores.append(np.percentile(sc, 90) if sc else 0.0)
    topo_scores = np.array(topo_scores)

    # 打包
    data = {
        "knn": knn,
        "sel_dims": sel_dims,
        "mean_loc": float(mean_loc),
        "std_loc": float(std_loc),
        "healthy_scores_geo": geo_scores,
        "healthy_scores_topo": topo_scores,
        "image_paths": image_paths,
        "image_level_feats": img_feats,
        "umap_embedding": umap_emb,
        "umap_reducer": reducer if HAS_UMAP else None,
        "pca_reducer": reducer if not HAS_UMAP else None,
    }
    with open(os.path.join(CACHE_DIR, f"{bench_sp}.pkl"), 'wb') as f:
        pickle.dump(data, f)
    print(f"  -> 已缓存，几何评分范围 [{geo_scores.min():.2f}, {geo_scores.max():.2f}]，拓扑评分范围 [{topo_scores.min():.3f}, {topo_scores.max():.3f}]")

print("\n全部完成，可启动增强版 Demo。")