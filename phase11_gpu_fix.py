import torch
import numpy as np
import cv2
import os
import json
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from ripser import ripser
from torch.utils.data import DataLoader, Dataset
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================
# 配置
# ============================================
PLANTVILLAGE_DIR = "data/PlantVillage/raw/color"
DINO_BACKBONE = "dinov2_vits14"
IMAGE_SIZE = 448
BATCH_SIZE = 16              # 增大批次
NUM_WORKERS = 4              # 多线程加载，GPU不再空等
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANCE_PERCENTILE = 50
K_NEIGHBORS = 30
TOPO_K_NEIGHBORS = 15
TOPO_SAMPLING_RATIO = 0.03
N_BENCHMARK_IMAGES = 100
MAX_TEST_HEALTHY = 100
MAX_TEST_DISEASE_PER_CLASS = 100

BENCHMARK_SPECIES_LIST = ["Apple"]
FEATURES_DIR = "features_cache"
RESULTS_DIR = "results"
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"设备: {DEVICE}, 批量: {BATCH_SIZE}, 线程: {NUM_WORKERS}")

# ============================================
# 1. DINOv2
# ============================================
print(f"加载 DINOv2 ({DINO_BACKBONE})...")
dinov2 = torch.hub.load('facebookresearch/dinov2', DINO_BACKBONE).to(DEVICE)
dinov2.eval()

# ============================================
# 2. 数据集类
# ============================================
class ImageDataset(Dataset):
    def __init__(self, img_paths):
        self.paths = img_paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = (t - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        return t, path

def extract_and_cache(img_paths, cache_name):
    """
    批量提取特征并缓存到磁盘。
    如果缓存文件已存在则直接加载，避免重复计算。
    """
    cache_path = os.path.join(FEATURES_DIR, f"{cache_name}.pkl")
    
    if os.path.exists(cache_path):
        print(f"  加载缓存: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    if len(img_paths) == 0:
        return {}
    
    print(f"  提取 {len(img_paths)} 张图像的特征...")
    ds = ImageDataset(img_paths)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    results = {}
    for i, (imgs, paths) in enumerate(dl):
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            feats = dinov2.forward_features(imgs)['x_norm_patchtokens']
        for j, p in enumerate(paths):
            results[p] = feats[j].cpu().numpy()
        if (i + 1) % 50 == 0:
            print(f"    进度: {(i+1)*BATCH_SIZE}/{len(img_paths)}")
    
    # 保存缓存
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  已缓存至: {cache_path}")
    return results

# ============================================
# 3. 扫描
# ============================================
def scan_plantvillage(base_dir):
    dataset = {}
    base = Path(base_dir)
    for subdir in base.iterdir():
        if not subdir.is_dir(): continue
        name = subdir.name
        if "___" in name:
            sp, cond = name.split("___", 1)
        else:
            sp, cond = name, "unknown"
        imgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.JPG")) + \
               list(subdir.glob("*.png")) + list(subdir.glob("*.PNG"))
        if len(imgs) == 0: continue
        dataset.setdefault(sp, {})[cond] = [str(x) for x in imgs]
    return dataset

print("\n扫描数据集...")
dataset = scan_plantvillage(PLANTVILLAGE_DIR)

# ============================================
# 4. 单基准实验
# ============================================
def run_benchmark(bench_sp, all_dataset):
    results = {}
    
    healthy_all = all_dataset[bench_sp]["healthy"]
    np.random.seed(42)
    bench_paths = np.random.choice(healthy_all, N_BENCHMARK_IMAGES, replace=False).tolist()
    
    print(f"\n{'='*60}")
    print(f"基准: {bench_sp} ({len(bench_paths)} 张健康)")
    print(f"{'='*60}")
    
    # 提取基准特征
    bench_feats = extract_and_cache(bench_paths, f"bench_{bench_sp}")
    all_healthy = [bench_feats[p] for p in bench_paths if p in bench_feats]
    healthy_ref = np.concatenate(all_healthy, axis=0)
    print(f"基准 patch: {healthy_ref.shape[0]}")
    
    variances = np.var(healthy_ref, axis=0)
    thr = np.percentile(variances, VARIANCE_PERCENTILE)
    sel_dims = np.where(variances <= thr)[0]
    healthy_sel = healthy_ref[:, sel_dims]
    print(f"稳定维度: {len(sel_dims)}")
    
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine')
    knn.fit(healthy_sel)
    
    h_dists = []
    for f in all_healthy:
        d, _ = knn.kneighbors(f[:, sel_dims])
        h_dists.append(d.mean(axis=1))
    h_dists = np.concatenate(h_dists)
    mean_loc, std_loc = h_dists.mean(), h_dists.std()
    print(f"健康局部距离: mean={mean_loc:.4f}, std={std_loc:.4f}")
    
    # 测试每个物种
    for test_sp in sorted(all_dataset.keys()):
        if test_sp == bench_sp: continue
        
        paths, labels = [], []
        if "healthy" in all_dataset[test_sp]:
            hl = all_dataset[test_sp]["healthy"]
            if len(hl) > MAX_TEST_HEALTHY:
                np.random.seed(42)
                hl = np.random.choice(hl, MAX_TEST_HEALTHY, replace=False).tolist()
            for p in hl: paths.append(p); labels.append(0)
        for cond, plist in all_dataset[test_sp].items():
            if "healthy" in cond.lower(): continue
            s = plist[:MAX_TEST_DISEASE_PER_CLASS]
            for p in s: paths.append(p); labels.append(1)
        
        if len(set(labels)) < 2 or len(paths) < 10:
            continue
        
        print(f"\n  测试: {test_sp} ({len(paths)} 张, H:{labels.count(0)} D:{labels.count(1)})")
        
        # 提取测试特征（缓存）
        test_feats = extract_and_cache(paths, f"test_{test_sp}")
        
        geo_list, topo_list = [], []
        for i, p in enumerate(paths):
            if (i + 1) % 100 == 0:
                print(f"    进度: {i+1}/{len(paths)}")
            
            feat = test_feats.get(p)
            if feat is None: 
                geo_list.append(0); topo_list.append(0); continue
            
            fsel = feat[:, sel_dims]
            ld, _ = knn.kneighbors(fsel)
            geo_list.append(np.percentile((ld.mean(axis=1) - mean_loc) / std_loc, 90))
            
            n_p = fsel.shape[0]
            n_s = max(8, int(n_p * TOPO_SAMPLING_RATIO))
            idx = np.random.choice(n_p, n_s, replace=False)
            sp_patches = fsel[idx]
            sc = []
            for patch in sp_patches:
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
            topo_list.append(np.percentile(sc, 90) if sc else 0.0)
        
        lab = np.array(labels)
        geo_auc = roc_auc_score(lab, geo_list)
        topo_auc = roc_auc_score(lab, topo_list)
        scaler = StandardScaler()
        gn = scaler.fit_transform(np.array(geo_list).reshape(-1, 1)).flatten()
        tn = scaler.fit_transform(np.array(topo_list).reshape(-1, 1)).flatten()
        fus_auc = roc_auc_score(lab, gn * 0.5 + tn * 0.5)
        
        results[test_sp] = {'geo': geo_auc, 'topo': topo_auc, 'fusion': fus_auc, 'n': len(paths)}
        print(f"    GEO:{geo_auc:.4f} TOPO:{topo_auc:.4f} FUS:{fus_auc:.4f}")
    
    return results

# ============================================
# 5. 运行
# ============================================
if __name__ == '__main__':
    # Windows 多进程必须在此保护块内运行
    all_res = {}
    for bench_sp in BENCHMARK_SPECIES_LIST:
        res = run_benchmark(bench_sp, dataset)
        all_res[bench_sp] = res
        with open(os.path.join(RESULTS_DIR, f"results_{bench_sp}.json"), "w") as f:
            json.dump(res, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(all_res, f, indent=2)

    print(f"\n完成。结果保存于 {RESULTS_DIR}/")