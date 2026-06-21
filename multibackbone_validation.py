import torch
import numpy as np
import cv2
import os
import json
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import warnings
from pathlib import Path
import clip

warnings.filterwarnings("ignore")

# ============================================
# 配置
# ============================================
PLANTVILLAGE_DIR = "data/PlantVillage/raw/color"
IMAGE_SIZE = 448  # DINOv2标准
IMAGE_SIZE_CLIP = 224  # CLIP标准
BATCH_SIZE = 16
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANCE_PERCENTILE = 50
K_NEIGHBORS = 30

# 代表性组合
REPRESENTATIVE_PAIRS = [
    ("Apple", "Strawberry"),
    ("Apple", "Corn_(maize)"),
    ("Corn_(maize)", "Apple"),
    ("Tomato", "Strawberry"),
    ("Grape", "Peach"),
    ("Strawberry", "Tomato"),
    ("Tomato", "Corn_(maize)"),
    ("Apple", "Potato"),
    ("Grape", "Apple"),
    ("Corn_(maize)", "Tomato"),
]

FEATURES_DIR = "features_cache"
MULTI_BACKBONE_DIR = "features_multibackbone"
RESULTS_DIR = "results_multibackbone"
os.makedirs(MULTI_BACKBONE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"设备: {DEVICE}")

# ============================================
# 1. 加载 DINOv2 + CLIP + ResNet-50
# ============================================
print("加载 DINOv2...")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
dinov2.eval()

print("加载 CLIP ViT-B/32...")
import open_clip
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(DEVICE)
clip_model.eval()

print("加载 ResNet-50...")
import torchvision.models as models
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # 去掉分类头
resnet = resnet.to(DEVICE)
resnet.eval()

# ============================================
# 2. 数据集类
# ============================================
class ImageDatasetDINO(Dataset):
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

class ImageDatasetCLIP(Dataset):
    def __init__(self, img_paths):
        self.paths = img_paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMAGE_SIZE_CLIP, IMAGE_SIZE_CLIP, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE_CLIP, IMAGE_SIZE_CLIP))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # CLIP标准化
        t = (t - torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)) / torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)
        return t, path

class ImageDatasetResNet(Dataset):
    def __init__(self, img_paths):
        self.paths = img_paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMAGE_SIZE_CLIP, IMAGE_SIZE_CLIP, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE_CLIP, IMAGE_SIZE_CLIP))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = (t - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        return t, path

# ============================================
# 3. 特征提取函数（三个骨干网络）
# ============================================
def extract_dino_features(img_paths, cache_name):
    cache_path = os.path.join(MULTI_BACKBONE_DIR, f"dino_{cache_name}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    if len(img_paths) == 0:
        return {}
    print(f"  提取 DINOv2 特征: {len(img_paths)} 张")
    ds = ImageDatasetDINO(img_paths)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    results = {}
    for imgs, paths in dl:
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            feats = dinov2.forward_features(imgs)['x_norm_patchtokens']
        for j, p in enumerate(paths):
            results[p] = feats[j].cpu().numpy()
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    return results

def extract_clip_features(img_paths, cache_name):
    cache_path = os.path.join(MULTI_BACKBONE_DIR, f"clip_{cache_name}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    if len(img_paths) == 0:
        return {}
    print(f"  提取 CLIP 特征: {len(img_paths)} 张")
    ds = ImageDatasetCLIP(img_paths)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    results = {}
    for imgs, paths in dl:
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            feats = clip_model.encode_image(imgs)
            # CLIP输出 [B, 512]，扩展为伪patch特征 [B, 1, 512]
            feats = feats.unsqueeze(1)
        for j, p in enumerate(paths):
            results[p] = feats[j].cpu().numpy()
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    return results

def extract_resnet_features(img_paths, cache_name):
    cache_path = os.path.join(MULTI_BACKBONE_DIR, f"resnet_{cache_name}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    if len(img_paths) == 0:
        return {}
    print(f"  提取 ResNet-50 特征: {len(img_paths)} 张")
    ds = ImageDatasetResNet(img_paths)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    results = {}
    for imgs, paths in dl:
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            feats = resnet(imgs)  # [B, 2048, 1, 1]
            feats = feats.view(feats.size(0), 1, -1)  # [B, 1, 2048]
        for j, p in enumerate(paths):
            results[p] = feats[j].cpu().numpy()
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    return results

# ============================================
# 4. 扫描数据集
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

print("扫描数据集...")
dataset = scan_plantvillage(PLANTVILLAGE_DIR)

# ============================================
# 5. 单骨干几何评分
# ============================================
def compute_geometric_auroc_backbone(bench_paths, bench_feats_dict, test_paths, 
                                     test_feats_dict, test_labels, dim_percentile, k_neighbors):
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
    
    return roc_auc_score(test_labels, geo_scores)

# ============================================
# 6. 运行跨骨干实验
# ============================================
def run_multibackbone_experiment(bench_sp, test_sp):
    print(f"\n{'='*60}")
    print(f"跨骨干: {bench_sp} -> {test_sp}")
    
    # 获取图像路径
    bench_healthy = dataset[bench_sp]["healthy"]
    np.random.seed(42)
    bench_paths = np.random.choice(bench_healthy, 100, replace=False).tolist()
    
    test_data = dataset[test_sp]
    disease_kw = ["scab", "rot", "rust", "blight", "spot", "mildew", 
                  "virus", "scorch", "mold", "mite", "curl", "mosaic", "esca"]
    
    test_paths, test_labels = [], []
    if "healthy" in test_data:
        hpaths = test_data["healthy"]
        if len(hpaths) > 100:
            np.random.seed(42)
            hpaths = np.random.choice(hpaths, 100, replace=False).tolist()
        for p in hpaths:
            test_paths.append(p)
            test_labels.append(0)
    
    for cond, paths in test_data.items():
        if "healthy" in cond.lower():
            continue
        spaths = paths[:100]
        for p in spaths:
            test_paths.append(p)
            test_labels.append(1)
    
    print(f"  测试: {len(test_paths)} (H:{test_labels.count(0)} D:{test_labels.count(1)})")
    
    results = {}
    
    # DINOv2
    bench_feats_dino = extract_dino_features(bench_paths, f"bench_{bench_sp}")
    test_feats_dino = extract_dino_features(test_paths, f"test_{test_sp}")
    results['dino'] = compute_geometric_auroc_backbone(
        bench_paths, bench_feats_dino, test_paths, test_feats_dino, 
        test_labels, VARIANCE_PERCENTILE, K_NEIGHBORS
    )
    
    # CLIP
    bench_feats_clip = extract_clip_features(bench_paths, f"bench_{bench_sp}")
    test_feats_clip = extract_clip_features(test_paths, f"test_{test_sp}")
    results['clip'] = compute_geometric_auroc_backbone(
        bench_paths, bench_feats_clip, test_paths, test_feats_clip,
        test_labels, VARIANCE_PERCENTILE, K_NEIGHBORS
    )
    
    # ResNet-50
    bench_feats_resnet = extract_resnet_features(bench_paths, f"bench_{bench_sp}")
    test_feats_resnet = extract_resnet_features(test_paths, f"test_{test_sp}")
    results['resnet'] = compute_geometric_auroc_backbone(
        bench_paths, bench_feats_resnet, test_paths, test_feats_resnet,
        test_labels, VARIANCE_PERCENTILE, K_NEIGHBORS
    )
    
    print(f"  DINOv2: {results['dino']:.4f} | CLIP: {results['clip']:.4f} | ResNet-50: {results['resnet']:.4f}")
    return results

if __name__ == '__main__':
    all_results = {}
    for bs, ts in REPRESENTATIVE_PAIRS:
        try:
            res = run_multibackbone_experiment(bs, ts)
            all_results[f"{bs}_{ts}"] = res
        except Exception as e:
            print(f"错误 {bs}->{ts}: {e}")
    
    with open(os.path.join(RESULTS_DIR, "multibackbone_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 汇总
    dino_aucs = [v['dino'] for v in all_results.values()]
    clip_aucs = [v['clip'] for v in all_results.values()]
    resnet_aucs = [v['resnet'] for v in all_results.values()]
    
    print(f"\n{'='*60}")
    print("跨骨干网络验证汇总")
    print(f"{'='*60}")
    print(f"  DINOv2:     {np.mean(dino_aucs):.4f} ({len(dino_aucs)} 组合)")
    print(f"  CLIP ViT-B: {np.mean(clip_aucs):.4f} ({len(clip_aucs)} 组合)")
    print(f"  ResNet-50:  {np.mean(resnet_aucs):.4f} ({len(resnet_aucs)} 组合)")
    print(f"\n结果保存至: {RESULTS_DIR}/multibackbone_results.json")