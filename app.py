import streamlit as st
import torch
import numpy as np
import cv2
import pickle
import os
import json
import time
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
from PIL import Image

# ------------------ 中文字体 ------------------
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ------------------ 配置 ------------------
FEATURES_DIR = "features_cache"
RESULTS_DIR = "results"
CACHE_DIR = "benchmark_cache"
BENCHMARK_SPECIES = ["Apple", "Tomato", "Grape", "Strawberry", "Corn_(maize)"]
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOPO_K = 15
TOPO_SAMPLE_RATIO = 0.01

# ------------------ 全局样式 ------------------
st.markdown("""
<style>
    .stDataFrame td { color: #1a1a1a !important; font-weight: 500 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    div[data-testid="stTabs"] button { font-size: 0.85rem !important; padding: 6px 12px !important; }
    .stTabs [data-testid="stImage"] img { max-height: 280px !important; object-fit: contain !important; }
    .loading-placeholder {
        display: flex; align-items: center; justify-content: center;
        height: 300px; background: #f8f9fa; border-radius: 12px;
        border: 2px dashed #dee2e6; color: #868e96; font-size: 1.1rem;
    }
    .footer {
        text-align: center; padding: 20px 0; color: #636e72;
    }
    .footer a {
        text-decoration: none; color: #2c3e50; font-weight: 500;
    }
    .footer img {
        vertical-align: middle; margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ 翻译表 ------------------
TEXTS = {
    "zh": {
        "title": "TopoLeaf：零样本开放集植物病害检测",
        "subtitle": "几何异常评分 · 拓扑异常评分 · 跨物种泛化 · 特征空间定位",
        "settings": "设置",
        "bench": "健康基准物种",
        "mode": "检测模式",
        "single": "单张图像",
        "batch": "批量对比",
        "note": "上传叶片图像，系统给出几何/拓扑异常评分、在基准分布中的位置、最相似健康叶片及跨物种性能矩阵。",
        "upload_single": "选择一张叶片图片",
        "upload_batch": "选择多张叶片图片",
        "score_geo": "几何评分",
        "score_topo": "拓扑评分",
        "unit_geo": "σ",
        "unit_topo": "",
        "pct_geo": "高于 {:.0f}% 健康基准",
        "pct_topo": "高于 {:.0f}% 健康基准",
        "heatmap": "异常热力图",
        "neighbors": "最相似健康叶片",
        "tab_geo_hist": "几何分布",
        "tab_topo_hist": "拓扑分布",
        "tab_umap": "空间定位",
        "tab_report": "综合报告",
        "tab_explain": "热力解读",
        "hist_geo": "几何评分分布",
        "hist_topo": "拓扑评分分布",
        "umap_title": "特征空间定位",
        "cross_title": "跨物种零样本异常检测性能 (AUROC)",
        "cross_caption": "几何 AUROC，100 张健康基准 / 100 健康测试 / 每类病害 100。",
        "language": "语言",
        "loading": "正在分析叶片图像，请稍候...",
        "placeholder": "📤 请上传一张叶片图像以开始检测",
        "report_title": "📋 综合诊断报告",
        "report_risk": "整体风险等级",
        "report_risk_high": "🔴 高度异常",
        "report_risk_mid": "🟡 中度偏离",
        "report_risk_low": "🟢 接近健康",
        "report_geo_note": "几何评分基于局部特征距离，反映纹理与结构的整体偏离程度。",
        "report_topo_note": "拓扑评分基于局部持续同调，捕捉特征空间中的结构断裂。",
        "report_neighbor_note": "当前图像与健康基准中最相似的 5 张叶片进行对比，相似度越高说明越接近健康状态。",
        "report_cross_note": "跨物种矩阵显示当前基准物种对其他物种的泛化检测能力，可用于评估结果可信度。",
        "explain_title": "🔬 热力图解读",
        "explain_text": "热力图高亮区域（暖色）表示该处 patch 特征与健康基准存在显著偏离，是潜在病斑位置。冷色区域表示与健康基准一致。热力图不依赖任何标注数据，完全基于健康基准的统计分布生成。",
    },
    "en": {
        "title": "TopoLeaf: Zero-Shot Open-Set Plant Disease Detection",
        "subtitle": "Geometric · Topological · Cross-Species · Feature Space",
        "settings": "Settings",
        "bench": "Benchmark Species",
        "mode": "Mode",
        "single": "Single Image",
        "batch": "Batch Comparison",
        "note": "Upload leaf images to get geometric/topological scores, benchmark position, similar healthy leaves, and cross-species matrix.",
        "upload_single": "Choose a leaf image",
        "upload_batch": "Choose multiple leaf images",
        "score_geo": "Geo Score",
        "score_topo": "Topo Score",
        "unit_geo": "σ",
        "unit_topo": "",
        "pct_geo": "Above {:.0f}% healthy",
        "pct_topo": "Above {:.0f}% healthy",
        "heatmap": "Anomaly Heatmap",
        "neighbors": "Most Similar Healthy Leaves",
        "tab_geo_hist": "Geo Distribution",
        "tab_topo_hist": "Topo Distribution",
        "tab_umap": "Feature Space",
        "tab_report": "Report",
        "tab_explain": "Heatmap Guide",
        "hist_geo": "Geometric Score Distribution",
        "hist_topo": "Topological Score Distribution",
        "umap_title": "Feature Space Location",
        "cross_title": "Cross-Species Zero-Shot Anomaly Detection (AUROC)",
        "cross_caption": "Geometric AUROC, 100 healthy benchmark / 100 healthy test / 100 per disease class.",
        "language": "Language",
        "loading": "Analyzing leaf image, please wait...",
        "placeholder": "📤 Please upload a leaf image to start detection",
        "report_title": "📋 Diagnostic Report",
        "report_risk": "Risk Level",
        "report_risk_high": "🔴 High Anomaly",
        "report_risk_mid": "🟡 Moderate Deviation",
        "report_risk_low": "🟢 Near Healthy",
        "report_geo_note": "Geometric score measures overall texture and structural deviation based on local feature distances.",
        "report_topo_note": "Topological score captures structural breaks in feature space using local persistent homology.",
        "report_neighbor_note": "The current image is compared with the 5 most similar healthy benchmark leaves. Higher similarity suggests a healthier state.",
        "report_cross_note": "The cross-species matrix shows detection generalizability of the current benchmark, useful for confidence assessment.",
        "explain_title": "🔬 Heatmap Interpretation",
        "explain_text": "Highlighted regions (warm colors) indicate patches whose features deviate significantly from the healthy benchmark, marking potential disease areas. Cool regions align with the benchmark. The heatmap requires no labeled data and is generated purely from the statistical distribution of healthy samples.",
    }
}

# ------------------ 加载模型 ------------------
@st.cache_resource
def load_dino():
    import warnings
    import os
    warnings.filterwarnings("ignore")
    local_dir = os.path.join(
        os.path.expanduser("~"), 
        ".cache", "torch", "hub", "facebookresearch_dinov2_main"
    )
    # 如果本地有 DINOv2 源码，优先本地加载，否则在线下载
    if os.path.exists(local_dir):
        model = torch.hub.load(local_dir, 'dinov2_vits14', source='local', pretrained=True).to(DEVICE)
    else:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True).to(DEVICE)
    model.eval()
    return model

# ------------------ 加载基准 ------------------
@st.cache_resource
def load_bench(bench_sp):
    with open(os.path.join(CACHE_DIR, f"{bench_sp}.pkl"), 'rb') as f:
        data = pickle.load(f)
    return (data["knn"], data["sel_dims"], data["mean_loc"], data["std_loc"],
            data["healthy_scores_geo"], data["healthy_scores_topo"],
            data["image_paths"], data["umap_embedding"],
            data.get("umap_reducer"), data.get("pca_reducer"),
            data["image_level_feats"])

# ------------------ 特征提取 ------------------
def extract_features(image, model):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255.0
    tensor = (tensor - torch.tensor([0.485,0.456,0.406]).view(3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = model.forward_features(tensor)['x_norm_patchtokens']
    return feats.squeeze(0).cpu().numpy()

# ------------------ 几何评分 ------------------
def geometric_score(feats, knn, sel_dims, mean_loc, std_loc):
    fsel = feats[:, sel_dims]
    dists, _ = knn.kneighbors(fsel)
    dev = (dists.mean(axis=1) - mean_loc) / std_loc
    return np.percentile(dev, 90), dev

# ------------------ 拓扑评分 ------------------
def topological_score(feats, knn, sel_dims, healthy_sel_all):
    fsel = feats[:, sel_dims]
    n_p = fsel.shape[0]
    n_s = max(5, int(n_p * TOPO_SAMPLE_RATIO))
    idx = np.random.choice(n_p, n_s, replace=False)
    sc = []
    for patch in fsel[idx]:
        q = patch.reshape(1, -1)
        _, ind = knn.kneighbors(q, n_neighbors=TOPO_K+1)
        nb = healthy_sel_all[ind[0][:TOPO_K]]
        if len(nb) < 3: sc.append(0.0); continue
        try:
            dn = ripser(nb, maxdim=0, n_perm=None)['dgms'][0]
            da = ripser(np.vstack([nb, q]), maxdim=0, n_perm=None)['dgms'][0]
            fn = dn[np.isfinite(dn[:,1])]; fa = da[np.isfinite(da[:,1])]
            mn = fn[:,1].max() if len(fn) else 0
            ma = fa[:,1].max() if len(fa) else 0
            pn = (fn[:,1]-fn[:,0]).mean() if len(fn) else 0
            pa = (fa[:,1]-fa[:,0]).mean() if len(fa) else 0
            sc.append(float(abs(ma-mn)+abs(pa-pn)*5))
        except: sc.append(0.0)
    return np.percentile(sc, 90) if sc else 0.0

# ------------------ 热力图 ------------------
def heatmap(feats, knn, sel_dims, mean_loc, std_loc, orig_img):
    fsel = feats[:, sel_dims]
    dists, _ = knn.kneighbors(fsel)
    dev = (dists.mean(axis=1) - mean_loc) / std_loc
    hmap = dev.reshape(int(np.sqrt(len(dev))), -1)
    hmap = np.clip(hmap, 0, None)
    hmap = cv2.resize(hmap, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    hmap = cv2.GaussianBlur(hmap, (21,21), 0)
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    colored = cv2.applyColorMap((hmap*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_img, 0.4, colored, 0.6, 0)
    return overlay, hmap

# ------------------ 邻居 ------------------
def similar_neighbors(img_feat, image_paths, image_level_feats, topk=5):
    dists = np.linalg.norm(image_level_feats - img_feat, axis=1)
    idx = np.argsort(dists)[:topk]
    return [image_paths[i] for i in idx], dists[idx]

# ------------------ 跨物种矩阵 ------------------
def load_matrix():
    matrix = {}
    for bench in BENCHMARK_SPECIES:
        path = os.path.join(RESULTS_DIR, f"results_{bench}.json")
        if os.path.exists(path):
            with open(path) as f:
                matrix[bench] = json.load(f)
    return matrix

# ==================== UI ====================
st.set_page_config(page_title="TopoLeaf", layout="wide")
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

col_t, col_l = st.columns([5, 1])
with col_l:
    lang = st.radio("", ["中文", "English"], index=0, label_visibility="collapsed")
    st.session_state.lang = "zh" if lang == "中文" else "en"
T = TEXTS[st.session_state.lang]

st.title(T["title"])
st.caption(T["subtitle"])

with st.sidebar:
    st.header(T["settings"])
    bench = st.selectbox(T["bench"], BENCHMARK_SPECIES, index=0)
    mode = st.radio(T["mode"], [T["single"], T["batch"]])
    st.divider()
    st.caption(T["note"])

model = load_dino()
knn, sel_dims, mean_loc, std_loc, geo_healthy, topo_healthy, img_paths, umap_emb, umap_red, pca_red, img_feats = load_bench(bench)
with open(os.path.join(FEATURES_DIR, f"bench_{bench}.pkl"), 'rb') as f:
    healthy_all_sel = np.concatenate([v[:, sel_dims] for v in pickle.load(f).values()], axis=0)

# ==================== 单张模式 ====================
if mode == T["single"]:
    file = st.file_uploader(T["upload_single"], type=["jpg", "jpeg", "png"])

    if not file:
        col_empty, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            st.markdown(
                f'<div class="loading-placeholder">{T["placeholder"]}</div>',
                unsafe_allow_html=True
            )
    else:
        with st.spinner(T["loading"]):
            image = Image.open(file).convert("RGB")
            feats = extract_features(image, model)
            geo, dev = geometric_score(feats, knn, sel_dims, mean_loc, std_loc)
            topo = topological_score(feats, knn, sel_dims, healthy_all_sel)
            geo_pct = (geo_healthy < geo).mean() * 100
            topo_pct = (topo_healthy < topo).mean() * 100
            orig_bgr = cv2.cvtColor(np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))), cv2.COLOR_RGB2BGR)
            over, _ = heatmap(feats, knn, sel_dims, mean_loc, std_loc, orig_bgr)
            img_feat = feats[:, sel_dims].mean(axis=0)
            neighbors, neighbor_dists = similar_neighbors(img_feat, img_paths, img_feats, topk=5)
            if geo_pct >= 95:
                risk_level = T["report_risk_high"]
            elif geo_pct >= 70:
                risk_level = T["report_risk_mid"]
            else:
                risk_level = T["report_risk_low"]
            time.sleep(0.3)

        c_img, c_score, c_heat = st.columns([0.25, 0.20, 0.55])
        with c_img:
            st.image(image, use_container_width=True)
        with c_score:
            st.metric(T["score_geo"], f"{geo:.2f} {T.get('unit_geo', 'σ')}")
            st.caption(T["pct_geo"].format(geo_pct))
            st.divider()
            st.metric(T["score_topo"], f"{topo:.3f} {T.get('unit_topo', '')}")
            st.caption(T["pct_topo"].format(topo_pct))
        with c_heat:
            st.image(over, caption=T["heatmap"], use_container_width=True)

        st.subheader("📊 分析面板", divider="gray")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            T["tab_geo_hist"], T["tab_topo_hist"], T["tab_umap"],
            T["tab_report"], T["tab_explain"]
        ])

        with tab1:
            fig, ax = plt.subplots(figsize=(4, 2.2))
            ax.hist(geo_healthy, bins=25, color='#3498db', alpha=0.85, edgecolor='white')
            ax.axvline(geo, color='#e74c3c', linestyle='--', linewidth=2, label=f'当前: {geo:.2f}σ')
            ax.set_title(T["hist_geo"], fontsize=10)
            ax.legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            fig, ax = plt.subplots(figsize=(4, 2.2))
            ax.hist(topo_healthy, bins=25, color='#e67e22', alpha=0.85, edgecolor='white')
            ax.axvline(topo, color='#e74c3c', linestyle='--', linewidth=2, label=f'当前: {topo:.3f}')
            ax.set_title(T["hist_topo"], fontsize=10)
            ax.legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            reducer = umap_red if umap_red is not None else pca_red
            if reducer is not None:
                fig, ax = plt.subplots(figsize=(4, 2.8))
                sc = ax.scatter(umap_emb[:,0], umap_emb[:,1], c=geo_healthy, cmap='viridis', s=10, alpha=0.8)
                plt.colorbar(sc, ax=ax, label='Geo Score (σ)', shrink=0.8)
                try:
                    new_pt = reducer.transform(img_feat.reshape(1, -1))
                    ax.scatter(new_pt[0,0], new_pt[0,1], c='red', marker='*', s=250, edgecolors='white', linewidth=2, label='当前图像')
                    ax.legend(fontsize=7)
                except:
                    pass
                ax.set_title(T["umap_title"], fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

        with tab4:
            st.subheader(T["report_title"])
            st.markdown(f"**{T['report_risk']}：{risk_level}**")
            st.markdown(f"- {T['score_geo']}：{geo:.2f} {T.get('unit_geo', 'σ')}（{T['pct_geo'].format(geo_pct)}）")
            st.markdown(f"- {T['score_topo']}：{topo:.3f}（{T['pct_topo'].format(topo_pct)}）")
            st.markdown(f"- {T['neighbors']}：最近邻距离 {neighbor_dists[0]:.4f}")
            st.caption(T["report_geo_note"])
            st.caption(T["report_topo_note"])
            st.caption(T["report_neighbor_note"])
            st.caption(T["report_cross_note"])

        with tab5:
            st.subheader(T["explain_title"])
            st.markdown(T["explain_text"])

        st.subheader("🔍 " + T["neighbors"], divider="gray")
        cols = st.columns(5)
        for i, path in enumerate(neighbors):
            with cols[i]:
                st.image(Image.open(path).resize((120, 120)), caption=f"距离: {neighbor_dists[i]:.4f}")

# ==================== 批量模式 ====================
else:
    files = st.file_uploader(T["upload_batch"], type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if not files:
        col_empty, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            st.markdown(
                f'<div class="loading-placeholder">{T["placeholder"]}</div>',
                unsafe_allow_html=True
            )
    else:
        with st.spinner(T["loading"]):
            results = []
            for file in files:
                image = Image.open(file).convert("RGB")
                feats = extract_features(image, model)
                geo, _ = geometric_score(feats, knn, sel_dims, mean_loc, std_loc)
                topo = topological_score(feats, knn, sel_dims, healthy_all_sel)
                orig_bgr = cv2.cvtColor(np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))), cv2.COLOR_RGB2BGR)
                over, _ = heatmap(feats, knn, sel_dims, mean_loc, std_loc, orig_bgr)
                results.append((image, geo, topo, over))
            time.sleep(0.3)

        cols = st.columns(min(4, len(results)))
        for i, (img, geo, topo, over) in enumerate(results):
            with cols[i % 4]:
                st.image(img, caption=f"Geo: {geo:.2f}σ | Topo: {topo:.3f}", use_container_width=True)
                st.image(over, use_container_width=True)

# ==================== 底部跨物种矩阵 ====================
st.divider()
st.header(T["cross_title"])
matrix = load_matrix()
if matrix:
    species_order = ["Cherry_(including_sour)", "Corn_(maize)", "Grape", "Peach",
                     "Pepper,_bell", "Potato", "Strawberry", "Tomato"]
    bench_list = [b for b in BENCHMARK_SPECIES if b in matrix]
    table = {}
    for sp in species_order:
        row = []
        for b in bench_list:
            row.append(round(matrix[b][sp]['geo'], 3) if sp in matrix[b] else None)
        table[sp] = row
    import pandas as pd
    df = pd.DataFrame(table, index=bench_list).T
    st.dataframe(df.style.background_gradient(cmap='YlGn', axis=1, vmin=0, vmax=1), use_container_width=True)
    st.caption(T["cross_caption"])

# ==================== 底部 GitHub 链接 ====================
st.divider()
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <a href="https://github.com/Shutong-Hou/TopoLeaf" target="_blank" style="text-decoration: none; color: #2c3e50; font-weight: 500;">
        <img src="https://github.com/fluidicon.png" width="22" style="vertical-align: middle; margin-right: 6px;">
        TopoLeaf on GitHub
    </a>
</div>
""", unsafe_allow_html=True)
