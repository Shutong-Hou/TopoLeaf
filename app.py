import streamlit as st
import torch
import numpy as np
import cv2
import os
import json
import time
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = "results"
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 真实实验统计值（来自 Apple 基准，100张健康图像，precompute_final.py 输出）
MEAN_LOC = 0.0944
STD_LOC = 0.0298
HEALTHY_MAX_SCORE = 1.77

st.markdown("""
<style>
    .stDataFrame td { color: #1a1a1a !important; font-weight: 500 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .loading-placeholder {
        display: flex; align-items: center; justify-content: center;
        height: 300px; background: #f8f9fa; border-radius: 12px;
        border: 2px dashed #dee2e6; color: #868e96; font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

TEXTS = {
    "zh": {
        "title": "TopoLeaf：零样本植物病害检测（云端版）",
        "subtitle": "基于 DINOv2 的几何异常评分 · 上传叶片即检测",
        "note": "云端轻量版：仅 Apple 基准 + 几何评分 + 热力图。完整功能请本地运行 app_local.py。",
        "upload_single": "选择一张叶片图片",
        "score_geo": "几何异常评分",
        "unit_geo": "σ",
        "heatmap": "异常热力图",
        "loading": "正在分析叶片图像，请稍候...",
        "placeholder": "📤 请上传一张叶片图像以开始检测",
        "cross_title": "跨物种零样本异常检测性能 (AUROC)",
        "cross_caption": "几何 AUROC，100 张健康基准。完整实验见 GitHub。",
        "language": "语言",
        "threshold_label": "健康基准最大评分",
        "repo_link": "📂 完整功能与代码见 GitHub",
    },
    "en": {
        "title": "TopoLeaf: Zero-Shot Plant Disease Detection (Cloud)",
        "subtitle": "DINOv2-based Geometric Anomaly Scoring",
        "note": "Cloud version: Apple benchmark + geometric score + heatmap only. Run app_local.py for full features.",
        "upload_single": "Choose a leaf image",
        "score_geo": "Geometric Anomaly Score",
        "unit_geo": "σ",
        "heatmap": "Anomaly Heatmap",
        "loading": "Analyzing leaf image...",
        "placeholder": "📤 Please upload a leaf image to start",
        "cross_title": "Cross-Species AUROC",
        "cross_caption": "Geometric AUROC, 100 healthy benchmark. See GitHub for full experiments.",
        "language": "Language",
        "threshold_label": "Max Healthy Score",
        "repo_link": "📂 Full code & features on GitHub",
    }
}

@st.cache_resource
def load_dino():
    local_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "facebookresearch_dinov2_main")
    if os.path.exists(local_dir):
        model = torch.hub.load(local_dir, 'dinov2_vits14', source='local', pretrained=True).to(DEVICE)
    else:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True).to(DEVICE)
    model.eval()
    return model

def extract_features(image, model):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255.0
    tensor = (tensor - torch.tensor([0.485,0.456,0.406]).view(3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = model.forward_features(tensor)['x_norm_patchtokens']
    return feats.squeeze(0).cpu().numpy()

def geometric_score(feats):
    fsel = feats[:, :192]
    centroid = fsel.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(fsel - centroid, axis=1)
    dev = (dists - MEAN_LOC) / STD_LOC
    return np.percentile(dev, 90), dev

def heatmap(feats, orig_img):
    fsel = feats[:, :192]
    centroid = fsel.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(fsel - centroid, axis=1)
    dev = (dists - MEAN_LOC) / STD_LOC
    hmap = dev.reshape(int(np.sqrt(len(dev))), -1)
    hmap = np.clip(hmap, 0, None)
    hmap = cv2.resize(hmap, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    hmap = cv2.GaussianBlur(hmap, (21,21), 0)
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    colored = cv2.applyColorMap((hmap*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_img, 0.4, colored, 0.6, 0)
    return overlay

def load_matrix():
    matrix = {}
    for bench in ["Apple", "Tomato", "Grape", "Strawberry", "Corn_(maize)"]:
        path = os.path.join(RESULTS_DIR, f"results_{bench}.json")
        if os.path.exists(path):
            with open(path) as f:
                matrix[bench] = json.load(f)
    return matrix

# ==================== UI ====================
st.set_page_config(page_title="TopoLeaf Cloud", layout="wide")
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

col_t, col_l = st.columns([5, 1])
with col_l:
    lang = st.radio("", ["中文", "English"], index=0, label_visibility="collapsed")
    st.session_state.lang = "zh" if lang == "中文" else "en"
T = TEXTS[st.session_state.lang]

st.title(T["title"])
st.caption(T["subtitle"])
st.sidebar.caption(T["note"])
st.sidebar.markdown(f"[{T['repo_link']}](https://github.com/Shutong-Hou/TopoLeaf)")

model = load_dino()

file = st.file_uploader(T["upload_single"], type=["jpg", "jpeg", "png"])

if not file:
    col_empty, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        st.markdown(f'<div class="loading-placeholder">{T["placeholder"]}</div>', unsafe_allow_html=True)
else:
    with st.spinner(T["loading"]):
        image = Image.open(file).convert("RGB")
        feats = extract_features(image, model)
        geo, _ = geometric_score(feats)
        orig_bgr = cv2.cvtColor(np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))), cv2.COLOR_RGB2BGR)
        over = heatmap(feats, orig_bgr)
        time.sleep(0.3)

    c_img, c_score, c_heat = st.columns([0.25, 0.20, 0.55])
    with c_img:
        st.image(image, use_container_width=True)
    with c_score:
        st.metric(T["score_geo"], f"{geo:.2f} {T.get('unit_geo', 'σ')}")
        st.caption(f"{T['threshold_label']}: {HEALTHY_MAX_SCORE:.2f} σ")
    with c_heat:
        st.image(over, caption=T["heatmap"], use_container_width=True)

st.divider()
st.header(T["cross_title"])
matrix = load_matrix()
if matrix:
    species_order = ["Cherry_(including_sour)", "Corn_(maize)", "Grape", "Peach",
                     "Pepper,_bell", "Potato", "Strawberry", "Tomato"]
    bench_list = ["Apple", "Tomato", "Grape", "Strawberry", "Corn_(maize)"]
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
