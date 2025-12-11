# app.py
import io
import json
import time
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

# === 당신 프로젝트 모듈에 맞게 경로/함수명을 확인하세요 ===
from .model import build_model                 # (필수) 모델 빌더
from .transforms import build_transforms       # (필수) 전처리 빌더
from .kb import load_nutrition, normalize_label  # (필수) 영양 DB 로더/라벨 정규화
# 선택: Grad-CAM
try:
    from src.gradcam import GradCAM, find_target_layer, overlay_cam_on_image
    HAS_CAM = True
except Exception:
    HAS_CAM = False

st.set_page_config(page_title="Food AI + Nutrition", layout="wide")

# ============== 사이드바: 설정 ==============
with st.sidebar:
    st.header("Settings")
    cfg_path = st.text_input("Config path", "configs/config.yaml")
    ckpt_path = st.text_input("Checkpoint", "runs/best.ckpt")
    topk = st.number_input("Top‑k", min_value=1, max_value=10, value=3, step=1)
    use_cuda = st.toggle("Use CUDA (if available)", value=torch.cuda.is_available())
    show_cam = st.toggle("Show Grad‑CAM", value=False and HAS_CAM)
    target_layer = st.text_input("Target layer (optional, e.g., layer4)", value="")
    st.caption("Grad‑CAM은 백본에 따라 target layer 지정이 필요할 수 있습니다(ResNet: layer4).")

# ============== 캐시된 로더들 ==============
@st.cache_resource(show_spinner=True)
def load_cfg_model(cfg_path: str, ckpt_path: str, use_cuda: bool):
    # 1) 설정 로드
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2) 디바이스
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # 3) 모델 생성/체크포인트 로드
    model = build_model(num_classes=cfg["model"]["num_classes"])
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd["model"])
    classes = sd.get("classes", [str(i) for i in range(cfg["model"]["num_classes"])])
    temperature = float(sd.get("temperature", 1.0))

    model = model.to(device)
    model.eval()

    # 4) 전처리 빌드
    img_size = int(cfg["data"]["img_size"])
    tfm_eval = build_transforms(img_size, is_train=False)

    return cfg, model, device, classes, temperature, tfm_eval

@st.cache_data(show_spinner=False)
def load_kb():
    # 당신의 영양 DB를 불러오는 함수 사용
    # 예: dict[label] = {"kcal": 250, "protein": 12.3, ...} (100g 기준)
    return load_nutrition()

# ============== 추론 함수 ==============
def predict_image(img_pil: Image.Image, model, device, tfm_eval, classes, temperature: float, topk: int = 3):
    x = tfm_eval(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)                      # [1,C]
        logits = logits / temperature          # 보정
        probs = F.softmax(logits, dim=1)[0]    # [C]
        vals, idxs = torch.topk(probs, k=topk)
    topk_list = [(classes[i], float(vals[j])) for j, i in enumerate(idxs.cpu().tolist())]
    pred_idx = int(idxs[0].item())
    pred_label = classes[pred_idx]
    return pred_label, float(vals[0]), topk_list, x  # x는 CAM용 반환

# ============== Grad‑CAM (선택) ==============
def compute_cam_overlay(x_tensor, img_pil, model, device, target_layer: str = ""):
    if not HAS_CAM:
        return None
    # 대상 레이어 결정
    try:
        if target_layer.strip():
            tl = model.get_submodule(target_layer.strip())
        else:
            tl = find_target_layer(model)
    except Exception:
        tl = find_target_layer(model)

    with torch.enable_grad():
        cam = GradCAM(model, target_layer=tl, use_cuda=(device.type == "cuda"))
        try:
            logits_local = model(x_tensor)
            pred_idx = int(logits_local.argmax(1).item())
            cam_map, _ = cam(x_tensor, class_idx=pred_idx)
            overlay = overlay_cam_on_image(cam_map, img_pil, alpha=0.4)  # cmap 인자 없는 버전 호환
        finally:
            if hasattr(cam, "close"):
                cam.close()
            elif hasattr(cam, "remove_hooks"):
                cam.remove_hooks()
    return overlay

# ============== 메인 UI ==============
st.title("Food Image → Prediction → Nutrition (per serving)")
cfg, model, device, classes, temperature, tfm_eval = load_cfg_model(cfg_path, ckpt_path, use_cuda)
kb = load_kb()

colL, colR = st.columns([1, 1])
with colL:
    uploaded = st.file_uploader("이미지 업로드(JPG/PNG)", type=["jpg", "jpeg", "png"])
    grams = st.slider("섭취량(gram)", min_value=10, max_value=1000, value=200, step=10)
    st.caption("영양 DB는 100g 기준입니다. 슬라이더 값에 따라 선형 변환합니다.")

with colR:
    st.markdown("#### Config")
    st.json({
        "img_size": cfg["data"]["img_size"],
        "num_classes": cfg["model"]["num_classes"],
        "temperature": temperature,
        "device": str(device),
        "classes_example": classes[:5]
    })

# ============== 처리 ==============
if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(img, caption="입력 이미지", use_container_width=True)

    with st.spinner("추론 중..."):
        pred_label, pred_conf, topk_list, x_tensor = predict_image(img, model, device, tfm_eval, classes, temperature, topk)
        st.success(f"예측: {pred_label}  (p={pred_conf:.3f})")
        st.write("Top‑k:", [{"label": l, "p": round(p, 4)} for l, p in topk_list])

        # 식품명 정규화 후 영양 DB 매핑
        label_norm = normalize_label(pred_label)
        nut = kb.get(label_norm, None)

        if nut is None:
            st.warning(f"영양 DB에서 '{label_norm}'를 찾지 못했습니다. label 매핑(aliases)을 점검하세요.")
        else:
            # 100g 기준 → grams g로 선형 변환
            scale = grams / 100.0
            scaled = {k: round(v * scale, 3) for k, v in nut.items()}

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 100g 기준 영양성분")
                st.json(nut)
            with c2:
                st.markdown(f"#### {grams} g 기준 영양성분")
                st.json(scaled)

        # 선택: Grad‑CAM
        if show_cam and HAS_CAM:
            with st.spinner("Grad‑CAM 생성 중..."):
                overlay = compute_cam_overlay(x_tensor, img, model, device, target_layer)
                if overlay is not None:
                    st.image(overlay, caption="Grad‑CAM", use_container_width=True)
                else:
                    st.info("Grad‑CAM 생성 실패 또는 비활성화.")
        elif show_cam and not HAS_CAM:
            st.info("src/gradcam.py를 찾지 못했습니다. CAM 비활성화.")

else:
    st.info("좌측에서 이미지를 업로드하세요.")
