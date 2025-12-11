import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models


# =========================
# 설정
# =========================
# app.py 상단
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_CHECKPOINT_PATH = BASE_DIR / "best_model.pth"
ALIASES_CSV_PATH      = BASE_DIR / "aliases.csv"        # 또는 실제 파일명
NUTRITION_CSV_PATH    = BASE_DIR / "nutrition.csv"      # 또는 실제 파일명

IMAGE_SIZE            = 224


# =========================
# 모델 / 전처리 유틸
# =========================
def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """EfficientNet-B0 분류기 생성 (네가 train.py / infer.py에서 쓰던 것과 동일하게)."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@st.cache_resource
def load_model_and_classes():
    """체크포인트에서 모델과 class_names 로딩 (캐시)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
    class_names = ckpt.get("class_names", None)
    if class_names is None:
        raise ValueError("Checkpoint 에 'class_names' 리스트가 없습니다.")

    num_classes = len(class_names)
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    transform = get_transform(IMAGE_SIZE)
    return transform(pil_image).unsqueeze(0)  # (1, C, H, W)


def predict_food(model, device, input_tensor, class_names):
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, pred_idx = torch.max(probs, dim=0)

    pred_idx = int(pred_idx.item())
    conf = float(conf.item())
    pred_class = class_names[pred_idx]

    # 확률 dict (원하면 테이블로 표시 가능)
    prob_dict = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}
    return pred_class, conf, prob_dict


# =========================
# DB 로딩 & 매핑 유틸
# =========================
@st.cache_data
def load_metadata():
    aliases = pd.read_csv(ALIASES_CSV_PATH)
    nutrition = pd.read_csv(NUTRITION_CSV_PATH)

    # food_id 기준으로 nutrition 인덱스 세팅
    nutrition = nutrition.set_index("food_id")

    return aliases, nutrition


def normalize_label(label: str) -> str:
    """
    모델의 class_name 과 aliases.normalized 를 맞추기 위한 간단한 정규화.
    네 쪽 클래스 네이밍 규칙에 맞게 수정해도 됨.
    """
    label = label.strip().lower()
    label = label.replace(" ", "_")
    return label


def find_food_id_from_label(pred_label: str, aliases_df: pd.DataFrame) -> str | None:
    """
    모델 예측 label(예: 'bibimbap')을 aliases 테이블에서 food_id 로 매핑.
    우선 normalized 컬럼, 안 되면 alias 컬럼에서 찾음.
    """
    norm = normalize_label(pred_label)

    # 1) normalized 로 우선 매칭
    row = aliases_df[aliases_df["normalized"] == norm]
    if len(row) == 0:
        # 2) alias 로 fallback
        row = aliases_df[aliases_df["alias"].str.lower() == pred_label.lower()]

    if len(row) == 0:
        return None

    return row.iloc[0]["food_id"]


def scale_nutrition(nutri_row: pd.Series, portion_g: float) -> pd.Series:
    """
    nutrition.csv 는 per_100g 기준 (serving_g = 100).
    사용자가 입력한 portion_g 에 맞게 선형 스케일링.
    """
    base_serving_g = nutri_row["serving_g"]  # 보통 100
    factor = portion_g / base_serving_g

    cols_to_scale = ["energy_kcal", "carb_g", "protein_g", "fat_g", "sodium_mg"]
    scaled = nutri_row.copy()
    for c in cols_to_scale:
        scaled[c] = nutri_row[c] * factor

    scaled["portion_g"] = portion_g
    return scaled


# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="Food Nutrition Estimator", page_icon="🍱", layout="centered")

    st.title("🍱 음식 이미지 기반 영양성분 추정 데모")
    st.markdown(
        """
        1. 음식 사진을 업로드하면 모델이 **어떤 음식인지 분류**합니다.  
        2. 아래 **분량 슬라이더**로 예상 섭취량(g)을 조절하면,  
           준비된 nutrition DB를 이용해 **영양성분 값을 선형 스케일링**해서 보여줍니다.
        """
    )

    # ---- 메타데이터 / 모델 로딩 ----
    aliases_df, nutrition_df = load_metadata()
    try:
        model, class_names, device = load_model_and_classes()
    except Exception as e:
        st.error(f"모델 로딩 중 오류가 발생했습니다: {e}")
        return

    # ---- 이미지 업로드 ----
    uploaded = st.file_uploader(
        "음식 사진을 업로드하세요 (jpg, png)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.info("왼쪽 상단에 이미지를 업로드하면 결과가 여기 표시됩니다.")
        return

    # PIL 이미지로 열기
    try:
        pil_image = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"이미지를 여는 중 오류가 발생했습니다: {e}")
        return

    st.image(pil_image, caption="업로드된 이미지", use_column_width=True)

    # ---- 모델 추론 ----
    with st.spinner("이미지 분류 중..."):
        input_tensor = preprocess_image(pil_image)
        pred_label, conf, prob_dict = predict_food(model, device, input_tensor, class_names)

    st.subheader("1️⃣ 분류 결과")
    st.write(f"**예측 음식:** `{pred_label}`  (신뢰도: {conf*100:.1f}%)")

    # ---- food_id 매핑 ----
    food_id = find_food_id_from_label(pred_label, aliases_df)
    if food_id is None or food_id not in nutrition_df.index:
        st.warning("예측된 음식이 nutrition DB에서 매칭되지 않았습니다. (aliases.csv / nutrition.csv 매핑 확인 필요)")
        return

    nutri_row = nutrition_df.loc[food_id]

    # ---- 사용자가 분량 슬라이더로 조절 ----
    st.subheader("2️⃣ 섭취량 설정")

    default_g = float(nutri_row["serving_g"])  # 보통 100g 기준
    min_g = 50
    max_g = 1000

    portion_g = st.slider(
        "예상 섭취량 (g)",
        min_value=min_g,
        max_value=max_g,
        value=int(default_g),
        step=10,
    )

    scaled = scale_nutrition(nutri_row, portion_g)

    # ---- 결과 표시 ----
    st.subheader("3️⃣ 추정 영양성분")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**기준 정보**  \n(food_id: `{food_id}`)")
        st.write(f"- 기준 serving: {nutri_row['serving_name']} ({nutri_row['serving_g']} g)")
        st.write(f"- 데이터 출처: {nutri_row['source']} / 업데이트: {nutri_row['updated']}")

    with col2:
        st.markdown("**사용자 설정 섭취량**")
        st.write(f"- 섭취량: **{scaled['portion_g']:.0f} g**")

    # 영양 성분 테이블
    result_df = pd.DataFrame(
        {
            "영양성분": ["에너지 (kcal)", "탄수화물 (g)", "단백질 (g)", "지방 (g)", "나트륨 (mg)"],
            "값": [
                scaled["energy_kcal"],
                scaled["carb_g"],
                scaled["protein_g"],
                scaled["fat_g"],
                scaled["sodium_mg"],
            ],
        }
    )

    st.table(result_df.style.format({"값": "{:.2f}"}))

    # (선택) 확률 상위 k개도 보고 싶다면:
    with st.expander("🔍 상위 예측 클래스 / 확률 보기"):
        topk = 5
        items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
        prob_df = pd.DataFrame(items, columns=["class_name", "probability"])
        st.table(prob_df.style.format({"probability": "{:.3f}"}))


if __name__ == "__main__":
    main()
