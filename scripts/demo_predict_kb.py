# scripts/demo_predict_kb.py
import sys, yaml, torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

# 프로젝트 모듈
from src.model import build_model
from src.transforms import build_transforms
from src.kb import load_nutrition, get_nutrition_for_label, scale_nutrition

def main():
    # 경로 설정(필요시 수정)
    cfg_path = "configs/config.yaml"
    ckpt_path = "runs/best.ckpt"
    nutrition_csv = "data/nutrition.csv"
    aliases_csv = "data/aliases.csv"
    image_path = sys.argv[1] if len(sys.argv) > 1 else "path/to/test.jpg"
    grams = 200  # 200g 기준으로 환산

    # 설정/모델/전처리
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=cfg["model"]["num_classes"])
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd["model"])
    classes = sd.get("classes", [str(i) for i in range(cfg["model"]["num_classes"])])
    temperature = float(sd.get("temperature", 1.0))
    model = model.to(device).eval()

    tfm = build_transforms(cfg["data"]["img_size"], is_train=False)

    # 이미지 → 예측
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x) / temperature
        probs = F.softmax(logits, dim=1)[0]
        conf, idx = probs.max(dim=0)
    pred_label = classes[int(idx)]
    print(f"[Predict] {pred_label} (p={float(conf):.3f})")

    # 영양 DB 로드
    nutrition_map, aliases_idx = load_nutrition(nutrition_csv, aliases_csv=aliases_csv, include_display_keys=True)

    # 라벨→영양(100g) 매칭
    per100 = get_nutrition_for_label(pred_label, nutrition_map, aliases_idx)
    if per100 is None:
        print("[KB] 매칭 실패: aliases.csv/정규화 규칙을 보강하세요.")
        return

    # g(그램) 변환
    per_grams = scale_nutrition(per100, grams)
    print(f"[KB] 100g 기준: {per100}")
    print(f"[KB] {grams} g 기준: {per_grams}")

if __name__ == "__main__":
    main()
