import argparse, yaml, torch, json
from PIL import Image
import torch.nn.functional as F
from .model import build_model
from .transforms import build_transforms
from .kb import load_aliases, load_nutrition, normalize_label
from pathlib import Path
from .gradcam import GradCAM, find_target_layer, overlay_cam_on_image


@torch.no_grad()
def predict(img_path, model, device, img_size, classes, topk):
    img = Image.open(img_path).convert("RGB")
    x = build_transforms(img_size, False)(img).unsqueeze(0).to(device)
    logits = model(x)
    logits = logits/T
    probs = F.softmax(logits, dim=1)[0]
    vals, idx = probs.topk(topk)
    return [(classes[i], float(vals[j])) for j, i in enumerate(idx.cpu().tolist())]

def scale_nutrition(row, portion):
    def r(v, p=1): return None if v is None else round(float(v)*portion, p)
    return {
        "basis": row.get("basis","per_100g"),
        "serving_name": row.get("serving_name","100 g"),
        "serving_g": r(row.get("serving_g",100), 1),
        "kcal": r(row.get("energy_kcal",0), 0),
        "carb_g": r(row.get("carb_g",0), 1),
        "protein_g": r(row.get("protein_g",0), 1),
        "fat_g": r(row.get("fat_g",0), 1),
        "sodium_mg": r(row.get("sodium_mg",0), 0),
        "source": row.get("source",""),
        "updated": row.get("updated",""),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--portion", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--gradcam", action="store_true")
    ap.add_argument("--gradcam_pp", action="store_true")
    ap.add_argument("--target_layer", type=str, default=None)
    ap.add_argument("--cam_out", type=str, default="report/figures/cam_infer.png")

    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config,"r"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.ckpt, map_location="cpu")
    T  = float(sd.get("temperature", 1.0))
    classes = sd["classes"]; model = build_model(len(classes))
    model.load_state_dict(sd["model"]); model.to(device).eval()

    preds = predict(args.image, model, device, cfg["data"]["img_size"], classes, args.topk)
        # Grad-CAM overlay (optional)
    if args.gradcam:
        # 모델이 실제 올라가 있는 디바이스 확인(안전)
        model_device = next(model.parameters()).device

        # 입력 준비(전처리 동일)
        img = Image.open(args.image).convert("RGB")
        x_cam = build_transforms(cfg["data"]["img_size"], False)(img).unsqueeze(0).to(model_device)

        # Grad-CAM은 grad가 필요 → enable_grad 컨텍스트에서 실행
        with torch.enable_grad():
            # 대상 레이어: 지정이 있으면 경로로 해석, 없으면 자동 탐색
            tgt = model.get_submodule(args.target_layer) if args.target_layer else find_target_layer(model)
            cam_engine = GradCAM(model, target_layer=tgt,
                                 use_cuda=(model_device.type == "cuda"),
                                 gradcam_pp=args.gradcam_pp)
            try:
                # 클래스 선택: Top-1
                logits_local = model(x_cam)
                pred_idx = int(logits_local.argmax(1).item())

                # CAM 생성
                cam_map, _ = cam_engine(x_cam, class_idx=pred_idx)

                # 오버레이 저장
                out_path = Path(args.cam_out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                overlay = overlay_cam_on_image(cam_map, img, alpha=0.4)  # cmap 인자가 없다면 넘기지 마세요
                overlay.save(out_path)
            finally:
                # Grad-CAM 훅 정리(구현 차이 고려)
                if hasattr(cam_engine, "close"):
                    cam_engine.close()
                elif hasattr(cam_engine, "remove_hooks"):
                    cam_engine.remove_hooks()

    warnings = []
    if preds[0][1] < args.tau: warnings.append("low_confidence")

    aliases = load_aliases(cfg["paths"]["aliases_csv"])
    nutrition = load_nutrition(cfg["paths"]["nutrition_csv"])

    mapped = []
    for label, prob in preds:
        norm = normalize_label(label)
        if aliases and norm in aliases:
            food_id, name_ko = aliases[norm]
        else:
            food_id, name_ko = f"food_{norm}", label
            if aliases: warnings.append("label_not_found_in_aliases")
        mapped.append({"food_id": food_id, "name_ko": name_ko, "prob": prob})

    nut = None
    if nutrition is not None and mapped[0]["food_id"] in nutrition.index:
        row = nutrition.loc[mapped[0]["food_id"]].to_dict()
        nut = scale_nutrition(row, args.portion)
    else:
        warnings.append("label_not_found_in_kb")

    print(json.dumps({
        "labels": mapped,
        "top1": mapped[0],
        "nutrition": nut,
        "portion_factor": args.portion,
        "calibrated": False,
        "warnings": sorted(set(warnings))
    }, ensure_ascii=False, indent=2))
