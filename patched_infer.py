
import argparse, yaml, torch, json
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from .model import build_model
from .transforms import build_transforms
from .kb import load_aliases, load_nutrition, normalize_label
from .gradcam import GradCAM, find_target_layer, overlay_cam_on_image

@torch.no_grad()
def predict(img_path, model, device, img_size, classes, topk, T=1.0):
    img = Image.open(img_path).convert("RGB")
    x = build_transforms(img_size, False)(img).unsqueeze(0).to(device)
    logits = model(x)
    logits = logits / T
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

def save_cam(img_path: str, model, device, img_size: int, class_idx: int, target_layer: str = None,
             gradcam_pp: bool = False, out_path: Path = None, T: float = 1.0):
    """Generate and save Grad-CAM overlay for a single image and class index."""
    model.eval()
    pil = Image.open(img_path).convert("RGB")
    x = build_transforms(img_size, False)(pil).unsqueeze(0).to(device)
    # Note: temperature scaling does not affect Grad-CAM computation (uses logits gradient).
    # If you insist, you can wrap model to divide logits by T, but it's not standard.
    if target_layer is None:
        target_layer = find_target_layer(model)
    cam = GradCAM(model, target_layer=target_layer, gradcam_pp=gradcam_pp)
    cam_map, _ = cam(x, class_idx=class_idx)
    cam.close()
    overlay = overlay_cam_on_image(cam_map, pil, alpha=0.4, cmap="jet")
    overlay.save(out_path)
    return str(out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--portion", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.55)
    # XAI options
    ap.add_argument("--gradcam", action="store_true", help="Save Grad-CAM overlay for top1")
    ap.add_argument("--gradcam_pp", action="store_true", help="Use Grad-CAM++ weighting")
    ap.add_argument("--target_layer", type=str, default=None, help="Target conv layer name (auto if omitted)")
    ap.add_argument("--cam_out", type=str, default="report/figures/cam_infer.png")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config,"r"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.ckpt, map_location="cpu")
    T  = float(sd.get("temperature", 1.0))
    classes = sd["classes"]; model = build_model(len(classes))
    model.load_state_dict(sd["model"]); model.to(device).eval()

    preds = predict(args.image, model, device, cfg["data"]["img_size"], classes, args.topk, T=T)
    warnings = []
    if preds[0][1] < args.tau: warnings.append("low_confidence")

    # Grad-CAM (top1) if requested
    cam_path = None
    if args.gradcam:
        pred_top1_label = preds[0][0]
        class_idx = classes.index(pred_top1_label)
        out_path = Path(args.cam_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cam_path = save_cam(args.image, model, device, cfg["data"]["img_size"], class_idx,
                            target_layer=args.target_layer, gradcam_pp=args.gradcam_pp,
                            out_path=out_path, T=T)

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
        "calibrated": True,
        "warnings": sorted(set(warnings)),
        "xai": {
            "gradcam": bool(args.gradcam),
            "gradcam_pp": bool(args.gradcam_pp),
            "target_layer": args.target_layer or "auto",
            "overlay_path": cam_path
        } if args.gradcam else None
    }, ensure_ascii=False, indent=2))
