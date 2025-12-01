
import argparse, yaml, torch, json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

from .dataset import build_dataloaders
from .model import build_model
from .utils import macro_f1, plot_confusion
import torch.nn.functional as F
from .calibration import ece_score
from .transforms import build_transforms
from .gradcam import GradCAM, find_target_layer, overlay_cam_on_image

@torch.no_grad()
def infer_all(model, loader, device):
    model.eval(); ys, logits_all = [], []
    for x,y in tqdm(loader, leave=False):
        x = x.to(device); logits = model(x)
        ys.append(y); logits_all.append(logits.cpu())
    return np.concatenate(ys), torch.cat(logits_all)

def generate_cam_for_indices(model, device, dataset, classes, indices, out_dir: Path,
                             img_size: int, target_layer: str = None, gradcam_pp: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build a single GradCAM instance (target layer may be auto-resolved once)
    if target_layer is None:
        target_layer = find_target_layer(model)
    cam_engine = GradCAM(model, target_layer=target_layer, gradcam_pp=gradcam_pp)
    saved = []
    for i in indices:
        img_path, true_idx = dataset.samples[i]
        pil = Image.open(img_path).convert("RGB")
        x = build_transforms(img_size, False)(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred_idx = int(logits.argmax(1).item())
            conf = float(F.softmax(logits, dim=1)[0, pred_idx].item())
        cam_map, _ = cam_engine(x, class_idx=pred_idx)
        overlay = overlay_cam_on_image(cam_map, pil, alpha=0.4, cmap="jet")
        pred_name, true_name = classes[pred_idx], classes[true_idx]
        fname = out_dir / f"{Path(img_path).stem}_pred-{pred_name}_true-{true_name}_conf-{conf:.2f}.png"
        overlay.save(fname)
        saved.append(str(fname))
    cam_engine.close()
    return saved

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", required=True)
    # XAI options
    ap.add_argument("--gradcam", action="store_true", help="Save Grad-CAM overlays for selected test samples")
    ap.add_argument("--gradcam_pp", action="store_true")
    ap.add_argument("--target_layer", type=str, default=None)
    ap.add_argument("--cam_out_dir", type=str, default="report/figures/cam_eval")
    ap.add_argument("--cam_samples", type=int, default=12, help="Max number of CAM images to save")
    ap.add_argument("--tau", type=float, default=0.55, help="Low-confidence threshold")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,"r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, te, classes = build_dataloaders(cfg["data"]["root"], cfg["data"]["img_size"],
                                          cfg["data"]["batch_size"], cfg["data"]["num_workers"], sampler="none")
    sd = torch.load(args.ckpt, map_location="cpu")
    T  = float(sd.get("temperature", 1.0))

    model = build_model(len(classes)); model.load_state_dict(sd["model"]); model.to(device).eval()

    # Evaluate
    y_true, logits = infer_all(model, te, device)
    probs = F.softmax(logits / T, dim=1)
    conf, y_pred = probs.max(1)
    y_true_np, y_pred_np = y_true, y_pred.numpy()
    acc = (y_true_np==y_pred_np).mean()
    f1 = macro_f1(y_true_np, y_pred_np)
    labels_t = torch.tensor(y_true_np, dtype=torch.long)

    ece_pre = ece_score(logits, labels_t, n_bins=15)
    ece_post = ece_score(logits / T, labels_t, n_bins=15)

    print(f"Test Acc={acc:.4f} MacroF1={f1:.4f} ECE(pre)={ece_pre:.4f} ECE(post)={ece_post:.4f}  T={T:.3f}")

    out_dir = Path("report")
    (out_dir/"figures").mkdir(parents=True, exist_ok=True)
    (out_dir/"tables").mkdir(exist_ok=True)
    # Confusion matrix
    from .utils import plot_confusion
    plot_confusion(y_true_np, y_pred_np, classes, out_dir/"figures/confusion_matrix.png")
    # JSON
    json.dump({"acc":float(acc), "macro_f1":float(f1), "ece_pre":float(ece_pre),
               "ece_post":float(ece_post), "temperature":float(T)},
               open(out_dir/"tables/test_report.json","w"), indent=2)

    # XAI: save CAMs for misclassified and low-confidence samples
    cam_paths = []
    if args.gradcam:
        mis_idx = np.where(y_true_np != y_pred_np)[0].tolist()
        lowc_idx = np.where(conf.numpy() < args.tau)[0].tolist()
        # combine and keep order with uniqueness
        selected = []
        for idx in mis_idx + lowc_idx:
            if idx not in selected:
                selected.append(idx)
        selected = selected[:args.cam_samples]
        print(f"[Grad-CAM] selected {len(selected)} samples (mis={len(mis_idx)}, low_conf={len(lowc_idx)})")
        cam_dir = Path(args.cam_out_dir)
        cam_paths = generate_cam_for_indices(model, device, te.dataset, classes, selected, cam_dir,
                                             cfg['data']['img_size'], target_layer=args.target_layer,
                                             gradcam_pp=args.gradcam_pp)
        # Write list for convenience
        json.dump({"cam_paths": cam_paths}, open(out_dir/"tables/cam_list.json","w"), indent=2, ensure_ascii=False)
        print(f"[Grad-CAM] saved {len(cam_paths)} images under {cam_dir}")

