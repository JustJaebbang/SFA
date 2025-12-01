import argparse, yaml, torch, json
from pathlib import Path
from tqdm import tqdm
from .dataset import build_dataloaders
from .model import build_model
from .utils import macro_f1, plot_confusion

import torch.nn.functional as F

def ece_score(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE).
    logits: [N, C] (on CPU ok)
    labels: [N]    (int64)
    """
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        acc = pred.eq(labels).float()
        bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
        ece = torch.zeros(1, device=logits.device)
        for i in range(n_bins):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            if m.any():
                ece += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
        return float(ece.item())

@torch.no_grad()
def infer_all(model, loader, device):
    model.eval(); ys, ps, logits_all = [], [], []
    for x,y in tqdm(loader, leave=False):
        x = x.to(device); logits = model(x)
        pred = logits.argmax(1).cpu()
        ys.append(y); ps.append(pred); logits_all.append(logits.cpu())
    import numpy as np, torch
    return np.concatenate(ys), np.concatenate(ps), torch.cat(logits_all)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config,"r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, te, classes = build_dataloaders(cfg["data"]["root"], cfg["data"]["img_size"],
                                          cfg["data"]["batch_size"], cfg["data"]["num_workers"], sampler="none")
    sd = torch.load(args.ckpt, map_location="cpu")
    model = build_model(len(classes)); model.load_state_dict(sd["model"]); model.to(device)

    y_true, y_pred, logits = infer_all(model, te, device)
    acc = (y_true==y_pred).mean(); f1 = macro_f1(y_true, y_pred)
    labels_t = torch.tensor(y_true)
    ece = ece_score(logits, labels_t, n_bins=15)
    print(f"Test Acc={acc:.4f} MacroF1={f1:.4f} ECE(pre)={ece:.4f}")

    out_dir = Path("report"); (out_dir/"figures").mkdir(parents=True, exist_ok=True); (out_dir/"tables").mkdir(exist_ok=True)
    plot_confusion(y_true, y_pred, classes, out_dir/"figures/confusion_matrix.png")
    json.dump({"acc":float(acc), "macro_f1":float(f1), "ece_pre":float(ece)}, open(out_dir/"tables/test_report.json","w"), indent=2)
