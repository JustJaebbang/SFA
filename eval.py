import argparse, yaml, torch, json
from pathlib import Path
from tqdm import tqdm
from .dataset import build_dataloaders
from .model import build_model
from .utils import macro_f1, plot_confusion

@torch.no_grad()
def infer_all(model, loader, device):
    model.eval(); ys, ps = [], []
    for x,y in tqdm(loader, leave=False):
        x = x.to(device); pred = model(x).argmax(1).cpu()
        ys.append(y); ps.append(pred)
    import numpy as np
    return np.concatenate(ys), np.concatenate(ps)

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

    y_true, y_pred = infer_all(model, te, device)
    acc = (y_true==y_pred).mean(); f1 = macro_f1(y_true, y_pred)
    print(f"Test Acc={acc:.4f} MacroF1={f1:.4f}")

    out_dir = Path("report"); (out_dir/"figures").mkdir(parents=True, exist_ok=True); (out_dir/"tables").mkdir(exist_ok=True)
    plot_confusion(y_true, y_pred, classes, out_dir/"figures/confusion_matrix.png")
    json.dump({"acc":float(acc), "macro_f1":float(f1)}, open(out_dir/"tables/test_report.json","w"), indent=2)
