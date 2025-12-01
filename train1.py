
import argparse
import json
import yaml
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------- Safe imports for local package ---------
def _import(name):
    try:
        return __import__(f"src.{name}", fromlist=["*"])
    except Exception:
        return __import__(name, fromlist=["*"])

dataset_mod = _import("dataset")
model_mod   = _import("model")
utils_mod   = _import("utils")
calib_mod   = _import("calibration")
grad_mod    = _import("gradcam")

# build_dataloaders vs build_loaders 양쪽 지원
build_loaders = getattr(dataset_mod, "build_dataloaders", None) or getattr(dataset_mod, "build_loaders")

build_model   = getattr(model_mod, "build_model")
freeze_all    = getattr(model_mod, "freeze_all", lambda m: None)
unfreeze_top  = getattr(model_mod, "unfreeze_top", lambda m, ratio=0.33: None)
unfreeze_all  = getattr(model_mod, "unfreeze_all", lambda m: None)

set_seed   = getattr(utils_mod, "set_seed")
ensure_dir = getattr(utils_mod, "ensure_dir")
macro_f1   = getattr(utils_mod, "macro_f1")
save_json  = getattr(utils_mod, "save_json")

ece_score        = getattr(calib_mod, "ece_score", None)
TemperatureScaler= getattr(calib_mod, "TemperatureScaler", None)

GradCAM             = getattr(grad_mod, "GradCAM", None)
overlay_cam_on_image= getattr(grad_mod, "overlay_cam_on_image", None)
find_target_layer    = getattr(grad_mod, "find_target_layer", None)

# AMP 호환 래퍼
from contextlib import nullcontext
try:
    from torch.amp import autocast as _autocast_new, GradScaler as _GradScalerNew
    def amp_autocast(use_amp: bool):
        return _autocast_new(device_type="cuda", enabled=use_amp, dtype=torch.float16)
    def amp_scaler(use_amp: bool):
        return _GradScalerNew(enabled=use_amp)
except Exception:  # 구버전 호환
    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScalerOld
    def amp_autocast(use_amp: bool):
        return _autocast_old(enabled=use_amp, dtype=torch.float16)
    def amp_scaler(use_amp: bool):
        return _GradScalerOld(enabled=use_amp)

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, non_blocking: bool):
    model.eval()
    ys, ps, logits_all, probs_max = [], [], [], []
    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=non_blocking) if device.type=="cuda" else x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu()
        ys.append(y)
        ps.append(pred)
        logits_all.append(logits.detach().cpu())
        probs = torch.softmax(logits, dim=1)
        probs_max.append(probs.max(1).values.cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    logits = torch.cat(logits_all).cpu()
    confs  = torch.cat(probs_max).numpy()
    return y_true, y_pred, logits, confs

def plot_curves(log, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(log["train_loss"])+1))
    plt.figure(figsize=(9,6))
    ax1 = plt.gca()
    ax1.plot(epochs, log["train_loss"], label="train_loss", color="#d62728")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="#d62728")
    ax1.tick_params(axis='y', labelcolor="#d62728")
    ax2 = ax1.twinx()
    if log["train_acc"]: ax2.plot(epochs, log["train_acc"], label="train_acc", color="#1f77b4")
    if log["val_acc"]:   ax2.plot(epochs, log["val_acc"], label="val_acc", linestyle="--", color="#1f77b4")
    if log["val_f1"]:    ax2.plot(epochs, log["val_f1"], label="val_f1", color="#2ca02c")
    ax2.set_ylabel("acc / f1")
    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l; labels += lab
    plt.legend(lines, labels, loc="best")
    if log.get("ece_pre") and any([v is not None for v in log["ece_pre"]]):
        # 작은 서브플롯로 ECE 표시
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax2, width="35%", height="35%", loc='lower right')
        axins.plot(epochs, [v if v is not None else np.nan for v in log["ece_pre"]], label="ECE(pre)", color="#9467bd")
        axins.plot(epochs, [v if v is not None else np.nan for v in log["ece_post"]], label="ECE(post)", color="#8c564b")
        axins.set_title("ECE", fontsize=9)
        axins.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_metrics_csv(log, out_csv: Path):
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = ["epoch","train_loss","train_acc","val_acc","val_f1","ece_pre","ece_post","T"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(len(log["train_loss"])):
            writer.writerow([
                i+1,
                log["train_loss"][i],
                log["train_acc"][i] if i < len(log["train_acc"]) else "",
                log["val_acc"][i] if i < len(log["val_acc"]) else "",
                log["val_f1"][i] if i < len(log["val_f1"]) else "",
                log["ece_pre"][i] if log.get("ece_pre") else "",
                log["ece_post"][i] if log.get("ece_post") else "",
                log["T"][i] if log.get("T") else "",
            ])

def maybe_gradcam_snapshots(model, device, val_loader, classes: List[str], out_dir: Path,
                            target_layer: Optional[str], gradcam_pp: bool,
                            limit_scan: int, mis_n: int, low_n: int, ok_n: int, tau: float):
    if GradCAM is None:
        print("[gradcam] GradCAM utility not found. Skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    non_blocking = (device.type=="cuda")
    # 수집
    picked = []
    cnt = 0
    for x, y in val_loader:
        x = x.to(device, non_blocking=non_blocking) if device.type=="cuda" else x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confs, yhat = probs.max(1)
        for i in range(x.size(0)):
            picked.append((x[i:i+1].cpu(), int(y[i].cpu()), int(yhat[i].cpu()), float(confs[i].cpu())))
            cnt += 1
            if cnt >= limit_scan:
                break
        if cnt >= limit_scan:
            break
    # 분류
    mis = [p for p in picked if p[1]!=p[2]]
    low = [p for p in picked if p[3] < tau]
    ok  = [p for p in picked if p[1]==p[2] and p[3] >= tau]
    sel = mis[:mis_n] + low[:low_n] + ok[:ok_n]
    if len(sel)==0 and len(picked)>0:
        sel = picked[:max(1, mis_n+low_n+ok_n)]
    # 타깃 레이어
    tlayer = target_layer or find_target_layer(model)
    cam = GradCAM(model, target_layer=tlayer, use_cuda=(device.type=="cuda"), gradcam_pp=gradcam_pp)
    # 저장
    from PIL import Image
    import torchvision.transforms as tvt
    to_pil = tvt.ToPILImage()
    for idx, (x1, yt, yp, conf) in enumerate(sel):
        x1 = x1.to(device)
        heat, _ = cam(x1, class_idx=yp)
        # 원본 복원
        # assume normalization mean/std
        mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
        x_vis = (x1*std + mean).clamp(0,1).squeeze(0).cpu()
        pil = to_pil(x_vis)
        over = overlay_cam_on_image(heat, pil)
        over.save(out_dir / f"ep_cam_{idx:03d}_pred-{classes[yp]}_true-{classes[yt]}_conf-{conf:.2f}.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--gradcam", action="store_true", help="save Grad-CAM snapshots each epoch (val split)")
    ap.add_argument("--gradcam_pp", action="store_true", help="use Grad-CAM++")
    ap.add_argument("--target_layer", type=str, default=None)
    ap.add_argument("--cam_scan_limit", type=int, default=256)
    ap.add_argument("--cam_mis_per_epoch", type=int, default=4)
    ap.add_argument("--cam_low_per_epoch", type=int, default=4)
    ap.add_argument("--cam_ok_per_epoch", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.55)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    set_seed(cfg.get("seed", 42))
    device = select_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # paths
    runs_dir   = Path(cfg.get("paths",{}).get("runs_dir", "runs"))
    figs_dir   = Path(cfg.get("paths",{}).get("figures_dir", "report/figures"))
    tables_dir = Path(cfg.get("paths",{}).get("tables_dir", "report/tables"))
    ensure_dir(runs_dir.as_posix()); ensure_dir(figs_dir.as_posix()); ensure_dir(tables_dir.as_posix())

    # data
    tr_loader, va_loader, te_loader, classes = build_loaders(
        root=cfg["data"]["root"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        sampler=cfg.get("train",{}).get("sampler","none")
    )

    # model
    model = build_model(num_classes=len(classes)).to(device)

    # loss
    if cfg.get("train",{}).get("use_focal", False):
        gamma = cfg.get("train",{}).get("focal_gamma", 2.0)
        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, reduction="mean"):
                super().__init__(); self.gamma=gamma; self.reduction=reduction; self.ce = nn.CrossEntropyLoss(reduction='none')
            def forward(self, logits, target):
                ce = self.ce(logits, target)
                pt = torch.softmax(logits, dim=1).gather(1, target.view(-1,1)).squeeze(1).clamp(min=1e-6)
                loss = (1-pt)**self.gamma * ce
                return loss.mean() if self.reduction=="mean" else loss.sum()
        criterion = FocalLoss(gamma=gamma)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("train",{}).get("label_smoothing", 0.1))

    # optim/sched
    base_lr = cfg.get("train",{}).get("lr", 3e-4)
    wd      = cfg.get("train",{}).get("weight_decay", 1e-4)
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, weight_decay=wd)

    ep_head    = cfg.get("train",{}).get("epochs_head", 2)
    ep_partial = cfg.get("train",{}).get("epochs_partial", 2)
    ep_full    = cfg.get("train",{}).get("epochs_full", 4)
    warmup_ep  = cfg.get("train",{}).get("warmup_epochs", 1)
    total_ep   = ep_head + ep_partial + ep_full
    warm = LinearLR(opt, start_factor=0.1, total_iters=max(1,warmup_ep))
    cos  = CosineAnnealingLR(opt, T_max=max(1,total_ep-warmup_ep))
    sched = SequentialLR(opt, [warm, cos], milestones=[warmup_ep])

    use_amp = (device.type=="cuda")
    scaler = amp_scaler(use_amp)

    # logs
    best_f1 = -1.0
    best_ckpt = runs_dir / cfg.get("paths",{}).get("ckpt_name","best.ckpt")
    log = {"train_loss":[], "train_acc":[], "val_acc":[], "val_f1":[], "ece_pre":[], "ece_post":[], "T":[]}

    def fit_epochs(n_epoch: int):
        nonlocal best_f1
        for _ in range(n_epoch):
            # train
            model.train()
            total_loss=0.0; total_n=0; correct=0
            for x, y in tqdm(tr_loader, leave=False):
                x = x.to(device, non_blocking=(device.type=="cuda")) if device.type=="cuda" else x.to(device)
                y = y.to(device, non_blocking=(device.type=="cuda")) if device.type=="cuda" else y.to(device)
                opt.zero_grad(set_to_none=True)
                with (amp_autocast(use_amp) if use_amp else nullcontext()):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                total_loss += loss.item()*x.size(0)
                total_n += x.size(0)
                correct += (logits.argmax(1)==y).sum().item()
            tr_loss = total_loss/max(1,total_n); tr_acc = correct/max(1,total_n)

            # val
            y_true, y_pred, logits_val, confs = evaluate(model, va_loader, device, non_blocking=(device.type=="cuda"))
            val_acc = float((y_true==y_pred).mean())
            val_f1  = float(macro_f1(y_true, y_pred))

            T = 1.0; epre=None; epost=None
            if ece_score is not None and TemperatureScaler is not None:
                labels_t = torch.tensor(y_true, dtype=torch.long)
                epre = float(ece_score(logits_val, labels_t, n_bins=15))
                try:
                    temp_scaler = TemperatureScaler()
                    temp_scaler.fit(logits_val, labels_t)
                    T = float(torch.exp(temp_scaler.log_t).item())
                    epost = float(ece_score(logits_val/T, labels_t, n_bins=15))
                except Exception as e:
                    T = 1.0; epost = epre
                    print(f"[temperature] fit failed → T=1.0 ({type(e).__name__}: {e})")

            print(f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  val_acc={val_acc:.4f} val_f1={val_f1:.4f}  "
                  f"ECE(pre)={(epre if epre is not None else float('nan')):.4f}  ECE(post)={(epost if epost is not None else float('nan')):.4f}  T={T:.3f}")

            log["train_loss"].append(tr_loss); log["train_acc"].append(tr_acc)
            log["val_acc"].append(val_acc); log["val_f1"].append(val_f1)
            log["ece_pre"].append(epre); log["ece_post"].append(epost); log["T"].append(T)

            sched.step()

            # best ckpt
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({"model": model.state_dict(), "classes": classes, "temperature": T}, best_ckpt)

            # per-epoch Grad-CAM snapshots
            if args.gradcam:
                out_dir = Path("report/figures/cam_train")/f"epoch_{len(log['train_loss']):03d}"
                maybe_gradcam_snapshots(model, device, va_loader, classes, out_dir,
                                        args.target_layer, args.gradcam_pp,
                                        args.cam_scan_limit, args.cam_mis_per_epoch, args.cam_low_per_epoch, args.cam_ok_per_epoch, args.tau)

            # curves/metrics save
            plot_curves(log, Path("report/figures/train_curves.png"))
            save_metrics_csv(log, Path("report/tables/train_metrics.csv"))
            save_json({"best_macro_f1": float(best_f1), "ckpt": str(best_ckpt)}, runs_dir/"train_summary.json")

    # staged fine-tuning
    freeze_all(model); 
    for p in getattr(model, "classifier", getattr(model, "fc", nn.Linear(1,1))).parameters(): p.requires_grad=True
    fit_epochs(ep_head)
    unfreeze_top(model, ratio=0.33); fit_epochs(ep_partial)
    unfreeze_all(model); fit_epochs(ep_full)

    print(f"[train] best Macro F1: {best_f1:.4f} saved: {best_ckpt}")

if __name__=="__main__":
    main()
