# src/train.py (patched with AMP compatibility + Temperature scaling + ECE pre/post)
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

# Project modules (adjust if names differ in your repo)
from .dataset import build_dataloaders as _build_dataloaders  # or build_loaders
from .model import build_model, freeze_all, unfreeze_top, unfreeze_all
from .losses import FocalLoss
from .utils import set_seed, ensure_dir, macro_f1, save_json

# Optional: ECE / TemperatureScaler. If missing, code still runs with _HAS_ECE=False
try:
    from .calibration import ece_score, TemperatureScaler
    _HAS_ECE = True
except Exception:
    _HAS_ECE = False

# AMP compatibility wrapper (supports both torch.amp and torch.cuda.amp across versions)
from contextlib import nullcontext
try:
    from torch.amp import autocast as _autocast_new, GradScaler as _GradScalerNew
    def amp_autocast(use_amp: bool):
        return _autocast_new(device_type="cuda", enabled=use_amp, dtype=torch.float16)
    def amp_scaler(use_amp: bool):
        # some older minor versions may not accept device_type kw; enabled is enough
        return _GradScalerNew(enabled=use_amp)
    AMP_BACKEND = "torch.amp"
except Exception:
    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScalerOld
    def amp_autocast(use_amp: bool):
        return _autocast_old(enabled=use_amp, dtype=torch.float16)
    def amp_scaler(use_amp: bool):
        return _GradScalerOld(enabled=use_amp)
    AMP_BACKEND = "torch.cuda.amp"


def select_device(pref: str = "auto") -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, non_blocking: bool):
    model.eval()  # important: call with parentheses
    ys, ps, logits_all = [], [], []
    for x, y in tqdm(loader, leave=False):
        if device.type == "cuda":
            x = x.to(device, non_blocking=non_blocking)
        else:
            x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu()
        ys.append(y)
        ps.append(pred)
        logits_all.append(logits.detach().cpu())
    y_true = torch.cat(ys).cpu().numpy()
    y_pred = torch.cat(ps).cpu().numpy()
    logits = torch.cat(logits_all).cpu()  # [N, C] on CPU
    return y_true, y_pred, logits


def main(cfg):
    # 0) Setup
    set_seed(cfg.get("seed", 42))
    device = select_device(cfg.get("device", "auto"))
    ensure_dir(cfg["paths"]["runs_dir"])

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = (device.type == "cuda")
    non_blocking = (device.type == "cuda")
    grad_scaler = amp_scaler(use_amp)

    # 1) Dataloaders (adjust if your function name differs)
    build_fn = _build_dataloaders
    tr_loader, va_loader, _, classes = build_fn(
        root=cfg["data"]["root"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        sampler=cfg["train"].get("sampler", "none"),
    )

    # 2) Model
    model = build_model(num_classes=len(classes)).to(device)

    # 3) Loss
    if cfg["train"].get("use_focal", False):
        criterion = FocalLoss(gamma=cfg["train"].get("focal_gamma", 2.0))
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg["train"].get("label_smoothing", 0.1)
        )

    # 4) Optimizer & Scheduler
    opt = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    total_epochs = (
        cfg["train"]["epochs_head"]
        + cfg["train"]["epochs_partial"]
        + cfg["train"]["epochs_full"]
    )
    warmup_ep = cfg["train"].get("warmup_epochs", 3)
    warm = LinearLR(opt, start_factor=0.1, total_iters=max(1, warmup_ep))
    cos = CosineAnnealingLR(opt, T_max=max(1, total_epochs - warmup_ep))
    sched = SequentialLR(opt, [warm, cos], milestones=[warmup_ep])

    runs_dir = Path(cfg["paths"]["runs_dir"])
    best_ckpt = runs_dir / cfg["paths"].get("ckpt_name", "best.ckpt")
    best_f1 = -1.0

    def fit_epochs(epochs: int):
        nonlocal best_f1
        for _ in range(epochs):
            # train
            model.train()
            total_loss = 0.0
            total_n = 0
            correct = 0

            cm = amp_autocast(use_amp) if use_amp else nullcontext()
            for x, y in tqdm(tr_loader, leave=False):
                if device.type == "cuda":
                    x = x.to(device, non_blocking=non_blocking)
                    y = y.to(device, non_blocking=non_blocking)
                else:
                    x = x.to(device)
                    y = y.to(device)

                opt.zero_grad(set_to_none=True)
                with cm:
                    logits = model(x)
                    loss = criterion(logits, y)

                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

                total_loss += loss.item() * x.size(0)
                total_n += x.size(0)
                correct += (logits.argmax(1) == y).sum().item()

            tr_loss = total_loss / max(1, total_n)
            tr_acc = correct / max(1, total_n)

            # validate
            y_true, y_pred, logits_val = evaluate(model, va_loader, device, non_blocking)
            f1 = macro_f1(y_true, y_pred)

            # ECE + Temperature scaling (optional)
            if _HAS_ECE:
                labels_t = torch.tensor(y_true, dtype=torch.long)
                ece_pre = ece_score(logits_val, labels_t, n_bins=15)
                try:
                    temp_scaler = TemperatureScaler()
                    temp_scaler.fit(logits_val, labels_t)  # CPU tensors recommended
                    T = float(temp_scaler.log_t.detach().exp().item())
                    ece_post = ece_score(logits_val / T, labels_t, n_bins=15)
                except Exception as e:
                    T = 1.0
                    ece_post = ece_pre
                    print(f"[temperature] fit failed → T=1.0 ({type(e).__name__}: {e})")

                print(
                    f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
                    f"val_f1={f1:.4f}  ECE(pre)={ece_pre:.4f}  ECE(post)={ece_post:.4f}  T={T:.3f}"
                )
            else:
                T = 1.0
                print(f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  val_f1={f1:.4f}")

            sched.step()

            # Save best (Macro F1)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(
                    {"model": model.state_dict(), "classes": classes, "temperature": float(T)},
                    best_ckpt,
                )

    # Stage 1: head only
    freeze_all(model)
    for p in getattr(model, "classifier", getattr(model, "fc", model)).parameters():
        p.requires_grad = True
    fit_epochs(cfg["train"]["epochs_head"])

    # Stage 2: partial unfreeze (top ratio)
    unfreeze_top(model, ratio=cfg["train"].get("partial_ratio", 0.33))
    fit_epochs(cfg["train"]["epochs_partial"])

    # Stage 3: unfreeze all
    unfreeze_all(model)
    fit_epochs(cfg["train"]["epochs_full"])

    print(f"best Macro F1: {best_f1:.4f}, saved: {str(best_ckpt)}")
    save_json({"best_macro_f1": float(best_f1), "ckpt": str(best_ckpt)}, runs_dir / "train_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
