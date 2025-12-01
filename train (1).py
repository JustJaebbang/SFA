
import argparse
import yaml
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

# -----------------------------
# Safe imports from project modules
# -----------------------------
def _safe_import_dataset():
    # Try both build_dataloaders and build_loaders
    try:
        from .dataset import build_dataloaders as _fn
        return _fn
    except Exception:
        from .dataset import build_loaders as _fn  # type: ignore
        return _fn

def _safe_import_model():
    try:
        from .model import build_model, freeze_all, unfreeze_top, unfreeze_all  # type: ignore
        return build_model, freeze_all, unfreeze_top, unfreeze_all
    except Exception as e:
        raise ImportError("model.py must define build_model, freeze_all, unfreeze_top, unfreeze_all") from e

def _safe_import_utils():
    try:
        from .utils import set_seed, ensure_dir, macro_f1, save_json  # type: ignore
        return set_seed, ensure_dir, macro_f1, save_json
    except Exception as e:
        raise ImportError("utils.py must define set_seed, ensure_dir, macro_f1, save_json") from e

def _safe_import_calibration():
    try:
        from .calibration import ece_score, TemperatureScaler  # type: ignore
        return True, ece_score, TemperatureScaler
    except Exception:
        return False, None, None

build_dataloaders = _safe_import_dataset()
build_model, freeze_all, unfreeze_top, unfreeze_all = _safe_import_model()
set_seed, ensure_dir, macro_f1, save_json = _safe_import_utils()
_HAS_CAL, ece_score, TemperatureScaler = _safe_import_calibration()

# -----------------------------
# AMP compatibility wrapper (torch.amp preferred, fallback to torch.cuda.amp)
# -----------------------------
from contextlib import nullcontext

try:
    from torch.amp import autocast as _autocast_new, GradScaler as _GradScalerNew  # type: ignore
    def amp_autocast(use_amp: bool):
        # CUDA 전용 AMP
        return _autocast_new(device_type="cuda", enabled=use_amp, dtype=torch.float16)
    def amp_scaler(use_amp: bool):
        # 일부 버전에선 device_type 인자를 받지 않음
        return _GradScalerNew(enabled=use_amp)
    AMP_BACKEND = "torch.amp"
except Exception:
    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScalerOld  # type: ignore
    def amp_autocast(use_amp: bool):
        return _autocast_old(enabled=use_amp, dtype=torch.float16)
    def amp_scaler(use_amp: bool):
        return _GradScalerOld(enabled=use_amp)
    AMP_BACKEND = "torch.cuda.amp"


# -----------------------------
# Device selection
# -----------------------------
def select_device(dev_arg: str = "auto") -> torch.device:
    if dev_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if dev_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if dev_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, non_blocking: bool):
    model.eval()  # 괄호 필수
    ys, ps, logits_all = [], [], []
    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=non_blocking) if device.type == "cuda" else x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu()
        ys.append(y)
        ps.append(pred)
        logits_all.append(logits.detach().cpu())

    y_true = torch.cat(ys).cpu().numpy()
    y_pred = torch.cat(ps).cpu().numpy()
    logits = torch.cat(logits_all).cpu()  # [N, C]
    return y_true, y_pred, logits


# -----------------------------
# Main training
# -----------------------------
def main(cfg):
    # 0) setup
    set_seed(cfg.get("seed", 42))
    device = select_device(cfg.get("device", "auto"))
    runs_dir = Path(cfg.get("paths", {}).get("runs_dir", "runs"))
    ensure_dir(str(runs_dir))

    use_amp = (device.type == "cuda")
    non_blocking = (device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # 1) data
    data_cfg = cfg.get("data", {})
    tr_loader, va_loader, te_loader, classes = build_dataloaders(
        root=data_cfg.get("root", "data/processed"),
        img_size=data_cfg.get("img_size", 224),
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 2),
        sampler=cfg.get("train", {}).get("sampler", "weighted"),
    )

    # 2) model
    model = build_model(num_classes=len(classes)).to(device)

    # 3) criterion
    train_cfg = cfg.get("train", {})
    if train_cfg.get("use_focal", False):
        try:
            from .losses import FocalLoss  # type: ignore
        except Exception as e:
            raise ImportError("use_focal=True 이면 src/losses.py에 FocalLoss가 필요합니다.") from e
        criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.1)))

    # 4) optimizer & scheduler
    base_lr = float(train_cfg.get("lr", 3e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    epochs_head = int(train_cfg.get("epochs_head", 1))
    epochs_partial = int(train_cfg.get("epochs_partial", 1))
    epochs_full = int(train_cfg.get("epochs_full", 5))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 1))

    # freeze for head training
    freeze_all(model)
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, weight_decay=wd)
    total_epochs = epochs_head + epochs_partial + epochs_full
    warm = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_epochs))
    cos = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, [warm, cos], milestones=[max(1, warmup_epochs)])

    # AMP scaler (항상 생성, CUDA일 때만 활성)
    grad_scaler = amp_scaler(use_amp)

    best_f1 = -1.0
    best_ckpt = runs_dir / cfg.get("paths", {}).get("ckpt_name", "best.ckpt")

    def fit_epochs(num_epochs: int):
        nonlocal best_f1, model, optimizer, scheduler

        for _ in range(num_epochs):
            model.train()
            total_loss = 0.0
            total_n = 0
            correct = 0

            for x, y in tqdm(tr_loader, leave=False):
                x = x.to(device, non_blocking=non_blocking) if device.type == "cuda" else x.to(device)
                y = y.to(device, non_blocking=non_blocking) if device.type == "cuda" else y.to(device)

                optimizer.zero_grad(set_to_none=True)
                with (amp_autocast(use_amp) if use_amp else nullcontext()):
                    logits = model(x)
                    loss = criterion(logits, y)

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                total_loss += float(loss.detach().cpu().item()) * x.size(0)
                total_n += x.size(0)
                correct += int((logits.argmax(1) == y).sum().detach().cpu().item())

            tr_loss = total_loss / max(1, total_n)
            tr_acc = correct / max(1, total_n)

            # Validation
            y_true, y_pred, logits_val = evaluate(model, va_loader, device, non_blocking)
            f1 = float(macro_f1(y_true, y_pred))

            # Calibration: ECE pre/post + Temperature
            if _HAS_CAL:
                labels_t = torch.tensor(y_true, dtype=torch.long)
                ece_pre = float(ece_score(logits_val, labels_t, n_bins=15))
                try:
                    temp_scaler = TemperatureScaler()
                    temp_scaler.fit(logits_val, labels_t)  # CPU 텐서 권장
                    T = float(temp_scaler.log_t.detach().exp().item())
                    ece_post = float(ece_score(logits_val / T, labels_t, n_bins=15))
                except Exception as e:
                    T = 1.0
                    ece_post = ece_pre
                    print(f"[temperature] fit failed → fallback T=1.0 ({type(e).__name__}: {e})")
                print(f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  val_f1={f1:.4f}  "
                      f"ECE(pre)={ece_pre:.4f}  ECE(post)={ece_post:.4f}  T={T:.3f}")
            else:
                T = 1.0
                print(f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  val_f1={f1:.4f}")

            scheduler.step()

            # Save best (Macro F1)
            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    "model": model.state_dict(),
                    "classes": classes,
                    "temperature": float(T),
                }, best_ckpt)

    # 5) staged fine-tuning
    # Stage 1: head only
    fit_epochs(epochs_head)

    # Stage 2: unfreeze top part
    unfreeze_top(model, ratio=float(train_cfg.get("unfreeze_ratio", 0.33)))
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, weight_decay=wd)
    fit_epochs(epochs_partial)

    # Stage 3: unfreeze all
    unfreeze_all(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, weight_decay=wd)
    fit_epochs(epochs_full)

    # 6) summary
    summary_path = runs_dir / "train_summary.json"
    save_json({"best_macro_f1": float(best_f1), "ckpt": str(best_ckpt), "amp_backend": AMP_BACKEND}, summary_path)
    print(f"best Macro F1: {best_f1:.4f}, saved: {str(best_ckpt)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
