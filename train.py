# src/train.py
import argparse
import yaml
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from .dataset import build_dataloaders
from .model import build_model, freeze_all, unfreeze_top, unfreeze_all
from .losses import FocalLoss
from .utils import set_seed, ensure_dir, macro_f1, save_json

# 선택: ece_score가 없으면 주석 처리해도 동작합니다.
try:
    from .calibration import TemperatureScaler, ece_score
    _HAS_ECE = True
except Exception:
    _HAS_ECE = False


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


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, use_non_blocking: bool) -> Tuple:
    model.eval()
    ys, ps, logits_all = [], [], []
    for x, y in tqdm(loader, leave=False):
        if device.type == "cuda":
            x = x.to(device, non_blocking=use_non_blocking)
        else:
            x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu()
        ys.append(y)
        ps.append(pred)
        logits_all.append(logits.detach().cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    logits = torch.cat(logits_all)  # [N, C] on CPU
    return y_true, y_pred, logits


def main(cfg):
    # 0) 기본 세팅
    set_seed(cfg.get("seed", 42))
    device = select_device(cfg.get("device", "auto"))
    ensure_dir(cfg["paths"]["runs_dir"])

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        use_amp = True
        use_non_blocking = True
    else:
        use_amp = False
        use_non_blocking = False

    # 1) 데이터 로더
    tr_loader, va_loader, _, classes = build_dataloaders(
        root=cfg["data"]["root"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        sampler=cfg["train"]["sampler"],
    )

    # 2) 모델
    model = build_model(num_classes=len(classes)).to(device)

    # 3) 손실함수
    if cfg["train"].get("use_focal", False):
        criterion = FocalLoss(gamma=cfg["train"].get("focal_gamma", 2.0))
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg["train"].get("label_smoothing", 0.1)
        )

    # 4) 옵티마 & 스케줄
    #    - 단계별 언프리즈가 바뀌어도 optimizer는 재사용
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
    warmup_ep = cfg["train"]["warmup_epochs"]
    warm = LinearLR(opt, start_factor=0.1, total_iters=warmup_ep)
    cos = CosineAnnealingLR(opt, T_max=max(1, total_epochs - warmup_ep))
    sched = SequentialLR(opt, [warm, cos], milestones=[warmup_ep])

    # 5) AMP 준비
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 6) 체크포인트 설정
    runs_dir = Path(cfg["paths"]["runs_dir"])
    best_ckpt = runs_dir / cfg["paths"]["ckpt_name"]
    best_f1 = -1.0

    def fit_epochs(epochs: int):
        nonlocal best_f1
        model.train()

        for _ in range(epochs):
            # 학습 루프
            total_loss = 0.0
            total_n = 0
            correct = 0

            for x, y in tqdm(tr_loader, leave=False):
                if device.type == "cuda":
                    x = x.to(device, non_blocking=use_non_blocking)
                    y = y.to(device, non_blocking=use_non_blocking)
                else:
                    x = x.to(device)
                    y = y.to(device)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                total_loss += loss.item() * x.size(0)
                total_n += x.size(0)
                correct += (logits.argmax(1) == y).sum().item()

            tr_loss = total_loss / max(1, total_n)
            tr_acc = correct / max(1, total_n)

            # 검증
            y_true, y_pred, logits_val = evaluate(model, va_loader, device, use_non_blocking)
            f1 = macro_f1(y_true, y_pred)
            
            if _HAS_ECE:
                import torch as _torch
                labels_t = _torch.tensor(y_true, dtype=_torch.long)
                ece_pre = ece_score(logits_val, labels_t, n_bins=15)

                 # 검증셋으로 Temperature 학습(로짓/라벨은 CPU로 두는 편이 L-BFGS 안정적)
                try:
                    scaler = TemperatureScaler()  # 내부 파라미터 log_t
                    scaler.fit(logits_val, labels_t)     # T 학습
                    T = float(scaler.log_t.detach().exp().item())
                    # 보정 후 ECE
                    ece_post = ece_score(logits_val / T, labels_t, n_bins=15)
                except Exception as _e:
                    # 예외 시 안전 폴백
                    T = 1.0
                    ece_post = ece_pre
                    print(f"[temperature] fit failed → fallback T=1.0 ({type(_e).__name__}: {_e})")

                print(f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  val_f1={f1:.4f} \
                        ECE(pre)={ece_pre:.4f} ECE(post)={ece_post:.4f}  T={T:.3f}")
            else:
                T = 1.0
                print(f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  val_f1={f1:.4f}")

            # 스케줄러
            sched.step()

            # 베스트 저장(Macro F1 기준)
            if f1 > best_f1:
                best_f1 = f1
                torch.save({"model": model.state_dict(), "classes": classes, "temperature": float(T)}, best_ckpt)

        return

    # 7) 단계적 미세조정
    # Stage 1: 헤드만 학습
    freeze_all(model)
    for p in model.classifier.parameters():
        p.requires_grad = True
    fit_epochs(cfg["train"]["epochs_head"])

    # Stage 2: 상단 블록 언프리즈(예: 33%)
    unfreeze_top(model, ratio=0.33)
    fit_epochs(cfg["train"]["epochs_partial"])

    # Stage 3: 전층 언프리즈
    unfreeze_all(model)
    fit_epochs(cfg["train"]["epochs_full"])

    # 8) 결과 로그 저장
    print(f"best Macro F1: {best_f1:.4f}, saved: {str(best_ckpt)}")
    save_json({"best_macro_f1": float(best_f1), "ckpt": str(best_ckpt)}, runs_dir / "train_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
