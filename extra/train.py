# train.py
import argparse
import os
import json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler  # ← WeightedRandomSampler 추가
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, accuracy_score

#전이학습단계추가
import math  # ← 새로 추가


# ---------------------- Metrics ---------------------- #
class ECELoss(nn.Module):
    """
    Expected Calibration Error (ECE)
    (multi-class, softmax probs)
    """
    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)

        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = confidences.gt(bin_lower) & confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_conf_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss
    alpha:
      - None: no class weighting
      - scalar(float): same weight for all classes
      - tensor(num_classes): per-class weight (e.g., inverse freq)
    """
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
        else:
            # alpha is tensor or None
            self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: (N, C) logits
        targets: (N,) 0~C-1
        """
        log_probs = F.log_softmax(inputs, dim=1)       # (N, C)
        probs = torch.exp(log_probs)                   # (N, C)

        # gather log_probs, probs for target classes
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # (N,)

        if self.alpha is None:
            at = 1.0
        else:
            if isinstance(self.alpha, torch.Tensor):
                # per-class alpha
                at = self.alpha[targets]
            else:
                # scalar alpha
                at = self.alpha

        loss = -at * (1 - pt)**self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss



def compute_epoch_metrics(
    logits_list: List[torch.Tensor], labels_list: List[torch.Tensor]
) -> Dict[str, float]:
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    ece_criterion = ECELoss(n_bins=15)
    ece = ece_criterion(logits, labels).item()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "ece": ece,
    }


# ---------------------- Model ---------------------- #
def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# ▼▼ 여기부터 새로 추가: fine-tuning 단계 제어 함수 ▼▼
def set_finetune_phase(model: nn.Module, phase: int):
    """
    phase 0: 헤드만 학습 (classifier)
    phase 1: 헤드 + 상위 블록 학습 (features[-1] + classifier)
    phase 2: 전층 학습
    """
    # 우선 전체 동결
    for p in model.parameters():
        p.requires_grad = False

    if phase == 0:
        # classifier만 학습
        for p in model.classifier.parameters():
            p.requires_grad = True

    elif phase == 1:
        # classifier + 마지막 feature block 학습
        for p in model.classifier.parameters():
            p.requires_grad = True
        for p in model.features[-1].parameters():
            p.requires_grad = True

    elif phase == 2:
        # 전층 학습
        for p in model.parameters():
            p.requires_grad = True
# ▲▲ 여기까지 ▼▼

# ▼▼ LR 스케줄러 람다 생성 함수 ▼▼
def create_warmup_cosine_lr_lambda(
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float,
):
    """
    epoch 단위로 사용하는 warmup + cosine decay 스케줄.
    - 0 ~ warmup_epochs-1: 선형으로 base_lr까지 증가
    - 그 이후 ~ total_epochs-1: cosine으로 min_lr까지 감소
    """
    def lr_lambda(epoch: int):
        # epoch은 0부터 시작한다고 가정
        if epoch < warmup_epochs:
            # 선형 워밍업: 0 -> 1
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            if total_epochs <= warmup_epochs:
                return 1.0
            # cosine 구간 진행도 (0 ~ 1)
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            # 1 -> (min_lr/base_lr)로 내려가는 cosine
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = min_lr / base_lr
            return min_ratio + (1.0 - min_ratio) * cosine
    return lr_lambda
# ▲▲ 여기까지 ▼▼


# ---------------------- Training / Evaluation Loops ---------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0

    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        logits_list.append(outputs.detach())
        labels_list.append(labels.detach())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_epoch_metrics(logits_list, labels_list)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0

    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        logits_list.append(outputs)
        labels_list.append(labels)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_epoch_metrics(logits_list, labels_list)
    metrics["loss"] = epoch_loss
    return metrics


# ---------------------- Plotting ---------------------- #
def plot_learning_curves(
    history: Dict[str, List[float]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(history["train_accuracy"], label="train_accuracy")
    plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png")
    plt.close()

    # Macro F1
    plt.figure()
    plt.plot(history["train_macro_f1"], label="train_macro_f1")
    plt.plot(history["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "macro_f1_curve.png")
    plt.close()

    # ECE
    plt.figure()
    plt.plot(history["train_ece"], label="train_ece")
    plt.plot(history["val_ece"], label="val_ece")
    plt.xlabel("Epoch")
    plt.ylabel("ECE")
    plt.title("Expected Calibration Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "ece_curve.png")
    plt.close()


# ---------------------- Dataset ---------------------- #
def get_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int,
    num_workers: int,
    image_size: int = 224,
    resize_size: int = 256,
    rrc_scale=(0.8, 1.0),
    rrc_ratio=(3.0/4.0, 4.0/3.0),
    use_weighted_sampler: bool = False,
):
    """
    train_dir/
        class1/
        class2/ ...
    val_dir/
        class1/
        class2/ ...
    """

    # --- 학습용 transform ---
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=image_size,
                scale=rrc_scale,
                ratio=rrc_ratio,
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # --- 검증용 transform ---
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Dataset
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    class_names = train_dataset.classes

    # 클래스별 샘플 수 계산
    # ImageFolder는 train_dataset.targets 리스트를 가짐
    train_targets = np.array(train_dataset.targets)
    num_classes = len(class_names)
    class_counts = np.bincount(train_targets, minlength=num_classes)

    # WeightedRandomSampler 설정 (원하면)
    if use_weighted_sampler:
        # 클래스가 적게 나올수록 큰 weight를 갖도록 역비례
        class_weights = 1.0 / (class_counts + 1e-8)  # (C,)
        sample_weights = class_weights[train_targets]  # (N,)

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,     # sampler 사용 시 shuffle=False
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # class_counts도 함께 반환하여 FocalLoss alpha 계산에 사용
    return train_loader, val_loader, class_names, class_counts.tolist()




# ---------------------- Main ---------------------- #
def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Food Classifier (Baseline)")
    parser.add_argument("--train_dir", type=str, default="/Users/Jaebbang/SFA/data/processed/train", help="Root directory of training images")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save models and logs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_dir", type=str, default="/Users/Jaebbang/SFA/data/processed/val", help="Root directory of validation images")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true", help="Do not use ImageNet pretraining")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # ▼▼ 여기부터 새로 추가 1▼▼
    parser.add_argument("--image_size", type=int, default=224,
                        help="최종 입력 크기 (RandomResizedCrop / CenterCrop 크기)")
    parser.add_argument("--resize_size", type=int, default=256,
                        help="평가/검증 시 먼저 Resize할 크기 (ex. 256, 320 등)")

    parser.add_argument("--rrc_scale_min", type=float, default=0.8,
                        help="RandomResizedCrop scale 하한 (예: 0.8)")
    parser.add_argument("--rrc_scale_max", type=float, default=1.0,
                        help="RandomResizedCrop scale 상한 (예: 1.0)")
    parser.add_argument("--rrc_ratio_min", type=float, default=3.0/4.0,
                        help="RandomResizedCrop ratio 하한 (예: 0.75)")
    parser.add_argument("--rrc_ratio_max", type=float, default=4.0/3.0,
                        help="RandomResizedCrop ratio 상한 (예: 1.3333)")
    # ▲▲ 여기까지 새로 추가 1 ▲▲

     # ▼▼ 전이학습 단계화용 옵션 추가 ▼▼
    parser.add_argument("--head_epochs", type=int, default=3,
                        help="1단계: classifier(헤드)만 학습하는 epoch 수")
    parser.add_argument("--top_blocks_epochs", type=int, default=3,
                        help="2단계: 헤드 + 상위 블록(features[-1]) 학습 epoch 수 (그 뒤는 전층)")
    # ▲▲ 여기까지 ▼▼

    # ▼▼ LR warmup + Cosine 스케줄러 옵션 추가 ▼▼
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="워밍업 epoch 수")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Cosine decay에서 내려갈 최소 learning rate")
    # ▲▲ 여기까지 ▼▼

     # ▼▼ 클래스 불균형 관련 옵션 추가 ▼▼
    parser.add_argument("--use_weighted_sampler", action="store_true",
                        help="클래스 분포 역비례 WeightedRandomSampler 사용 여부")
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="CrossEntropy 대신 Focal Loss 사용 여부")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss gamma")
    parser.add_argument("--focal_alpha_balanced", action="store_true",
                        help="클래스 빈도 역비례로 alpha (per-class weight) 자동 계산")
    # ▲▲ 여기까지 ▲▲

    # ▼▼ 여기 추가 ▼▼
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="CrossEntropyLoss에 사용할 label smoothing 계수 (예: 0.05~0.1)")
    # ▲▲ 여기까지 ▲▲


    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataloaders
    train_loader, val_loader, class_names, class_counts = get_dataloaders(
    train_dir=args.train_dir,
    val_dir=args.val_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    image_size=args.image_size,
    resize_size=args.resize_size,
    rrc_scale=(args.rrc_scale_min, args.rrc_scale_max),
    rrc_ratio=(args.rrc_ratio_min, args.rrc_ratio_max),
    use_weighted_sampler=args.use_weighted_sampler,
)



    num_classes = len(class_names)

        # Model
    model = build_model(num_classes=num_classes, pretrained=(not args.no_pretrained))
    model = model.to(device)

    # class_counts: list[int] -> numpy array
    class_counts_arr = np.array(class_counts, dtype=np.float32)

    # --- Loss 설정 (FocalLoss vs CrossEntropy + Label Smoothing) ---
    if args.use_focal_loss:
        # alpha_balanced 옵션이면 클래스 빈도 역비례 alpha 사용
        if args.focal_alpha_balanced:
            # 빈도 비율
            freq = class_counts_arr / (class_counts_arr.sum() + 1e-8)
            # 역비례 가중치
            alpha_np = 1.0 / (freq + 1e-8)
            # 스케일 조정(선택): 평균이 1 정도가 되도록
            alpha_np = alpha_np / alpha_np.mean()

            alpha_tensor = torch.tensor(alpha_np, dtype=torch.float32, device=device)
        else:
            alpha_tensor = None

        criterion = FocalLoss(alpha=alpha_tensor, gamma=args.focal_gamma, reduction="mean")
        print(f"[Info] Using FocalLoss (gamma={args.focal_gamma}, alpha_balanced={args.focal_alpha_balanced})")

    else:
        # PyTorch의 CrossEntropyLoss(label_smoothing=ε)를 사용
        if args.label_smoothing > 0.0:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            print(f"[Info] Using CrossEntropyLoss with label_smoothing={args.label_smoothing}")
        else:
            criterion = nn.CrossEntropyLoss()
            print("[Info] Using CrossEntropyLoss (no label smoothing)")



    # Optimizer (처음에는 전체 파라미터 기준으로 만듦)
    base_lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    # LR scheduler (warmup + cosine)
    lr_lambda = create_warmup_cosine_lr_lambda(
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=base_lr,
        min_lr=args.min_lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    # History for curves
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "train_ece": [],
        "val_ece": [],
    }

    best_val_macro_f1 = -np.inf
    best_model_path = output_dir / "best_model.pth"
    class_idx_path = output_dir / "class_indices.json"
    with open(class_idx_path, "w", encoding="utf-8") as f:
        json.dump({i: c for i, c in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    total_epochs = args.epochs
    head_epochs = args.head_epochs
    top_epochs = args.top_blocks_epochs

    for epoch in range(1, total_epochs + 1):
        # ---------- 1) 현재 epoch에 따른 fine-tune phase 결정 ----------
        if epoch <= head_epochs:
            phase = 0  # 헤드만
        elif epoch <= head_epochs + top_epochs:
            phase = 1  # 헤드 + 상위 블록
        else:
            phase = 2  # 전층

        set_finetune_phase(model, phase)

        # (원하면) phase 출력
        if phase == 0:
            phase_name = "head only"
        elif phase == 1:
            phase_name = "head + top block"
        else:
            phase_name = "full fine-tune"

        # ---------- 2) 한 epoch 학습/검증 ----------
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # ---------- 3) history 기록 ----------
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        history["train_ece"].append(train_metrics["ece"])
        history["val_ece"].append(val_metrics["ece"])

        # ---------- 4) 현재 LR (scheduler 적용 전/후 중 택1) ----------
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch}/{total_epochs}] "
            f"(phase: {phase_name}) "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Train Acc: {train_metrics['accuracy']:.4f} "
            f"Val Acc: {val_metrics['accuracy']:.4f} "
            f"Train F1: {train_metrics['macro_f1']:.4f} "
            f"Val F1: {val_metrics['macro_f1']:.4f} "
            f"Train ECE: {train_metrics['ece']:.4f} "
            f"Val ECE: {val_metrics['ece']:.4f}"
        )

        # ---------- 5) best model 저장 ----------
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "class_names": class_names,
                },
                best_model_path,
            )

        # ---------- 6) epoch 끝에 scheduler.step() ----------
        scheduler.step()


    # Save final model
    final_model_path = output_dir / "last_model.pth"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "class_names": class_names,
        },
        final_model_path,
    )

    # Save history as json
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot curves
    plot_learning_curves(history, output_dir)


if __name__ == "__main__":
    main()
