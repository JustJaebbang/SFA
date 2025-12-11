# eval.py
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, accuracy_score


class ECELoss(nn.Module):
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


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_dataloader(
    data_dir: str, batch_size: int, num_workers: int, image_size: int = 224
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return loader


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, num_bins: int = 15
) -> Dict[str, float]:
    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        logits_list.append(outputs)
        labels_list.append(labels)

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    ece_criterion = ECELoss(n_bins=num_bins)
    ece = ece_criterion(logits, labels).item()

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "ece": ece,
    }

    return metrics, probs.cpu().numpy(), y_true


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    num_bins: int = 15,
) -> None:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0

    bin_accuracies = np.zeros(num_bins, dtype=np.float32)
    bin_confidences = np.zeros(num_bins, dtype=np.float32)
    bin_counts = np.zeros(num_bins, dtype=np.int32)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_counts[i] = in_bin.sum()
        if bin_counts[i] > 0:
            bin_accuracies[i] = accuracies[in_bin].mean()
            bin_confidences[i] = confidences[in_bin].mean()

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.bar(
        bin_centers,
        bin_accuracies,
        width=1.0 / num_bins,
        alpha=0.6,
        edgecolor="black",
        label="Empirical Accuracy",
    )
    plt.plot(
        bin_centers,
        bin_confidences,
        marker="o",
        color="red",
        label="Average Confidence",
    )
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 Food Classifier")
    parser.add_argument("--data_dir", type=str, default="/Users/Jaebbang/SFA/data/processed/test", help="Root directory of evaluation images")
    parser.add_argument("--checkpoint", type=str, default="/Users/Jaebbang/SFA/outputs/best_model.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_bins", type=int, default=15, help="Number of bins for ECE and reliability diagram")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names", None)
    if class_names is None:
        raise ValueError("Checkpoint must contain 'class_names' list.")

    num_classes = len(class_names)

    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Data
    loader = get_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=224,
    )

    # Evaluate
    metrics, probs, labels = evaluate(model, loader, device, num_bins=args.num_bins)

    print("Evaluation Results:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1 : {metrics['macro_f1']:.4f}")
    print(f"  ECE      : {metrics['ece']:.4f}")

    # Save metrics
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Reliability diagram
    plot_reliability_diagram(
        probs=probs,
        labels=labels,
        out_path=output_dir / "reliability_diagram.png",
        num_bins=args.num_bins,
    )


if __name__ == "__main__":
    main()
