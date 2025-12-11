# infer.py
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ---------------------- Model ---------------------- #
def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ---------------------- Preprocess ---------------------- #
def get_transform(image_size: int = 224):
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
    return transform


def load_image(image_path: str, image_size: int = 224):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(image_size)
    tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
    return image, tensor


# ---------------------- Grad-CAM ---------------------- #
class GradCAM:
    """
    Simple Grad-CAM implementation for EfficientNet-B0
    Uses the last feature block (model.features[-1]).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)  # (1, num_classes)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        target_score = output[0, target_class]
        target_score.backward()

        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = cam.relu()

        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam += 1e-8
        cam /= cam.max()
        return cam


def overlay_cam_on_image(
    img: np.ndarray, cam: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """
    img: H x W x 3 (RGB, uint8)
    cam: H x W (float, 0~1)
    """
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img)
    return overlay


# ---------------------- Inference ---------------------- #
@torch.no_grad()
def predict(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_names,
    device: torch.device,
) -> Tuple[int, float, Dict[int, float]]:
    model.eval()
    input_tensor = input_tensor.to(device)
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    conf, pred_idx = torch.max(probs, dim=0)

    prob_dict = {i: float(probs[i].item()) for i in range(len(class_names))}
    return int(pred_idx.item()), float(conf.item()), prob_dict


def main():
    parser = argparse.ArgumentParser(description="Inference & Grad-CAM for EfficientNet-B0 Food Classifier")
    parser.add_argument("--image", type=str, default="/Users/Jaebbang/SFA/val8.jpg", help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="/Users/Jaebbang/SFA/outputs/best_model.pth", help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="./infer_outputs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--image_size", type=int, default=224)
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

    # Build and load model
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Load image & preprocess
    pil_image, input_tensor = load_image(args.image, image_size=args.image_size)

    # Prediction
    pred_idx, conf, prob_dict = predict(model, input_tensor, class_names, device)
    pred_class = class_names[pred_idx]

    print(f"Predicted class: {pred_class} (index={pred_idx})")
    print(f"Confidence: {conf:.4f}")

    # Save raw prediction probabilities
    with open(output_dir / "prediction.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "pred_class_index": pred_idx,
                "pred_class_name": pred_class,
                "confidence": conf,
                "probabilities": {class_names[i]: prob for i, prob in prob_dict.items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Grad-CAM
    # target layer: last feature block
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(input_tensor.to(device), target_class=pred_idx)

    # Convert original PIL image to numpy
    img_np = np.array(pil_image)  # RGB uint8
    cam_overlay = overlay_cam_on_image(img_np, cam, alpha=0.4)

    # Save visualization
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cam_overlay)
    plt.title(f"Grad-CAM: {pred_class}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "gradcam.png")
    plt.close()


if __name__ == "__main__":
    main()
