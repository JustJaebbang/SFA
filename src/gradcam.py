
"""
Grad-CAM and Grad-CAM++ utility.

Usage (minimal):
    from gradcam import GradCAM, find_target_layer, overlay_cam_on_image
    cam = GradCAM(model, target_layer=find_target_layer(model), use_cuda=(device.type=="cuda"))
    heatmap, class_idx = cam(input_tensor)  # input_tensor: [1,3,H,W], normalized like train/eval
    overlay = overlay_cam_on_image(heatmap, pil_image)
    overlay.save("cam.png")

Dependencies: torch, torchvision, Pillow, numpy, matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# matplotlib is only used for colormap; import lazily
try:
    import matplotlib.cm as cm
except Exception:  # pragma: no cover
    cm = None


def _to_numpy_img(img: Image.Image) -> np.ndarray:
    """PIL RGB -> HxWx3 float32 [0,1]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def overlay_cam_on_image(cam: np.ndarray, image: Image.Image, alpha: float = 0.35,
                         colormap: str = "jet") -> Image.Image:
    """
    Overlay a CAM heatmap onto a PIL image.
    - cam: 2D array in [0,1], shape (H, W)
    - image: PIL.Image (RGB)
    - alpha: heatmap transparency
    - colormap: matplotlib colormap name
    Returns: PIL.Image with overlay.
    """
    if cam.ndim != 2:
        raise ValueError("cam must be 2D array (H, W) in [0,1]")
    base = _to_numpy_img(image)
    H, W, _ = base.shape
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    cam_resized = np.asarray(cam_resized).astype(np.float32) / 255.0

    if cm is None:
        # Fallback: simple red overlay without matplotlib
        heat_rgb = np.stack([cam_resized, np.zeros_like(cam_resized), np.zeros_like(cam_resized)], axis=-1)
    else:
        cmap = cm.get_cmap(colormap)
        heat_rgb = cmap(cam_resized)[:, :, :3]  # drop alpha

    out = (1 - alpha) * base + alpha * heat_rgb
    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))


def find_target_layer(model: nn.Module) -> str:
    """
    Heuristic to find the last Conv2d layer name for Grad-CAM.
    Traverses modules in depth-first order and returns the last nn.Conv2d name.
    """
    last_name = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_name = name
    if last_name is None:
        raise ValueError("No Conv2d layer found in the model. Please specify target_layer manually.")
    return last_name


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Retrieve a submodule by dotted path name."""
    module = model
    if not name:
        return module
    for attr in name.split("."):
        if attr.isdigit():
            module = module[int(attr)]  # for Sequential indices
        else:
            module = getattr(module, attr)
    return module


@dataclass
class _HookStore:
    activations: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None


class GradCAM:
    """
    Generic Grad-CAM / Grad-CAM++ implementation.

    Args:
        model: nn.Module (classification model)
        target_layer: layer name (str) or nn.Module whose activations to visualize
        use_cuda: if True, expects inputs/model on CUDA
        gradcam_pp: if True, use Grad-CAM++ weighting; else original Grad-CAM
    """
    def __init__(self,
                 model: nn.Module,
                 target_layer: Union[str, nn.Module, None],
                 use_cuda: bool = False,
                 gradcam_pp: bool = False):
        self.model = model
        self.model.eval()
        self.gradcam_pp = gradcam_pp
        self.device = torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")

        # Resolve target layer
        if target_layer is None:
            target_layer = find_target_layer(model)
        if isinstance(target_layer, str):
            self.target_module = _get_module_by_name(model, target_layer)
            self.target_layer_name = target_layer
        elif isinstance(target_layer, nn.Module):
            self.target_module = target_layer
            # Try to get its dotted name (optional)
            self.target_layer_name = None
        else:
            raise TypeError("target_layer must be str, nn.Module, or None")

        self.hook_store = _HookStore()
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.hook_store.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            # grad_out is a tuple, we need grad w.r.t. outputs (same shape as activations)
            self.hook_store.gradients = grad_out[0].detach()

        self._remove_handles_if_any()
        self._handles = [
            self.target_module.register_forward_hook(fwd_hook),
            self.target_module.register_full_backward_hook(bwd_hook),
        ]

    def _remove_handles_if_any(self):
        if hasattr(self, "_handles"):
            for h in self._handles:
                try:
                    h.remove()
                except Exception:
                    pass
        self._handles = []

    def __del__(self):
        self._remove_handles_if_any()

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> np.ndarray:
        # cam: [H, W] tensor
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy()

    def _compute_weights(self, grads: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        """
        Compute channel weights.
        - For Grad-CAM: global-average of gradients over spatial dims.
        - For Grad-CAM++: per-channel alpha coefficients (per paper).
        grads: [N, C, H, W], acts: [N, C, H, W]
        Returns weights: [N, C]
        """
        if not self.gradcam_pp:
            # Grad-CAM: GAP over HxW
            weights = grads.mean(dim=(2, 3))  # [N, C]
            return weights

        # Grad-CAM++
        # alpha_k = sum over ij of (d2Y/dA^2) / (2 * d2Y/dA^2 + sum over ij A_ij * d3Y/dA^3)
        # Practical simplified implementation as in common repos.
        grads2 = grads.pow(2)
        grads3 = grads.pow(3)
        # Prevent divide by zero
        eps = 1e-8
        numerator = grads2
        denominator = 2 * grads2 + (acts * grads3).sum(dim=(2, 3), keepdim=True)
        denominator = torch.where(denominator != 0.0, denominator, torch.full_like(denominator, eps))
        alphas = numerator / denominator  # [N, C, H, W]
        positive_grads = torch.relu(grads)  # only positive gradients
        weights = (alphas * positive_grads).sum(dim=(2, 3))  # [N, C]
        return weights

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None
                 ) -> Tuple[np.ndarray, int]:
        """
        Compute CAM for a single image tensor.
        Args:
            input_tensor: [1, 3, H, W] Tensor on the same device as model (or will be moved)
            class_idx: if None, use argmax of logits as target class
        Returns:
            heatmap (H, W) in [0,1], class_idx used
        """
        was_training = self.model.training
        self.model.eval()

        # Ensure device
        x = input_tensor.to(self.device)
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        # Forward
        logits = self.model(x)  # [1, C]
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, class_idx]

        # Backward to get gradients at target layer
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        # Retrieve stored activations and gradients
        acts = self.hook_store.activations  # [1, C, H, W]
        grads = self.hook_store.gradients   # [1, C, H, W]
        if acts is None or grads is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Check target_layer.")

        # Compute weights and CAM
        weights = self._compute_weights(grads, acts)  # [1, C]
        cam = (weights[:, :, None, None] * acts).sum(dim=1)  # [1, H, W]
        cam = torch.relu(cam)[0]  # [H, W]

        heatmap = self._normalize_cam(cam)

        # Restore training mode if needed
        if was_training:
            self.model.train()

        return heatmap, class_idx
