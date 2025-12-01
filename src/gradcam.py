
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


def overlay_cam_on_image(cam_map, pil_img, alpha=0.4, cmap="jet"):
    """
    cam_map: torch.Tensor or np.ndarray, shape (H, W) or (1, H, W), 값 범위 임의
    pil_img: PIL.Image
    alpha: heatmap 가중치(0~1)
    cmap: matplotlib 컬러맵 이름(str) 또는 None(기본 'jet')
    """
    # 1) cam_map -> numpy [H,W], 0~1 정규화
    if hasattr(cam_map, "detach"):
        cam = cam_map.detach().cpu().numpy()
    else:
        cam = np.array(cam_map)
    if cam.ndim == 3 and cam.shape[0] == 1:
        cam = cam[0]
    if cam.ndim != 2:
        raise ValueError(f"cam_map must be 2D, got shape {cam.shape}")
    cam = cam.astype(np.float32)
    if np.isnan(cam).any() or np.isinf(cam).any():
        cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    # 정규화
    cmin, cmax = cam.min(), cam.max()
    if cmax > cmin:
        cam_norm = (cam - cmin) / (cmax - cmin)
    else:
        cam_norm = np.zeros_like(cam, dtype=np.float32)

    # 2) heatmap 생성(컬러맵 적용)
    try:
        import matplotlib.cm as cm
        cmap_obj = cm.get_cmap(cmap or "jet")
        heatmap = cmap_obj(cam_norm)[:, :, :3]  # RGB, shape (H,W,3), 0~1
    except Exception:
        # matplotlib이 없거나 cmap 이름이 잘못되면 기본 'jet' 유사 처리
        heatmap = np.stack([cam_norm, np.zeros_like(cam_norm), 1.0 - cam_norm], axis=-1)

    # 3) 크기 맞추기 + 오버레이
    img = pil_img.convert("RGB")
    if (img.size[1], img.size[0]) != heatmap.shape[:2]:
        heatmap = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0

    img_np = np.array(img).astype(np.float32) / 255.0
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(overlay)


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
