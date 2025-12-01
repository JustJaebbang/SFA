# src/dataset.py
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from PIL import Image

from .transforms import build_transforms


def rgb_safe_loader(path: str) -> Image.Image:
    """
    안전 로더: P/LA/RGBA → RGB로 통일.
    - P(팔레트) → RGBA로 변환 후
    - LA/RGBA는 흰 배경에 합성해 3채널 RGB로
    - 그 외 모드도 RGB 강제 변환
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()  # 파일 핸들 닫기 전 디코딩 완료

    if img.mode == "P":  # 팔레트 → 먼저 RGBA
        img = img.convert("RGBA")

    if img.mode in ("LA", "RGBA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def _make_sampler_if_needed(train_ds, mode: str):
    if mode != "weighted":
        return None

    targets = np.array([y for _, y in train_ds.samples])
    class_count = np.bincount(targets, minlength=len(train_ds.classes))
    class_weights = 1.0 / np.clip(class_count, 1, None)
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def build_dataloaders(
    root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    sampler: str = "weighted"
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    ImageFolder 기반 로더 구성.
    - loader=rgb_safe_loader 로 입력 모드 정규화
    - CUDA에서만 pin_memory=True (CPU/MPS는 False)
    """
    root_path = Path(root)
    train_ds = datasets.ImageFolder(
        str(root_path / "train"),
        loader=rgb_safe_loader,
        transform=build_transforms(img_size, train=True),
    )
    val_ds = datasets.ImageFolder(
        str(root_path / "val"),
        loader=rgb_safe_loader,
        transform=build_transforms(img_size, train=False),
    )
    test_ds = datasets.ImageFolder(
        str(root_path / "test"),
        loader=rgb_safe_loader,
        transform=build_transforms(img_size, train=False),
    )

    train_sampler = _make_sampler_if_needed(train_ds, sampler)

    use_pin = (torch.cuda.is_available() and torch.cuda.device_count() > 0)  # CUDA일 때만
    persistent = (num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )

    return train_loader, val_loader, test_loader, train_ds.classes
