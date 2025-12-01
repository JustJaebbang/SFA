from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
import numpy as np
from .transforms import build_transforms

def build_dataloaders(root, img_size, batch_size, num_workers, sampler="weighted"):
    train_ds = datasets.ImageFolder(f"{root}/train", transform=build_transforms(img_size, True))
    val_ds   = datasets.ImageFolder(f"{root}/val",   transform=build_transforms(img_size, False))
    test_ds  = datasets.ImageFolder(f"{root}/test",  transform=build_transforms(img_size, False))

    train_sampler = None
    if sampler == "weighted":
        targets = np.array([y for _, y in train_ds.samples])
        class_count = np.bincount(targets, minlength=len(train_ds.classes))
        class_w = 1.0 / np.clip(class_count, 1, None)
        sample_w = class_w[targets]
        train_sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes
