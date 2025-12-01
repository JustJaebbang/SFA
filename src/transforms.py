from torchvision import transforms

def build_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(int(img_size*1.15)),
            transforms.RandomResizedCrop(img_size, scale=(0.7,1.0), ratio=(3/4,4/3)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
