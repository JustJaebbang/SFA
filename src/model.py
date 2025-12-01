import torch.nn as nn
import torchvision.models as tv

def build_model(num_classes:int):
    m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m

def freeze_all(m:nn.Module):
    for p in m.parameters(): p.requires_grad = False
    for p in m.classifier.parameters(): p.requires_grad = True

def unfreeze_top(m:nn.Module, ratio:float=0.33):
    feat = list(m.features)
    cutoff = int(len(feat)*(1.0-ratio))
    for i,blk in enumerate(feat):
        req = (i>=cutoff)
        for p in blk.parameters(): p.requires_grad = req
    for p in m.classifier.parameters(): p.requires_grad = True

def unfreeze_all(m:nn.Module):
    for p in m.parameters(): p.requires_grad = True
