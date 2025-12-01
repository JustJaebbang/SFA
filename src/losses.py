import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"): super().__init__(); self.gamma=gamma; self.reduction=reduction
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = ((1 - pt) ** self.gamma) * F.nll_loss(logp, target, reduction='none')
        return loss.mean() if self.reduction=="mean" else loss.sum()
