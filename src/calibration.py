import torch, torch.nn.functional as F
from torch.optim import LBFGS

class TemperatureScaler(torch.nn.Module):
    def __init__(self): super().__init__(); self.log_t = torch.nn.Parameter(torch.zeros(1))
    def forward(self, logits): return logits / torch.exp(self.log_t)
    def fit(self, logits, labels):
        criterion = torch.nn.CrossEntropyLoss()
        opt = LBFGS([self.log_t], lr=0.01, max_iter=50)
        logits, labels = logits.detach(), labels.detach()
        def closure():
            opt.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward(); return loss
        opt.step(closure)

@torch.no_grad()
def ece_score(logits, labels, n_bins=15):
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(1); acc = pred.eq(labels).float()
    bins = torch.linspace(0,1,n_bins+1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)
    for i in range(n_bins):
        m = (conf>bins[i]) & (conf<=bins[i+1])
        if m.any(): ece += m.float().mean()*(conf[m].mean()-acc[m].mean()).abs()
    return ece.item()
