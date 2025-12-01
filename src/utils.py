import random, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# src/utils.py
import json
from pathlib import Path

def save_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # 출력 폴더 자동 생성
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(path:str):
    Path(path).mkdir(parents=True, exist_ok=True)

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def plot_confusion(y_true, y_pred, classes, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cmn = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(cmn, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True"); plt.xlabel("Pred")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
