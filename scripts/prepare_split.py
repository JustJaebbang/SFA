import argparse, shutil, random
from pathlib import Path

def split(src, dst, val_ratio=0.15, test_ratio=0.15, seed=42, move=False):
    random.seed(seed)
    src, dst = Path(src), Path(dst)
    for sp in ["train","val","test"]:
        (dst/sp).mkdir(parents=True, exist_ok=True)
    classes = [d.name for d in src.iterdir() if d.is_dir()]
    for c in classes:
        imgs = [p for p in (src/c).glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]
        random.shuffle(imgs)
        n = len(imgs); n_val = int(n*val_ratio); n_test = int(n*test_ratio)
        splits = {"train": imgs[n_val+n_test:], "val": imgs[:n_val], "test": imgs[n_val:n_val+n_test]}
        for sp, lst in splits.items():
            (dst/sp/c).mkdir(parents=True, exist_ok=True)
            for p in lst:
                out = dst/sp/c/p.name
                (shutil.move if move else shutil.copy2)(str(p), str(out))
    print("done:", dst)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true")
    args = ap.parse_args()
    split(args.src, args.dst, args.val_ratio, args.test_ratio, args.seed, args.move)
