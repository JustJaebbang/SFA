
import argparse, yaml, torch, json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

from .dataset import build_dataloaders
from .model import build_model
from .utils import macro_f1, plot_confusion
import torch.nn.functional as F
from .calibration import ece_score
from .transforms import build_transforms
from .gradcam import GradCAM, find_target_layer, overlay_cam_on_image

@torch.no_grad()
def infer_all(model, loader, device):
    model.eval(); ys, logits_all = [], []
    for x,y in tqdm(loader, leave=False):
        x = x.to(device); logits = model(x)
        ys.append(y); logits_all.append(logits.cpu())
    return np.concatenate(ys), torch.cat(logits_all)

def generate_cam_for_indices(model, device, dataset, classes, indices, out_dir: Path,
                             img_size: int, target_layer: str = None, gradcam_pp: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_device = next(model.parameters()).device

    # 1) 타깃 레이어 모듈 해석(문자열이면 안전하게 get_submodule, 없으면 자동 탐색)
    tgt = model.get_submodule(target_layer) if target_layer else find_target_layer(model)

    # 2) GradCAM 엔진 생성
    cam_engine = GradCAM(
        model,
        target_layer=tgt,                         # 모듈을 직접 전달
        use_cuda=(model_device.type == "cuda"),   # 모델이 올라간 디바이스 기준
        gradcam_pp=gradcam_pp
    )

    try:
        saved = []
        for i in indices:
            img_path, true_idx = dataset.samples[i]
            pil = Image.open(img_path).convert("RGB")
            x = build_transforms(img_size, False)(pil).unsqueeze(0).to(model_device)

            # pred/class/conf 계산(여기서는 no_grad로 충분)
            with torch.no_grad():
                logits = model(x)
                pred_idx = int(logits.argmax(1).item())
                conf = float(F.softmax(logits, dim=1)[0, pred_idx].item())

            # CAM 계산(엔진이 내부에서 hook/backward를 처리)
            cam_map, _ = cam_engine(x, class_idx=pred_idx)

            # overlay 생성: overlay_cam_on_image가 cmap 인자를 지원하지 않으면 cmap="jet"를 제거하세요.
            overlay = overlay_cam_on_image(cam_map, pil, alpha=0.4)  # , cmap="jet"  ← 지원 시에만 사용

            pred_name, true_name = classes[pred_idx], classes[true_idx]
            fname = out_dir / f"{Path(img_path).stem}_pred-{pred_name}_true-{true_name}_conf-{conf:.2f}.png"
            overlay.save(fname)
            saved.append(str(fname))
        return saved
    finally:
        # 훅 정리(구현 차이를 고려한 안전 가드)
        if hasattr(cam_engine, "close"):
            cam_engine.close()
        elif hasattr(cam_engine, "remove_hooks"):
            cam_engine.remove_hooks()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", required=True)
    # XAI options
    ap.add_argument("--gradcam", action="store_true", help="Save Grad-CAM overlays for selected test samples")
    ap.add_argument("--gradcam_pp", action="store_true")
    ap.add_argument("--target_layer", type=str, default=None)
    ap.add_argument("--cam_out_dir", type=str, default="report/figures/cam_eval")
    ap.add_argument("--cam_samples", type=int, default=12, help="Max number of CAM images to save")
    ap.add_argument("--tau", type=float, default=0.55, help="Low-confidence threshold")
    ap.add_argument("--cam_ok", type=int, default=8, help="정답(OK) 샘플 CAM 저장 개수")
    ap.add_argument("--cam_for", type=str, default="pred", choices=["pred","true","both"],
                    help="CAM을 어떤 클래스 기준으로 만들지")
    ap.add_argument("--cam_balance_per_class", type=int, default=0,
                    help=">0이면 OK 샘플을 클래스별로 균등 샘플링(k개)하여 CAM 생성")

    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,"r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, te, classes = build_dataloaders(cfg["data"]["root"], cfg["data"]["img_size"],
                                          cfg["data"]["batch_size"], cfg["data"]["num_workers"], sampler="none")
    sd = torch.load(args.ckpt, map_location="cpu")
    T  = float(sd.get("temperature", 1.0))

    model = build_model(len(classes))
    model.load_state_dict(sd["model"]); #model.to(device).eval()
    # (기존) model = build_model(...); sd = torch.load(...); model.load_state_dict(sd["model"])
    # 아래 한 줄을 모델 로드 후에 반드시 넣으세요.
    model = model.to(device)     # ← 모델을 device(CUDA/CPU)에 올림
    model.eval()


    # Evaluate
    y_true, logits = infer_all(model, te, device)
    probs = F.softmax(logits / T, dim=1)
    conf, y_pred = probs.max(1)
    y_true_np, y_pred_np = y_true, y_pred.numpy()
    acc = (y_true_np==y_pred_np).mean()
    f1 = macro_f1(y_true_np, y_pred_np)
    labels_t = torch.tensor(y_true_np, dtype=torch.long)

    ece_pre = ece_score(logits, labels_t, n_bins=15)
    ece_post = ece_score(logits / T, labels_t, n_bins=15)

    print(f"Test Acc={acc:.4f} MacroF1={f1:.4f} ECE(pre)={ece_pre:.4f} ECE(post)={ece_post:.4f}  T={T:.3f}")

    out_dir = Path("report")
    (out_dir/"figures").mkdir(parents=True, exist_ok=True)
    (out_dir/"tables").mkdir(exist_ok=True)
    # Confusion matrix
    from .utils import plot_confusion
    plot_confusion(y_true_np, y_pred_np, classes, out_dir/"figures/confusion_matrix.png")
    # JSON
    json.dump({"acc":float(acc), "macro_f1":float(f1), "ece_pre":float(ece_pre),
               "ece_post":float(ece_post), "temperature":float(T)},
               open(out_dir/"tables/test_report.json","w"), indent=2)

    # -------- Grad-CAM selection & saving (OK 포함) --------
    if args.gradcam:
        import numpy as np

        # 이미 위에서 계산한 값 재사용: y_true_np, y_pred_np, conf
        conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)

        # 1) 인덱스 집합 생성
        mis_idx  = np.where(y_true_np != y_pred_np)[0].tolist()                     # 오분류
        lowc_idx = np.where(conf_np < args.tau)[0].tolist()                         # 저신뢰(보정 확률 기준)
        ok_idx   = np.where((y_true_np == y_pred_np) & (conf_np >= 0.0))[0].tolist()# 정답(임계는 자유)

        # 2) 선택 로직: 오분류 → 저신뢰 → OK(별도 쿼터)
        selected, seen = [], set()

        def take(pool, k):
            cnt = 0
            for i in pool:
                if i not in seen:
                    selected.append(i); seen.add(i)
                    cnt += 1
                    if cnt >= k:
                        break

        # cam_samples는 '최대' 개수로 사용(오분류 우선, 남으면 저신뢰)
        take(mis_idx, args.cam_samples)
        remain = max(0, args.cam_samples - len(selected))
        if remain > 0:
            take(lowc_idx, remain)

        # OK(정답) 샘플은 별도 쿼터로 추가
        ok_selected = []
        if getattr(args, "cam_balance_per_class", 0) and args.cam_balance_per_class > 0:
            # 클래스별 균등 샘플링
            import collections, random
            per = args.cam_balance_per_class
            buckets = collections.defaultdict(list)
            for i in ok_idx:
                buckets[y_true_np[i]].append(i)
            for c, lst in buckets.items():
                random.shuffle(lst)
                for i in lst[:per]:
                    if i not in seen:
                        ok_selected.append(i); seen.add(i)
        else:
            # 순서대로 cam_ok개
            take(ok_idx, getattr(args, "cam_ok", 8))

        selected += ok_selected

        # 3) CAM 생성/저장
        cam_dir = Path(args.cam_out_dir) if hasattr(args, "cam_out_dir") else (out_dir/"figures"/"cam_eval")
        saved_paths = generate_cam_for_indices(
            model=model,
            device=device,
            dataset=te.dataset,
            classes=classes,
            indices=selected,
            out_dir=cam_dir,
            img_size=cfg["data"]["img_size"],
            target_layer=args.target_layer,
            gradcam_pp=args.gradcam_pp
        )

        # 4) CAM 목록 저장
        (out_dir/"tables").mkdir(exist_ok=True)
        json.dump({"cam_paths": saved_paths},
                open(out_dir/"tables"/"cam_list.json","w"), indent=2, ensure_ascii=False)
        print(f"[Grad-CAM] selected={len(selected)} → saved={len(saved_paths)} @ {cam_dir}")
    # ------------------------------------------------------

