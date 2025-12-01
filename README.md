# SFA

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/prepare_split.py --src data/raw --dst data/processed --seed 42 --copy
python -m src.train --config configs/config.yaml
python -m src.eval  --config configs/config.yaml --ckpt runs/best.ckpt
python -m src.infer --config configs/config.yaml --ckpt runs/best.ckpt --image path/to/image.jpg --topk 3 --portion 1.2 --tau 0.55
```
