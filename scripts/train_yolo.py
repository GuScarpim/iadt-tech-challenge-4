#!/usr/bin/env python3
"""Treina YOLOv8n customizado para classe `bleeding`."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, YOLO_DATA_YAML  # noqa: E402


def main() -> int:
    if not YOLO_DATA_YAML.exists():
        print("Dataset ausente. Rode: python scripts/prepare_sample_data.py")
        return 1

    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(YOLO_DATA_YAML),
        epochs=15,
        imgsz=640,
        batch=8,
        project=str(ROOT / "runs"),
        name="bleeding_yolov8n",
        exist_ok=True,
        verbose=True,
    )
    # Copia best.pt para models/
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    dest = MODELS_DIR / "bleeding_yolov8n.pt"
    if best.exists():
        shutil.copy2(best, dest)
        print(f"[ok] Pesos salvos em {dest}")
    else:
        print("[warn] best.pt não encontrado; verifique runs/")
        return 1

    # Validação rápida
    metrics = model.val(data=str(YOLO_DATA_YAML))
    print("[ok] Validação:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
