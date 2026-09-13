#!/usr/bin/env python3
"""Gera dataset sintético YOLO (sangramento) + samples de vídeo/áudio/texto para demo."""

from __future__ import annotations

import json
import math
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SAMPLES_DIR, YOLO_DATA_DIR  # noqa: E402


def _make_scene(seed: int, with_bleed: bool, size: int = 640) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 210, dtype=np.uint8)
    # fundo "tecido" bege/rosado
    noise = rng.integers(0, 25, (size, size, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) - 30, 0, 255).astype(np.uint8)  # menos azul
    img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + 20, 0, 255).astype(np.uint8)

    # instrumento / estrutura elíptica
    center = (int(rng.integers(180, 460)), int(rng.integers(180, 460)))
    axes = (int(rng.integers(60, 120)), int(rng.integers(40, 90)))
    cv2.ellipse(img, center, axes, int(rng.integers(0, 180)), 0, 360, (180, 160, 150), -1)

    boxes: list[tuple[int, int, int, int]] = []
    if with_bleed:
        for _ in range(int(rng.integers(1, 3))):
            bw = int(rng.integers(40, 120))
            bh = int(rng.integers(30, 90))
            x = int(np.clip(center[0] + rng.integers(-80, 40) - bw // 2, 10, size - bw - 10))
            y = int(np.clip(center[1] + rng.integers(-40, 80) - bh // 2, 10, size - bh - 10))
            overlay = img.copy()
            color = (int(rng.integers(10, 40)), int(rng.integers(10, 40)), int(rng.integers(150, 220)))
            cv2.ellipse(overlay, (x + bw // 2, y + bh // 2), (bw // 2, bh // 2), 0, 0, 360, color, -1)
            img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
            # gotas
            for __ in range(8):
                gx = int(rng.integers(x, x + bw))
                gy = int(rng.integers(y, y + bh))
                cv2.circle(img, (gx, gy), int(rng.integers(2, 6)), color, -1)
            boxes.append((x, y, x + bw, y + bh))
    return img, boxes


def _to_yolo_label(box: tuple[int, int, int, int], size: int = 640) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / size
    cy = ((y1 + y2) / 2) / size
    w = (x2 - x1) / size
    h = (y2 - y1) / size
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_yolo_dataset(n_train: int = 40, n_val: int = 10) -> None:
    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = YOLO_DATA_DIR / "images" / split
        lbl_dir = YOLO_DATA_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            with_bleed = i % 5 != 0  # 80% positivos
            seed = (0 if split == "train" else 10_000) + i
            img, boxes = _make_scene(seed, with_bleed=with_bleed)
            name = f"{split}_{i:03d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
            lines = [_to_yolo_label(b) for b in boxes]
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml = f"""path: {YOLO_DATA_DIR.as_posix()}
train: images/train
val: images/val
names:
  0: bleeding
"""
    (YOLO_DATA_DIR / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"[ok] Dataset YOLO em {YOLO_DATA_DIR}")


def build_sample_video(path: Path, with_bleed: bool = True, frames: int = 45, fps: int = 15) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = w = 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(frames):
        # Mesmo gerador do dataset para maximizar recall do YOLO treinado
        img, _ = _make_scene(2000 + i, with_bleed=with_bleed and i > 8, size=w)
        writer.write(img)
    writer.release()
    print(f"[ok] Vídeo sample: {path}")


def build_tone_wav(path: Path, duration: float = 3.0, freq: float = 220.0) -> None:
    """WAV com tom + silêncios (simula hesitação) — STT pode falhar; use transcript override."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 16000
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    tone = 0.2 * np.sin(2 * math.pi * freq * t)
    # silêncios intercalados
    mask = ((t % 1.0) < 0.55).astype(np.float32)
    samples = (tone * mask * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    print(f"[ok] Áudio sample: {path}")


def build_text_samples() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    cases = {
        "consulta_pos_parto.txt": (
            "Paciente relata que está muito triste e chorando todos os dias. "
            "Diz que não consegue dormir e se sente sem energia desde o nascimento do bebê. "
            "Menciona culpa e que não se sente mãe."
        ),
        "consulta_ansiedade.txt": (
            "Gestante de 28 semanas, ansiosa e nervosa, com medo do parto. "
            "Refere coração acelerado e preocupação constante com o bebê."
        ),
        "triagem_violencia.txt": (
            "Paciente fala baixo e hesita. Relata que tem medo dele e que ele me bate. "
            "Diz que não posso falar abertamente e que sofre ameaça e controle."
        ),
        "nota_clinica_normal.txt": (
            "Consulta de rotina sem queixas. Paciente estável, humor preservado, "
            "acompanhamento pré-natal em dia."
        ),
    }
    for name, content in cases.items():
        (SAMPLES_DIR / name).write_text(content + "\n", encoding="utf-8")
    meta = {
        "demo_transcripts": {
            "pos_parto": cases["consulta_pos_parto.txt"],
            "violencia": cases["triagem_violencia.txt"],
            "ansiedade": cases["consulta_ansiedade.txt"],
        }
    }
    (SAMPLES_DIR / "demo_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Textos sample em {SAMPLES_DIR}")


def main() -> None:
    build_yolo_dataset()
    build_sample_video(SAMPLES_DIR / "consulta_com_sangramento.mp4", with_bleed=True)
    build_sample_video(SAMPLES_DIR / "consulta_sem_sangramento.mp4", with_bleed=False)
    build_tone_wav(SAMPLES_DIR / "consulta_hesitante.wav")
    build_text_samples()
    print("[done] Samples prontos.")


if __name__ == "__main__":
    main()
