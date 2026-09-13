"""Análise de expressões emocionais em vídeo (DeepFace — aula FIAP 02)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Path used by Haar fallback when cv2.data is unavailable (OpenCV 5+)

from src.config import EMOTION_DISTRESS


def _analyze_deepface(frame: np.ndarray) -> dict[str, float] | None:
    try:
        from deepface import DeepFace

        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        emotions = result.get("emotion", {})
        return {k: float(v) / 100.0 for k, v in emotions.items()}
    except Exception:  # noqa: BLE001
        return None


def _analyze_heuristic(frame: np.ndarray) -> dict[str, float]:
    """Fallback sem DeepFace: usa saturação/contraste facial aproximado."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
    faces = []
    if cascade_path.exists():
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    else:
        # OpenCV 5+ pode não embarcar cascades; usa região central como proxy
        h, w = gray.shape[:2]
        faces = [(w // 4, h // 4, w // 2, h // 2)]

    if len(faces) == 0:
        return {"neutral": 1.0, "sad": 0.0, "fear": 0.0, "angry": 0.0, "happy": 0.0, "surprise": 0.0, "disgust": 0.0}

    x, y, w, h = faces[0]
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return {"neutral": 1.0, "sad": 0.0, "fear": 0.0, "angry": 0.0, "happy": 0.0, "surprise": 0.0, "disgust": 0.0}
    std = float(np.std(roi)) / 128.0
    mean = float(np.mean(roi)) / 255.0
    # Faces mais escuras / alto contraste → maior peso em fear/sad (demo)
    fear = min(0.7, std * 0.5 + (1 - mean) * 0.3)
    sad = min(0.6, (1 - mean) * 0.5)
    neutral = max(0.1, 1.0 - fear - sad)
    return {
        "neutral": neutral,
        "sad": sad,
        "fear": fear,
        "angry": 0.05,
        "happy": 0.05,
        "surprise": 0.05,
        "disgust": 0.0,
    }


def analyze_video_emotions(
    video_path: str | Path,
    sample_every: int = 15,
    max_frames: int | None = 180,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    aggregates: dict[str, list[float]] = {}
    frame_idx = 0
    used_deepface = False
    samples = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        if frame_idx % sample_every == 0:
            emotions = _analyze_deepface(frame)
            if emotions is None:
                emotions = _analyze_heuristic(frame)
            else:
                used_deepface = True
            for k, v in emotions.items():
                aggregates.setdefault(k, []).append(v)
            samples += 1
        frame_idx += 1
    cap.release()

    mean_emotions = {k: float(np.mean(v)) for k, v in aggregates.items()} if aggregates else {"neutral": 1.0}
    distress = sum(mean_emotions.get(e, 0.0) for e in EMOTION_DISTRESS)
    distress_score = min(1.0, distress)

    return {
        "modality": "video_emotion",
        "model": "DeepFace" if used_deepface else "Haar+heuristic",
        "frames_sampled": samples,
        "mean_emotions": {k: round(v, 3) for k, v in mean_emotions.items()},
        "distress_score": round(distress_score, 3),
        "dominant": max(mean_emotions, key=mean_emotions.get) if mean_emotions else "neutral",
        "fps": fps,
    }
