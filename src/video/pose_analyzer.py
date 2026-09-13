"""Detecção de pose / postura com MediaPipe (aula FIAP 03) + fallback OpenCV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _closed_posture(landmarks) -> bool:
    """Postura fechada: braços cruzados / ombros contraídos (proxy de desconforto)."""
    ls, rs = landmarks[11], landmarks[12]
    lw, rw = landmarks[15], landmarks[16]
    shoulder_width = abs(rs.x - ls.x) + 1e-6
    wrists_close = abs(lw.x - rw.x) < shoulder_width * 0.6
    wrists_inward = (lw.x > ls.x) and (rw.x < rs.x)
    return wrists_close or wrists_inward


def _arm_up(landmarks) -> bool:
    le, re = landmarks[13], landmarks[14]
    ls, rs = landmarks[11], landmarks[12]
    return (le.y < ls.y) or (re.y < rs.y)


def _analyze_mediapipe_solutions(video_path: Path, sample_every: int, max_frames: int | None) -> dict[str, Any] | None:
    try:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            return None
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    except Exception:  # noqa: BLE001
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        pose.close()
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    closed_count = 0
    samples = 0
    arm_up = False
    arm_movements = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        if frame_idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if _closed_posture(lm):
                    closed_count += 1
                if _arm_up(lm):
                    if not arm_up:
                        arm_up = True
                        arm_movements += 1
                else:
                    arm_up = False
            samples += 1
        frame_idx += 1

    cap.release()
    pose.close()
    closed_score = (closed_count / samples) if samples else 0.0
    return {
        "modality": "video_pose",
        "model": "MediaPipe Pose",
        "frames_sampled": samples,
        "closed_frames": closed_count,
        "pose_closed_score": round(closed_score, 3),
        "arm_movements": arm_movements,
        "physio_activity_score": round(min(1.0, arm_movements / 10.0), 3),
    }


def _analyze_motion_fallback(video_path: Path, sample_every: int, max_frames: int | None) -> dict[str, Any]:
    """Fallback sem MediaPipe solutions: movimento óptico + assimetria (proxy de atividade/desconforto)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    prev = None
    motion_vals: list[float] = []
    asymmetry_vals: list[float] = []
    frame_idx = 0
    samples = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if prev is not None:
                diff = cv2.absdiff(prev, gray)
                motion_vals.append(float(np.mean(diff)) / 255.0)
                h, w = gray.shape
                left = gray[:, : w // 2]
                right = cv2.flip(gray[:, w // 2 :], 1)
                minh = min(left.shape[0], right.shape[0])
                minw = min(left.shape[1], right.shape[1])
                asym = float(np.mean(cv2.absdiff(left[:minh, :minw], right[:minh, :minw]))) / 255.0
                asymmetry_vals.append(asym)
            prev = gray
            samples += 1
        frame_idx += 1
    cap.release()

    mean_motion = float(np.mean(motion_vals)) if motion_vals else 0.0
    mean_asym = float(np.mean(asymmetry_vals)) if asymmetry_vals else 0.0
    # Baixo movimento + alta assimetria → posture "fechada"/desconforto (proxy)
    closed_score = min(1.0, (1.0 - min(1.0, mean_motion * 8)) * 0.5 + mean_asym * 0.8)
    arm_movements = int(mean_motion * 40)

    return {
        "modality": "video_pose",
        "model": "OpenCV-motion-fallback",
        "frames_sampled": samples,
        "closed_frames": int(closed_score * samples),
        "pose_closed_score": round(closed_score, 3),
        "arm_movements": arm_movements,
        "physio_activity_score": round(min(1.0, mean_motion * 10), 3),
        "mean_motion": round(mean_motion, 4),
    }


def analyze_video_pose(
    video_path: str | Path,
    sample_every: int = 5,
    max_frames: int | None = 240,
) -> dict[str, Any]:
    path = Path(video_path)
    result = _analyze_mediapipe_solutions(path, sample_every, max_frames)
    if result is not None:
        return result
    return _analyze_motion_fallback(path, sample_every, max_frames)
