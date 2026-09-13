"""Motor de fusão multimodal e geração de alertas."""

from __future__ import annotations

from typing import Any

from src.config import ALERT_THRESHOLDS, FUSION_WEIGHTS
from src.text.cloud_nlp import analyze_with_comprehend, generate_clinical_report
from src.text.classifier import classify_clinical_text


def fuse_modalities(
    video_result: dict[str, Any] | None = None,
    audio_result: dict[str, Any] | None = None,
    text_input: str | None = None,
) -> dict[str, Any]:
    video_result = video_result or {}
    audio_result = audio_result or {}
    scores_v = video_result.get("scores", {})

    bleeding = float(scores_v.get("bleeding", 0.0))
    emotion = float(scores_v.get("emotion_distress", 0.0))
    pose = float(scores_v.get("pose_closed", 0.0))
    audio_risk = float(audio_result.get("audio_risk_score", 0.0))

    transcript = audio_result.get("transcript") or ""
    combined_text = " ".join(t for t in [transcript, text_input or ""] if t).strip()
    text_clf = classify_clinical_text(combined_text) if combined_text else {
        "text_risk_score": 0.0,
        "top_label": "indeterminado",
        "labels": {},
    }
    cloud = analyze_with_comprehend(combined_text) if combined_text else {"engine": "empty", "sentiment": "NEUTRAL"}
    text_risk = float(text_clf.get("text_risk_score", 0.0))

    final = (
        FUSION_WEIGHTS["bleeding"] * bleeding
        + FUSION_WEIGHTS["emotion_distress"] * emotion
        + FUSION_WEIGHTS["pose_closed"] * pose
        + FUSION_WEIGHTS["audio_risk"] * audio_risk
        + FUSION_WEIGHTS["text_risk"] * text_risk
    )
    final = round(min(1.0, final), 3)

    alerts: list[dict[str, Any]] = []
    if bleeding >= 0.35:
        alerts.append(
            {
                "type": "complicacao_cirurgica_visual",
                "severity": "CRITICO" if bleeding >= 0.55 else "ATENCAO",
                "message": "Possível sangramento anômalo detectado por YOLOv8/HSV.",
                "score": bleeding,
            }
        )
    if emotion >= 0.45 or pose >= 0.5:
        alerts.append(
            {
                "type": "desconforto_psicologico_visual",
                "severity": "ATENCAO",
                "message": "Indicadores visuais de desconforto (emoção/postura).",
                "score": max(emotion, pose),
            }
        )
    labels = {**(audio_result.get("labels") or {}), **(text_clf.get("labels") or {})}
    if labels.get("violencia", 0) >= 0.35 or (emotion >= 0.5 and audio_risk >= 0.4):
        alerts.append(
            {
                "type": "possivel_violencia_domestica",
                "severity": "CRITICO",
                "message": "Padrões multimodais compatíveis com risco de violência/abuso.",
                "score": max(labels.get("violencia", 0), audio_risk),
            }
        )
    if labels.get("depressao_pos_parto", 0) >= 0.35:
        alerts.append(
            {
                "type": "risco_depressao_pos_parto",
                "severity": "ATENCAO",
                "message": "Sinais linguísticos/vocais de depressão pós-parto.",
                "score": labels.get("depressao_pos_parto", 0),
            }
        )
    if labels.get("ansiedade", 0) >= 0.35:
        alerts.append(
            {
                "type": "ansiedade_gestacional",
                "severity": "ATENCAO",
                "message": "Sinais de ansiedade em consulta pré-natal/ginecológica.",
                "score": labels.get("ansiedade", 0),
            }
        )

    # Nível agregado: maior entre score final e severidade dos alertas tipados
    if final >= ALERT_THRESHOLDS["CRITICO"]:
        level = "CRITICO"
    elif final >= ALERT_THRESHOLDS["ATENCAO"]:
        level = "ATENCAO"
    else:
        level = "INFO"

    if any(a.get("severity") == "CRITICO" for a in alerts):
        level = "CRITICO"
    elif any(a.get("severity") == "ATENCAO" for a in alerts) and level == "INFO":
        level = "ATENCAO"

    if not alerts and level != "INFO":
        alerts.append(
            {
                "type": "risco_agregado",
                "severity": level,
                "message": "Score multimodal elevado sem alerta tipado dominante.",
                "score": final,
            }
        )

    scores = {
        "bleeding": round(bleeding, 3),
        "emotion_distress": round(emotion, 3),
        "pose_closed": round(pose, 3),
        "audio_risk": round(audio_risk, 3),
        "text_risk": round(text_risk, 3),
        "final": final,
    }

    report_ctx = {
        "alerts": alerts,
        "alert_level": level,
        "scores": scores,
        "transcript": combined_text,
        "emotions": (video_result.get("emotion") or {}).get("mean_emotions"),
        "yolo_summary": {
            "frames_with_bleeding": (video_result.get("yolo") or {}).get("frames_with_bleeding"),
            "max_confidence": (video_result.get("yolo") or {}).get("max_confidence"),
            "model": (video_result.get("yolo") or {}).get("model"),
        },
    }
    report = generate_clinical_report(report_ctx)

    return {
        "alert_level": level,
        "scores": scores,
        "alerts": alerts,
        "text_classification": text_clf,
        "cloud_nlp": cloud,
        "report": report,
        "fusion_weights": FUSION_WEIGHTS,
    }
