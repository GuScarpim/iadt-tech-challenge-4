"""Testes das regras de fusão e classificação clínica."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fusion.multimodal_engine import fuse_modalities
from src.text.classifier import classify_clinical_text


def test_classify_violencia():
    r = classify_clinical_text("Tenho medo dele porque ele me bate e ameaça")
    assert r["labels"].get("violencia", 0) >= 0.35
    assert r["text_risk_score"] >= 0.35


def test_classify_pos_parto():
    r = classify_clinical_text("Estou triste e chorando, não consigo dormir, sem energia")
    assert r["labels"].get("depressao_pos_parto", 0) >= 0.35


def test_classify_normal():
    r = classify_clinical_text("Consulta de rotina sem queixas, paciente estável")
    assert r["top_label"] in {"normal", "indeterminado"} or r["text_risk_score"] < 0.35


def test_fusion_critico_bleeding():
    video = {"scores": {"bleeding": 0.8, "emotion_distress": 0.2, "pose_closed": 0.1}}
    audio = {"audio_risk_score": 0.2, "transcript": "", "labels": {}}
    out = fuse_modalities(video_result=video, audio_result=audio)
    assert out["alert_level"] in {"ATENCAO", "CRITICO"}
    types = [a["type"] for a in out["alerts"]]
    assert "complicacao_cirurgica_visual" in types


def test_fusion_violencia_multimodal():
    video = {"scores": {"bleeding": 0.0, "emotion_distress": 0.6, "pose_closed": 0.5}}
    audio = {
        "audio_risk_score": 0.7,
        "transcript": "tenho medo dele e ele me bate",
        "labels": {"violencia": 0.7},
    }
    out = fuse_modalities(video_result=video, audio_result=audio)
    types = [a["type"] for a in out["alerts"]]
    assert "possivel_violencia_domestica" in types
    assert out["alert_level"] in {"ATENCAO", "CRITICO"}


def test_fusion_info_baixo():
    video = {"scores": {"bleeding": 0.0, "emotion_distress": 0.05, "pose_closed": 0.0}}
    audio = {"audio_risk_score": 0.05, "transcript": "tudo bem", "labels": {"normal": 1.0}}
    out = fuse_modalities(video_result=video, audio_result=audio, text_input="consulta de rotina")
    assert out["alert_level"] == "INFO"
    assert out["scores"]["final"] < 0.4
