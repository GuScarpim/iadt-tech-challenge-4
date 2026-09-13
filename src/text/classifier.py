"""Classificação clínica de texto (aula FIAP 05 — tópicos/categorização)."""

from __future__ import annotations

import re
from typing import Any

from src.config import RISK_LEXICON


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\sáàâãéêíóôõúç]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_clinical_text(text: str) -> dict[str, Any]:
    """Classifica texto clínico com léxico supervisionado leve (demo robusto offline)."""
    norm = _normalize(text or "")
    labels: dict[str, float] = {}
    hits: dict[str, list[str]] = {}

    if not norm:
        return {"labels": {}, "top_label": "indeterminado", "text_risk_score": 0.0, "hits": {}}

    for label, phrases in RISK_LEXICON.items():
        matched = [p for p in phrases if p in norm]
        if matched:
            labels[label] = min(1.0, 0.35 * len(matched))
            hits[label] = matched

    if not labels:
        # Heurística adicional por palavras isoladas
        tokens = set(norm.split())
        if tokens & {"triste", "choro", "chorando", "depressao", "depressão"}:
            labels["depressao_pos_parto"] = 0.45
            hits["depressao_pos_parto"] = list(tokens & {"triste", "choro", "chorando", "depressao", "depressão"})
        if tokens & {"medo", "ansiosa", "ansiedade", "nervosa"}:
            labels["ansiedade"] = 0.45
        if tokens & {"bate", "ameaça", "ameaca", "violencia", "violência", "machuca"}:
            labels["violencia"] = 0.6

    text_risk = max(labels.values()) if labels else 0.0
    # Violência pesa mais
    if labels.get("violencia", 0) > 0:
        text_risk = max(text_risk, min(1.0, labels["violencia"] + 0.15))

    top = max(labels, key=labels.get) if labels else "normal"
    if not labels:
        labels = {"normal": 1.0}

    return {
        "labels": {k: round(v, 3) for k, v in labels.items()},
        "top_label": top,
        "text_risk_score": round(text_risk, 3),
        "hits": hits,
    }
