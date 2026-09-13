"""Sumarização de texto (aula FIAP 06) com fallback local."""

from __future__ import annotations

from typing import Any


def summarize_text(text: str, max_sentences: int = 3) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"summary": "", "engine": "empty"}

    # Preferência: OpenAI
    try:
        import os

        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Resuma em português, de forma clínica e objetiva, para equipe de saúde da mulher.",
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
            )
            return {"summary": resp.choices[0].message.content.strip(), "engine": "openai-gpt"}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Sumarização OpenAI falhou: {exc}")

    # Fallback extractivo simples
    parts = [p.strip() for p in text.replace("!", ".").replace("?", ".").split(".") if p.strip()]
    summary = ". ".join(parts[:max_sentences])
    if summary and not summary.endswith("."):
        summary += "."
    return {"summary": summary, "engine": "extractive-local"}
