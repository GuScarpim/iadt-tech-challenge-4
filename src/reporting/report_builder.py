"""Persistência de saídas (JSON + Markdown)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_run_outputs(output_dir: str | Path, payload: dict[str, Any], prefix: str = "run") -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"{prefix}_{stamp}.json"
    md_path = output_dir / f"{prefix}_{stamp}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    report = (payload.get("fusion") or {}).get("report", {}).get("report", "")
    level = (payload.get("fusion") or {}).get("alert_level", "INFO")
    alerts = (payload.get("fusion") or {}).get("alerts", [])
    md = [
        f"# Resultado multimodal — {stamp}",
        "",
        f"**Nível:** {level}",
        "",
        "## Alertas",
    ]
    for a in alerts:
        md.append(f"- **{a.get('type')}** ({a.get('severity')}): {a.get('message')}")
    if not alerts:
        md.append("- Nenhum")
    md.extend(["", "## Relatório", "", report, "", f"JSON completo: `{json_path.name}`"])
    md_path.write_text("\n".join(md), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}
