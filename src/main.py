#!/usr/bin/env python3
"""CLI principal — pipeline multimodal Saúde da Mulher (Tech Challenge Fase 4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.voice_risk import analyze_audio_risk
from src.config import OUTPUT_DIR
from src.fusion.multimodal_engine import fuse_modalities
from src.reporting.report_builder import save_run_outputs
from src.video.pipeline import run_video_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Análise multimodal (vídeo YOLOv8 + áudio + texto) para saúde da mulher"
    )
    p.add_argument("--video", type=str, help="Caminho do vídeo clínico")
    p.add_argument("--audio", type=str, help="Caminho de áudio de consulta (wav/mp3/mp4)")
    p.add_argument("--text", type=str, help="Texto clínico / transcrição manual")
    p.add_argument("--text-file", type=str, help="Arquivo de texto clínico")
    p.add_argument("--transcript-override", type=str, help="Força transcrição (demo sem STT real)")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--skip-emotion", action="store_true")
    p.add_argument("--skip-pose", action="store_true")
    p.add_argument("--prefix", type=str, default="run")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not any([args.video, args.audio, args.text, args.text_file]):
        print("Informe ao menos --video, --audio, --text ou --text-file", file=sys.stderr)
        return 2

    text_input = args.text or ""
    if args.text_file:
        text_input = Path(args.text_file).read_text(encoding="utf-8")

    video_result = None
    if args.video:
        print(f"[video] Processando {args.video} ...")
        video_result = run_video_pipeline(
            args.video,
            output_dir=output_dir,
            enable_emotion=not args.skip_emotion,
            enable_pose=not args.skip_pose,
        )
        print(
            f"[video] bleeding={video_result['scores']['bleeding']} "
            f"distress={video_result['scores']['emotion_distress']} "
            f"pose={video_result['scores']['pose_closed']}"
        )

    audio_result = None
    if args.audio or args.transcript_override:
        audio_source = args.audio or args.video
        if not audio_source and args.transcript_override:
            # Cria WAV silencioso só para prosódia quando há override sem arquivo
            silent = output_dir / "audio_work" / "silent_override.wav"
            silent.parent.mkdir(parents=True, exist_ok=True)
            from src.audio.transcriber import _write_silent_wav

            _write_silent_wav(silent, duration_sec=2.0)
            audio_source = str(silent)
        if audio_source:
            print(f"[audio] Processando {audio_source} ...")
            audio_result = analyze_audio_risk(
                audio_source,
                work_dir=output_dir / "audio_work",
                transcript_override=args.transcript_override,
            )
            print(
                f"[audio] risk={audio_result['audio_risk_score']} "
                f"top={audio_result['top_label']} engine={audio_result['stt_engine']}"
            )
    elif args.video:
        print(f"[audio] Extraindo áudio de {args.video} ...")
        audio_result = analyze_audio_risk(
            args.video,
            work_dir=output_dir / "audio_work",
            transcript_override=None,
        )
        print(
            f"[audio] risk={audio_result['audio_risk_score']} "
            f"top={audio_result['top_label']} engine={audio_result['stt_engine']}"
        )

    print("[fusion] Combinando modalidades ...")
    fusion = fuse_modalities(video_result=video_result, audio_result=audio_result, text_input=text_input)
    print(f"[fusion] nível={fusion['alert_level']} final={fusion['scores']['final']}")

    payload = {
        "video": video_result,
        "audio": audio_result,
        "text_input": text_input,
        "fusion": fusion,
    }
    paths = save_run_outputs(output_dir, payload, prefix=args.prefix)
    print(f"[ok] JSON: {paths['json']}")
    print(f"[ok] MD:   {paths['markdown']}")
    print(json.dumps({"alert_level": fusion["alert_level"], "scores": fusion["scores"], "alerts": fusion["alerts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
