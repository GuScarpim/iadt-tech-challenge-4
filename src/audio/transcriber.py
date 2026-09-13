"""Transcrição de áudio/vídeo (MoviePy + SpeechRecognition — aula FIAP 04)."""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Any


def extract_audio_wav(media_path: str | Path, wav_path: str | Path) -> Path:
    """Extrai áudio mono 16kHz WAV a partir de vídeo ou áudio."""
    media_path = Path(media_path)
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        try:
            from moviepy import AudioFileClip, VideoFileClip  # moviepy >= 2
        except ImportError:
            from moviepy.editor import AudioFileClip, VideoFileClip  # moviepy 1.x

        if media_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            clip = VideoFileClip(str(media_path))
            audio = clip.audio
        else:
            audio = AudioFileClip(str(media_path))
            clip = None

        if audio is None:
            raise ValueError("Mídia sem trilha de áudio")

        tmp = str(wav_path.with_suffix(".tmp.wav"))
        # moviepy 2 usa write_audiofile sem verbose/logger iguais
        try:
            audio.write_audiofile(tmp, fps=16000, nbytes=2, codec="pcm_s16le", logger=None)
        except TypeError:
            audio.write_audiofile(tmp)
        audio.close()
        if clip is not None:
            clip.close()

        try:
            from pydub import AudioSegment

            seg = AudioSegment.from_file(tmp)
            seg = seg.set_channels(1).set_frame_rate(16000)
            seg.export(str(wav_path), format="wav")
            os.remove(tmp)
        except Exception:  # noqa: BLE001
            Path(tmp).rename(wav_path)
        return wav_path
    except Exception as exc:  # noqa: BLE001
        # Se já for WAV válido, apenas copia
        if media_path.suffix.lower() == ".wav":
            import shutil

            shutil.copy2(media_path, wav_path)
            return wav_path
        print(f"[WARN] Extração de áudio falhou ({exc}); gerando WAV silencioso de demo.")
        return _write_silent_wav(wav_path, duration_sec=2.0)


def _write_silent_wav(path: Path, duration_sec: float = 2.0, rate: int = 16000) -> Path:
    n = int(duration_sec * rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n)
    return path


def transcribe_wav(wav_path: str | Path, language: str = "pt-BR") -> dict[str, Any]:
    """Transcreve WAV com SpeechRecognition; tenta OpenAI Whisper se disponível."""
    wav_path = Path(wav_path)
    text = ""
    engine = "none"

    # Tentativa 1: OpenAI Whisper API
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            with open(wav_path, "rb") as f:
                result = client.audio.transcriptions.create(model="whisper-1", file=f, language="pt")
            text = result.text.strip()
            engine = "openai-whisper"
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Whisper API falhou: {exc}")

    # Tentativa 2: SpeechRecognition (Google free API / sphinx)
    if not text:
        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()
            with sr.AudioFile(str(wav_path)) as source:
                audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio, language=language)
                engine = "speechrecognition-google"
            except sr.UnknownValueError:
                text = ""
                engine = "speechrecognition-empty"
            except sr.RequestError:
                try:
                    text = recognizer.recognize_sphinx(audio)
                    engine = "speechrecognition-sphinx"
                except Exception:  # noqa: BLE001
                    text = ""
                    engine = "speechrecognition-failed"
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] SpeechRecognition falhou: {exc}")

    return {"transcript": text, "engine": engine, "wav_path": str(wav_path)}


def transcribe_media(media_path: str | Path, work_dir: str | Path) -> dict[str, Any]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / f"{Path(media_path).stem}.wav"
    extract_audio_wav(media_path, wav_path)
    result = transcribe_wav(wav_path)
    result["source"] = str(media_path)
    return result
