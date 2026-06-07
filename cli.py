#!/usr/bin/env python3
"""
cli.py — Tarteel PoC command-line interface
--------------------------------------------

Two modes:

  1. FILE mode (default)
     Process an existing audio file end-to-end.

     python cli.py file path/to/recitation.wav
     python cli.py file recitation.wav --surah 1 --ayah 1
     python cli.py file recitation.wav --surah 2          # search within surah

  2. REALTIME mode
     Stream from microphone in real time.

     python cli.py realtime
     python cli.py realtime --surah 1 --ayah 1            # lock to a verse

Options
-------
  --surah   INT    Quran surah number (1–114)
  --ayah    INT    Ayah number within surah
  --device  STR    Compute device: cuda | cpu | mps
  --model   STR    HuggingFace model ID (default: tarteel-ai/whisper-base-ar-quran)
  --threshold FLOAT Minimum fuzzy match score 0–100 (default: 40)
  --timestamps      Return word-level timestamps (file mode only)
  --verbose         Enable DEBUG logging

Examples
--------
  # Transcribe a local WAV and auto-detect the verse
  python cli.py file sample.wav

  # Align against Al-Fatiha exactly
  python cli.py file sample.wav --surah 1 --ayah 1

  # Real-time mic, aligned against Al-Baqarah
  python cli.py realtime --surah 2
"""

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tarteel-poc",
        description="Tarteel Quran recitation recognition — proof of concept",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── file mode ─────────────────────────────────────────────────────
    file_p = subparsers.add_parser("file", help="Process an audio file")
    file_p.add_argument("audio", type=str, help="Path to audio file")
    file_p.add_argument("--surah", type=int, default=None)
    file_p.add_argument("--ayah", type=int, default=None)
    file_p.add_argument(
        "--timestamps", action="store_true", help="Return word-level timestamps"
    )

    # ── realtime mode ─────────────────────────────────────────────────
    rt_p = subparsers.add_parser("realtime", help="Stream from microphone")
    rt_p.add_argument("--surah", type=int, default=None)
    rt_p.add_argument("--ayah", type=int, default=None)
    rt_p.add_argument(
        "--vad", type=int, default=2, choices=[0, 1, 2, 3],
        help="VAD aggressiveness (0=least, 3=most, default=2)",
    )

    # ── shared options ─────────────────────────────────────────────────
    for p in [file_p, rt_p]:
        p.add_argument("--model", type=str, default="tarteel-ai/whisper-base-ar-quran")
        p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "mps"])
        p.add_argument("--threshold", type=float, default=40.0)
        p.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        level=level,
        stream=sys.stderr,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Lazy import so help/--help is fast
    from pipeline.runner import TarteelPipeline

    pipeline = TarteelPipeline(
        model_id=args.model,
        device=args.device,
        similarity_threshold=args.threshold,
    )

    if args.mode == "file":
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"[ERROR] File not found: {audio_path}", file=sys.stderr)
            sys.exit(1)

        result = pipeline.run(
            audio_path,
            surah=args.surah,
            ayah=args.ayah,
            return_timestamps=args.timestamps,
        )
        pipeline.print_result(result)

    elif args.mode == "realtime":
        from realtime.stream import RealtimeReciter

        reciter = RealtimeReciter(
            pipeline=pipeline,
            surah=args.surah,
            ayah=args.ayah,
            vad_aggressiveness=args.vad,
            on_result=pipeline.print_result,
        )
        reciter.start()


if __name__ == "__main__":
    main()