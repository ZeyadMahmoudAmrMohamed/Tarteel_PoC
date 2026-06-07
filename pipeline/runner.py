"""
pipeline/runner.py
------------------
Orchestrates the complete Tarteel pipeline:

  Audio file  →  AudioLoader  →  TarteelModel  →  QuranAligner  →  Result

This is the main entry point for the offline (file-based) workflow.
For real-time mic streaming, see realtime/stream.py.

Usage
-----
    from pipeline import TarteelPipeline

    pipe = TarteelPipeline()
    result = pipe.run("my_recitation.wav", surah=1, ayah=1)
    pipe.print_result(result)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from core.model import TarteelModel
from core.audio import AudioLoader
from core.alignment import QuranAligner, AlignmentResult
from quran.loader import QuranLoader

logger = logging.getLogger(__name__)
console = Console()


class TarteelPipeline:
    """
    Full offline pipeline: audio file → transcription → Quran alignment.

    Parameters
    ----------
    model_id          : HuggingFace model repo id.
    device            : 'cuda' | 'cpu' | None (auto).
    similarity_threshold : Min fuzzy score to accept a verse match (0–100).
    quran_cache_path  : Path to quran.json cache.
    """

    def __init__(
        self,
        model_id: str = "tarteel-ai/whisper-base-ar-quran",
        device: Optional[str] = None,
        similarity_threshold: float = 40.0,
        quran_cache_path: Optional[str] = None,
    ) -> None:
        console.rule("[bold green]Tarteel PoC — Initialising")

        with console.status("[bold cyan]Loading Quran data…"):
            self.quran_loader = QuranLoader(cache_path=quran_cache_path)
            quran_data = self.quran_loader.load()

        with console.status("[bold cyan]Loading Whisper model…"):
            self.model = TarteelModel(model_id=model_id, device=device)

        self.audio_loader = AudioLoader()
        self.aligner = QuranAligner(
            quran_data=quran_data,
            similarity_threshold=similarity_threshold,
        )

        console.print("[bold green]✓ Pipeline ready\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        audio_path: str | Path,
        surah: Optional[int] = None,
        ayah: Optional[int] = None,
        return_timestamps: bool = False,
    ) -> AlignmentResult:
        """
        Run the full pipeline on an audio file.

        Parameters
        ----------
        audio_path        : Path to audio file.
        surah             : If given, constrain verse search to this surah.
        ayah              : If given WITH surah, align against this exact verse.
        return_timestamps : Include word-level timestamps from Whisper.

        Returns
        -------
        AlignmentResult
        """
        t_start = time.perf_counter()
        audio_path = Path(audio_path)

        console.rule(f"[bold blue]Processing: {audio_path.name}")

        # ── Step 1: Load audio ─────────────────────────────────────────
        with console.status("[cyan]Loading audio…"):
            audio, sr = self.audio_loader.load(audio_path)
            duration = self.audio_loader.get_duration(audio)
        console.print(f"  [green]✓[/] Audio loaded — {duration:.1f}s @ {sr} Hz")

        # ── Step 2: Transcribe ─────────────────────────────────────────
        with console.status("[cyan]Transcribing with Whisper…"):
            transcription = self.model.transcribe(
                audio,
                return_timestamps=return_timestamps,
            )
        console.print(
            f"  [green]✓[/] Transcription ({transcription['duration_s']:.1f}s): "
            f"[bold yellow]{transcription['text']}[/]"
        )

        # ── Step 3: Align ──────────────────────────────────────────────
        with console.status("[cyan]Aligning with Quran…"):
            result = self.aligner.align(
                transcription["text"],
                surah=surah,
                ayah=ayah,
            )

        total = time.perf_counter() - t_start
        console.print(
            f"  [green]✓[/] Matched: Surah {result.surah}, Ayah {result.ayah} "
            f"(score: {result.match_score:.0f}/100)\n"
            f"  [dim]Total time: {total:.2f}s[/]"
        )

        return result

    def run_on_array(
        self,
        audio_array,
        surah: Optional[int] = None,
        ayah: Optional[int] = None,
    ) -> AlignmentResult:
        """
        Run pipeline directly on a numpy audio array (used by real-time module).
        Skips file loading step.
        """
        text = self.model.transcribe_segment(audio_array)
        return self.aligner.align(text, surah=surah, ayah=ayah)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_result(self, result: AlignmentResult) -> None:
        """Pretty-print the alignment result to the terminal."""
        console.rule("[bold green]Alignment Result")

        # ── Verse info ─────────────────────────────────────────────────
        console.print(
            Panel(
                f"[bold]Surah {result.surah}, Ayah {result.ayah}[/bold]\n\n"
                f"[bold cyan]Ground truth (with tashkeel):[/bold cyan]\n"
                f"  {result.reference_with_diacritics}\n\n"
                f"[bold yellow]Your recitation (as transcribed):[/bold yellow]\n"
                f"  {result.hypothesis}",
                title="Verse Match",
                border_style="green",
            )
        )

        # ── Metrics ────────────────────────────────────────────────────
        wer_pct = result.wer * 100
        cer_pct = result.cer * 100
        accuracy = (1 - result.wer) * 100

        metrics = Table(box=box.ROUNDED, show_header=False)
        metrics.add_column("Metric", style="bold")
        metrics.add_column("Value")

        metrics.add_row(
            "Match Score",
            f"[green]{result.match_score:.0f} / 100[/]",
        )
        metrics.add_row(
            "Word Accuracy",
            self._colour_pct(accuracy),
        )
        metrics.add_row("WER", f"{wer_pct:.1f}%")
        metrics.add_row("CER", f"{cer_pct:.1f}%")
        metrics.add_row(
            "Correct Words",
            f"[green]{len(result.correctly_recited_words)}[/] / "
            f"{len(result.word_alignments)}",
        )
        console.print(metrics)

        # ── Word-level alignment ───────────────────────────────────────
        console.print("\n[bold]Word-level alignment:[/bold]")
        align_text = Text()
        for wa in result.word_alignments:
            if wa.status == "correct":
                align_text.append(
                    f"{wa.ref_word_with_diac} ", style="bold green"
                )
            elif wa.status == "substitution":
                align_text.append(
                    f"[{wa.hyp_word}→{wa.ref_word_with_diac}] ",
                    style="bold yellow",
                )
            elif wa.status == "deletion":
                align_text.append(
                    f"<{wa.ref_word_with_diac}> ", style="bold red"
                )
            else:  # insertion
                align_text.append(
                    f"({wa.hyp_word}) ", style="bold magenta"
                )

        console.print(
            Panel(
                align_text,
                title="Legend: [green]correct[/] [yellow]substitution[/] "
                "[red]deletion[/] [magenta]insertion[/]",
                border_style="dim",
            )
        )

    @staticmethod
    def _colour_pct(pct: float) -> str:
        if pct >= 90:
            return f"[bold green]{pct:.1f}%[/]"
        elif pct >= 70:
            return f"[bold yellow]{pct:.1f}%[/]"
        else:
            return f"[bold red]{pct:.1f}%[/]"