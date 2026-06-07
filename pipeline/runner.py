"""
pipeline/runner.py
------------------
Orchestrates the complete Tarteel pipeline:
  Audio file → AudioLoader → TarteelModel → QuranAligner → Result
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from core.model import TarteelModel
from core.audio import AudioLoader
from core.alignment import QuranAligner, AlignmentResult
from quran.loader import QuranLoader

logger = logging.getLogger(__name__)


def _print(msg: str) -> None:
    """Plain print — works everywhere (Kaggle, Colab, terminal, scripts)."""
    print(msg, flush=True)


class TarteelPipeline:
    """
    Full offline pipeline: audio file → transcription → Quran alignment.

    Parameters
    ----------
    model_id             : HuggingFace model repo id.
    device               : 'cuda' | 'cpu' | None (auto).
    similarity_threshold : Min fuzzy score to accept a verse match (0–100).
    quran_cache_path     : Path to quran.json cache.
    """

    def __init__(
        self,
        model_id: str = "tarteel-ai/whisper-base-ar-quran",
        device: Optional[str] = None,
        similarity_threshold: float = 40.0,
        quran_cache_path: Optional[str] = None,
    ) -> None:
        _print("=" * 60)
        _print("  Tarteel PoC — Initialising")
        _print("=" * 60)

        _print("[1/3] Loading Quran data...")
        self.quran_loader = QuranLoader(cache_path=quran_cache_path)
        quran_data = self.quran_loader.load()
        _print(f"      Done. {sum(len(v) for v in quran_data.values())} verses loaded.")

        _print("[2/3] Loading Whisper model...")
        self.model = TarteelModel(model_id=model_id, device=device)
        _print(f"      Done. Device: {self.model.device}")

        _print("[3/3] Setting up aligner...")
        self.audio_loader = AudioLoader()
        self.aligner = QuranAligner(
            quran_data=quran_data,
            similarity_threshold=similarity_threshold,
        )
        _print("      Done.")
        _print("=" * 60)
        _print("  Pipeline ready.\n")

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
        surah             : Constrain verse search to this surah (optional).
        ayah              : Align against this exact verse (needs surah).
        return_timestamps : Include word-level timestamps from Whisper.

        Returns
        -------
        AlignmentResult
        """
        t_start = time.perf_counter()
        audio_path = Path(audio_path)

        _print(f"\n--- Processing: {audio_path.name} ---")

        # ── Step 1: Load audio ─────────────────────────────────────────
        _print("  [1/3] Loading audio...")
        audio, sr = self.audio_loader.load(audio_path)
        duration = self.audio_loader.get_duration(audio)
        _print(f"        {duration:.1f}s @ {sr} Hz")

        # ── Step 2: Transcribe ─────────────────────────────────────────
        _print("  [2/3] Transcribing with Whisper...")
        transcription = self.model.transcribe(
            audio,
            return_timestamps=return_timestamps,
        )
        _print(f"        Done in {transcription['duration_s']:.1f}s")
        _print(f"        Text: {transcription['text']}")

        # ── Step 3: Align ──────────────────────────────────────────────
        _print("  [3/3] Aligning with Quran...")
        result = self.aligner.align(
            transcription["text"],
            surah=surah,
            ayah=ayah,
        )

        total = time.perf_counter() - t_start
        _print(f"        Matched: Surah {result.surah}, Ayah {result.ayah} "
               f"(score: {result.match_score:.0f}/100)")
        _print(f"        Total time: {total:.2f}s\n")

        return result

    def run_on_array(
        self,
        audio_array,
        surah: Optional[int] = None,
        ayah: Optional[int] = None,
    ) -> AlignmentResult:
        """Run pipeline on a numpy array — used by real-time module."""
        text = self.model.transcribe_segment(audio_array)
        return self.aligner.align(text, surah=surah, ayah=ayah)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_result(self, result: AlignmentResult) -> None:
        """Print the alignment result."""
        w = 60
        _print("\n" + "=" * w)
        _print("  ALIGNMENT RESULT")
        _print("=" * w)
        _print(f"  Surah {result.surah}, Ayah {result.ayah}")
        _print("-" * w)
        _print("  Ground truth (with tashkeel):")
        _print(f"    {result.reference_with_diacritics}")
        _print("")
        _print("  Your recitation (transcribed):")
        _print(f"    {result.hypothesis}")
        _print("-" * w)

        wer_pct = result.wer * 100
        cer_pct = result.cer * 100
        accuracy = (1 - result.wer) * 100

        _print(f"  Match Score  : {result.match_score:.0f} / 100")
        _print(f"  Word Accuracy: {accuracy:.1f}%")
        _print(f"  WER          : {wer_pct:.1f}%")
        _print(f"  CER          : {cer_pct:.1f}%")
        _print(f"  Correct Words: {len(result.correctly_recited_words)} / "
               f"{len(result.word_alignments)}")
        _print("-" * w)

        # Word-level alignment legend
        _print("  Word alignment:  [correct]  (substitution)  <deletion>  {insertion}")
        _print("")
        parts = []
        for wa in result.word_alignments:
            if wa.status == "correct":
                parts.append(f"[{wa.ref_word_with_diac}]")
            elif wa.status == "substitution":
                parts.append(f"({wa.hyp_word}->{wa.ref_word_with_diac})")
            elif wa.status == "deletion":
                parts.append(f"<{wa.ref_word_with_diac}>")
            else:  # insertion
                parts.append(f"{{{wa.hyp_word}}}")
        _print("  " + "  ".join(parts))
        _print("=" * w + "\n")