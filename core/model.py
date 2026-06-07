"""
core/model.py
-------------
Loads the Tarteel Whisper model from HuggingFace and exposes a clean
transcribe() interface that works on numpy arrays or file paths.

The model: tarteel-ai/whisper-base-ar-quran
  - Fine-tuned from openai/whisper-base
  - Trained on Quranic Arabic recitation
  - WER ~5.75 on evaluation set
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)

logger = logging.getLogger(__name__)

MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
SAMPLE_RATE = 16_000  # Whisper always expects 16 kHz


class TarteelModel:
    """
    Wrapper around the Tarteel Whisper ASR model.

    Usage
    -----
    >>> model = TarteelModel()
    >>> result = model.transcribe("path/to/audio.wav")
    >>> print(result["text"])
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model_id  : HuggingFace model repo id.
        device    : 'cuda', 'cpu', or None (auto-detect).
        cache_dir : Where to cache the downloaded model weights.
        """
        self.model_id = model_id
        self.device = device or self._auto_device()
        self.cache_dir = cache_dir

        logger.info(f"[TarteelModel] Using device: {self.device}")
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: str = "ar",
        return_timestamps: bool = False,
        chunk_length_s: int = 30,
    ) -> dict:
        """
        Transcribe audio to Arabic text.

        Parameters
        ----------
        audio            : File path (str/Path) OR numpy float32 array at 16kHz.
        language         : ISO language code. Default 'ar' (Arabic).
        return_timestamps: If True, returns word/chunk-level timestamps.
        chunk_length_s   : For long audio, split into chunks of this size (seconds).

        Returns
        -------
        dict with keys:
          - "text"   : str  — full transcription (Arabic, no diacritics from Whisper)
          - "chunks" : list — present only when return_timestamps=True
          - "duration_s" : float — processing time
        """
        t0 = time.perf_counter()

        # Build kwargs for the pipeline
        generate_kwargs = {"language": language, "task": "transcribe"}

        pipeline_kwargs: dict = {
            "chunk_length_s": chunk_length_s,
            "generate_kwargs": generate_kwargs,
        }
        if return_timestamps:
            pipeline_kwargs["return_timestamps"] = "word"

        # Accept numpy arrays directly
        if isinstance(audio, np.ndarray):
            inputs = {"array": audio, "sampling_rate": SAMPLE_RATE}
        else:
            inputs = str(audio)

        raw = self._pipe(inputs, **pipeline_kwargs)

        elapsed = time.perf_counter() - t0

        result: dict = {
            "text": raw["text"].strip(),
            "duration_s": round(elapsed, 3),
        }
        if return_timestamps and "chunks" in raw:
            result["chunks"] = raw["chunks"]

        logger.info(
            f"[TarteelModel] Transcribed in {elapsed:.2f}s → '{result['text'][:60]}…'"
        )
        return result

    def transcribe_segment(self, audio_array: np.ndarray, language: str = "ar") -> str:
        """
        Lightweight call for short (<30s) audio segments.
        Returns plain text string — used by the real-time streaming module.
        """
        result = self.transcribe(audio_array, language=language)
        return result["text"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        logger.info(f"[TarteelModel] Loading '{self.model_id}' …")
        t0 = time.perf_counter()

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=self.cache_dir
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
        self.model.to(self.device)

        # Use the high-level pipeline for convenient chunked inference
        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

        elapsed = time.perf_counter() - t0
        logger.info(f"[TarteelModel] Model loaded in {elapsed:.1f}s")

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"