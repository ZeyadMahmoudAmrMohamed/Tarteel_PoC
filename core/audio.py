"""
core/audio.py
-------------
Handles loading audio from files and preprocessing it to the 16 kHz
mono float32 format that Whisper expects.

Supported input formats: WAV, MP3, FLAC, OGG, M4A (anything librosa handles).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SR = 16_000  # Whisper's required sample rate


class AudioLoader:
    """
    Loads, resamples, and normalises audio for the Tarteel pipeline.

    Usage
    -----
    >>> loader = AudioLoader()
    >>> audio, sr = loader.load("recitation.wav")
    >>> print(audio.shape, sr)   # (N,) 16000
    """

    def __init__(self, target_sr: int = TARGET_SR) -> None:
        self.target_sr = target_sr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and return (audio_array, sample_rate).

        - Converts to mono.
        - Resamples to self.target_sr.
        - Normalises amplitude to [-1, 1].

        Returns
        -------
        (audio, sr) where audio is float32 numpy array.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info(f"[AudioLoader] Loading '{path.name}' …")

        audio, sr = librosa.load(str(path), sr=self.target_sr, mono=True)
        audio = self._normalise(audio)

        duration = len(audio) / self.target_sr
        logger.info(
            f"[AudioLoader] Loaded {duration:.1f}s of audio "
            f"({len(audio)} samples @ {self.target_sr} Hz)"
        )
        return audio.astype(np.float32), self.target_sr

    def load_raw(self, path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Load without resampling — returns original sample rate.
        Useful for inspection / debugging.
        """
        path = Path(path)
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono
        return audio, sr

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample an already-loaded array to target_sr."""
        if orig_sr == self.target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)

    def split_into_chunks(
        self, audio: np.ndarray, chunk_s: float = 30.0, overlap_s: float = 0.5
    ) -> list[np.ndarray]:
        """
        Split long audio into overlapping chunks for batch processing.

        Parameters
        ----------
        audio     : float32 mono array at self.target_sr.
        chunk_s   : Chunk length in seconds.
        overlap_s : Overlap between consecutive chunks (seconds).

        Returns
        -------
        List of numpy arrays.
        """
        chunk_len = int(chunk_s * self.target_sr)
        overlap_len = int(overlap_s * self.target_sr)
        step = chunk_len - overlap_len

        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_len, len(audio))
            chunks.append(audio[start:end])
            if end == len(audio):
                break
            start += step

        logger.info(
            f"[AudioLoader] Split into {len(chunks)} chunk(s) "
            f"({chunk_s}s each, {overlap_s}s overlap)"
        )
        return chunks

    def get_duration(self, audio: np.ndarray) -> float:
        """Return duration of audio array in seconds."""
        return len(audio) / self.target_sr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Peak-normalise to avoid clipping artefacts."""
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (target_peak / peak)
        return audio