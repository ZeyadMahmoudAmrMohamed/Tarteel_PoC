"""
core/audio.py
-------------
Loads and preprocesses audio using torchaudio (no librosa/numba dependency).
Converts any format to 16kHz mono float32 — what Whisper expects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

TARGET_SR = 16_000


class AudioLoader:
    def __init__(self, target_sr: int = TARGET_SR) -> None:
        self.target_sr = target_sr

    def load(self, path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file → mono float32 numpy array at 16kHz.
        Uses torchaudio — no librosa/numba required.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info(f"[AudioLoader] Loading '{path.name}' …")

        # torchaudio.load returns (waveform_tensor, sample_rate)
        # waveform shape: (channels, samples)
        waveform, sr = torchaudio.load(str(path))

        # Stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # (1, N) → (N,) numpy float32
        audio = waveform.squeeze(0).numpy().astype(np.float32)
        audio = self._normalise(audio)

        duration = len(audio) / self.target_sr
        logger.info(f"[AudioLoader] {duration:.1f}s @ {self.target_sr} Hz")
        return audio, self.target_sr

    def get_duration(self, audio: np.ndarray) -> float:
        return len(audio) / self.target_sr

    def split_into_chunks(
        self, audio: np.ndarray, chunk_s: float = 30.0, overlap_s: float = 0.5
    ) -> list[np.ndarray]:
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
        return chunks

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.target_sr:
            return audio
        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampler = T.Resample(orig_freq=orig_sr, new_freq=self.target_sr)
        return resampler(waveform).squeeze(0).numpy().astype(np.float32)

    @staticmethod
    def _normalise(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (target_peak / peak)
        return audio