"""
core/model.py
-------------
Loads the Tarteel Whisper model and transcribes audio.

We bypass the HuggingFace pipeline() entirely and call the model directly.
This is necessary because tarteel-ai/whisper-base-ar-quran was trained with
transformers 4.26, but the pipeline() in newer transformers always injects a
`language` kwarg into generate(), which triggers a ValueError when the saved
generation_config lacks the `lang_to_id` field introduced in 4.27+.

Calling model.generate() directly lets us control exactly what gets passed,
avoiding the compatibility issue completely.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

logger = logging.getLogger(__name__)

MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
SAMPLE_RATE = 16_000  # Whisper always expects 16 kHz
CHUNK_LENGTH_S = 30   # max seconds per chunk (Whisper's context window)
CHUNK_SAMPLES = CHUNK_LENGTH_S * SAMPLE_RATE


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
        return_timestamps: bool = False,
        chunk_length_s: int = 30,
        **_kwargs,  # absorb legacy language= / task= kwargs gracefully
    ) -> dict:
        """
        Transcribe audio to Arabic text.

        Parameters
        ----------
        audio            : File path (str/Path) OR numpy float32 array at 16kHz.
        return_timestamps: If True, include chunk-level timestamps in result.
        chunk_length_s   : Split long audio into chunks of this size (seconds).

        Returns
        -------
        dict with keys:
          "text"       — full Arabic transcription
          "chunks"     — list of {text, timestamp} dicts (if return_timestamps)
          "duration_s" — wall-clock inference time
        """
        t0 = time.perf_counter()

        # ── Load audio array ───────────────────────────────────────────
        if isinstance(audio, (str, Path)):
            import torchaudio, torchaudio.transforms as T
            waveform, sr = torchaudio.load(str(audio))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                waveform = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
            audio_array = waveform.squeeze(0).numpy().astype(np.float32)
        else:
            audio_array = np.asarray(audio, dtype=np.float32)

        # ── Split into 30-second chunks (Whisper's context window) ─────
        chunk_samples = int(chunk_length_s * SAMPLE_RATE)
        chunks = [
            audio_array[i : i + chunk_samples]
            for i in range(0, len(audio_array), chunk_samples)
        ]

        texts = []
        all_chunks = []

        for idx, chunk in enumerate(chunks):
            chunk_text, chunk_data = self._transcribe_chunk(
                chunk, return_timestamps=return_timestamps
            )
            texts.append(chunk_text)
            if return_timestamps:
                all_chunks.extend(chunk_data)

        elapsed = time.perf_counter() - t0
        full_text = " ".join(t.strip() for t in texts if t.strip())

        result: dict = {"text": full_text, "duration_s": round(elapsed, 3)}
        if return_timestamps:
            result["chunks"] = all_chunks

        logger.info(
            f"[TarteelModel] Transcribed {len(chunks)} chunk(s) in "
            f"{elapsed:.2f}s → '{full_text[:60]}…'"
        )
        return result

    def transcribe_segment(self, audio_array: np.ndarray, **_kwargs) -> str:
        """
        Lightweight call for short (<30s) audio segments.
        Returns plain text — used by the real-time streaming module.
        """
        return self.transcribe(audio_array)["text"]

    # ------------------------------------------------------------------
    # Internal: single-chunk transcription
    # ------------------------------------------------------------------

    def _transcribe_chunk(
        self, chunk: np.ndarray, return_timestamps: bool = False
    ) -> tuple[str, list]:
        """
        Run Whisper on a single ≤30s chunk using direct model.generate().
        No pipeline(), no language kwarg — fully compatible with 4.26 weights.
        """
        # Feature extraction — produces log-mel spectrogram
        inputs = self.processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            self.device, dtype=self.torch_dtype
        )

        # Decode — we pass ONLY input_features and max_new_tokens.
        # The forced_decoder_ids already encode <|ar|> + <|transcribe|> +
        # <|notimestamps|> so we never need to pass language= to generate().
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                max_new_tokens=447,
            )

        # Decode token ids → string
        text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        chunks_out = []
        if return_timestamps:
            chunks_out = [{"text": text, "timestamp": None}]

        return text, chunks_out

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        logger.info(f"[TarteelModel] Loading '{self.model_id}' …")
        t0 = time.perf_counter()

        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=self.cache_dir
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
        self.model.to(self.device)
        self.model.eval()

        # Build forced_decoder_ids manually from the tokenizer.
        # This is what the pipeline used to do internally — we do it once
        # here so generate() never needs a `language` argument at all.
        #
        # Token sequence: <|startoftranscript|> <|ar|> <|transcribe|> <|notimestamps|>
        # forced_decoder_ids = [(1, ar_token), (2, transcribe_token), (3, notimestamps_token)]
        tok = self.processor.tokenizer
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="arabic", task="transcribe"
        )

        elapsed = time.perf_counter() - t0
        logger.info(f"[TarteelModel] Model loaded in {elapsed:.1f}s")
        logger.info(f"[TarteelModel] forced_decoder_ids = {self.forced_decoder_ids}")

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"