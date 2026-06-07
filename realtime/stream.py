"""
realtime/stream.py
------------------
Real-time microphone streaming with Voice Activity Detection (VAD).

How it works
------------
  1. Capture audio from mic in small frames (e.g. 30ms each).
  2. Use WebRTC VAD to detect when speech starts and ends.
  3. Accumulate speech frames into a "segment" buffer.
  4. When silence is detected after speech, flush the buffer to the
     Tarteel model for transcription.
  5. Align the transcription result against the Quran in real time.

This module requires:
  - pyaudio  (pip install pyaudio)
  - webrtcvad (pip install webrtcvad)

On Linux you may need:  sudo apt-get install portaudio19-dev
On macOS:               brew install portaudio

Usage
-----
    from pipeline import TarteelPipeline
    from realtime.stream import RealtimeReciter

    pipe = TarteelPipeline()
    reciter = RealtimeReciter(pipeline=pipe, surah=1)
    reciter.start()   # blocks; press Ctrl+C to stop
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000        # Hz — Whisper requires 16 kHz
FRAME_DURATION_MS = 30      # ms per VAD frame (10, 20, or 30 are valid)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # samples per frame
BYTES_PER_SAMPLE = 2        # int16 = 2 bytes
FRAME_BYTES = FRAME_SIZE * BYTES_PER_SAMPLE

# How many consecutive silent frames before we flush the segment
SILENCE_THRESHOLD_FRAMES = 20   # 20 × 30ms = 600ms of silence


class RealtimeReciter:
    """
    Real-time Quran recitation recogniser using microphone input.

    Parameters
    ----------
    pipeline        : A fully initialised TarteelPipeline instance.
    surah           : Restrict alignment search to this surah (optional).
    ayah            : Restrict to exact verse (optional, needs surah).
    vad_aggressiveness : WebRTC VAD aggressiveness 0–3 (3 = most aggressive).
    on_result       : Optional callback(AlignmentResult) called on each segment.
    """

    def __init__(
        self,
        pipeline,                          # TarteelPipeline (avoid circular import)
        surah: Optional[int] = None,
        ayah: Optional[int] = None,
        vad_aggressiveness: int = 2,
        on_result: Optional[Callable] = None,
    ) -> None:
        self.pipeline = pipeline
        self.surah = surah
        self.ayah = ayah
        self.vad_aggressiveness = vad_aggressiveness
        self.on_result = on_result or self._default_result_handler

        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False

        self._check_dependencies()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the real-time capture + inference loop.
        Blocks until stop() is called or KeyboardInterrupt.
        """
        import pyaudio
        from rich.console import Console
        console = Console()

        self._running = True

        console.print(
            "[bold green]🎙  Real-time Quran recitation recognition started.[/]\n"
            "[dim]Speak into the microphone — press Ctrl+C to stop.[/]\n"
        )

        # Start the inference thread
        inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        inference_thread.start()

        # Main thread handles audio capture
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
        )

        try:
            self._capture_loop(stream)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping…[/]")
        finally:
            self._running = False
            stream.stop_stream()
            stream.close()
            pa.terminate()
            inference_thread.join(timeout=5)
            console.print("[green]Done.[/]")

    def stop(self) -> None:
        """Signal the streaming loop to stop."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal: capture loop (runs in main thread)
    # ------------------------------------------------------------------

    def _capture_loop(self, stream) -> None:
        """
        Read audio frames from the mic and push completed speech segments
        to the inference queue.
        """
        import webrtcvad
        vad = webrtcvad.Vad(self.vad_aggressiveness)

        speech_frames: list[bytes] = []
        silence_count = 0
        in_speech = False

        # Ring buffer: keep last N frames to catch the start of speech
        pre_roll = deque(maxlen=10)  # 10 × 30ms = 300ms pre-roll

        while self._running:
            frame_bytes = stream.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)

            if is_speech:
                if not in_speech:
                    # Speech just started — include pre-roll
                    in_speech = True
                    silence_count = 0
                    speech_frames.extend(list(pre_roll))
                    logger.debug("[Stream] Speech started")
                speech_frames.append(frame_bytes)
            else:
                if in_speech:
                    silence_count += 1
                    speech_frames.append(frame_bytes)  # include trailing silence
                    if silence_count >= SILENCE_THRESHOLD_FRAMES:
                        # Speech segment complete → flush
                        segment = b"".join(speech_frames)
                        self._audio_queue.put(segment)
                        speech_frames = []
                        silence_count = 0
                        in_speech = False
                        logger.debug("[Stream] Segment flushed to queue")
                else:
                    pre_roll.append(frame_bytes)

    # ------------------------------------------------------------------
    # Internal: inference loop (runs in background thread)
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        """
        Consumes audio segments from the queue, runs Whisper + alignment,
        and calls on_result callback.
        """
        while self._running or not self._audio_queue.empty():
            try:
                segment_bytes = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            audio_array = self._bytes_to_float32(segment_bytes)

            # Skip very short segments (< 0.3s) — likely noise
            if len(audio_array) < SAMPLE_RATE * 0.3:
                continue

            try:
                result = self.pipeline.run_on_array(
                    audio_array,
                    surah=self.surah,
                    ayah=self.ayah,
                )
                self.on_result(result)
            except Exception as e:
                logger.warning(f"[Stream] Inference error: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
        """Convert raw int16 PCM bytes to float32 [-1, 1] array."""
        audio = np.frombuffer(raw_bytes, dtype=np.int16)
        return audio.astype(np.float32) / 32768.0

    @staticmethod
    def _default_result_handler(result) -> None:
        """Default: pretty-print alignment result."""
        from rich.console import Console
        from rich.panel import Panel
        c = Console()
        c.print(
            Panel(
                f"[cyan]Surah {result.surah}, Ayah {result.ayah}[/]\n"
                f"[yellow]{result.hypothesis}[/]\n"
                f"WER: {result.wer*100:.1f}%  |  "
                f"Accuracy: {(1-result.wer)*100:.1f}%",
                title="[green]Segment Result",
                border_style="green",
            )
        )

    @staticmethod
    def _check_dependencies() -> None:
        """Check that pyaudio and webrtcvad are installed."""
        missing = []
        try:
            import pyaudio  # noqa: F401
        except ImportError:
            missing.append("pyaudio")
        try:
            import webrtcvad  # noqa: F401
        except ImportError:
            missing.append("webrtcvad")
        if missing:
            raise ImportError(
                f"Missing real-time dependencies: {', '.join(missing)}\n"
                f"Install with:  pip install {' '.join(missing)}\n"
                "On Linux also: sudo apt-get install portaudio19-dev"
            )