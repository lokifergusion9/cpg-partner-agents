"""Local speech for the voice demo (ADR-021, HLD section 6.11 mode ``live_voice``).

Speech-to-text with faster-whisper and text-to-speech with Piper, both on this machine. Audio is
processed in memory only: nothing here writes a file. The two packages are an optional extra
(``uv sync --extra speech``) and are imported only when ``SPEECH_ENABLED`` is on, so the default
install and the tests never need them. ``FakeSpeech`` in ``fakes.py`` stands in for tests.

A transcript is partner text like any other (CLAUDE.md rule 11): it goes to the call sub-graph,
which wraps it for the classifier.
"""

import asyncio
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from partner_agents.config import Settings

Audio = tuple[int, npt.NDArray[Any]]  # (sample rate in Hz, samples), as Gradio hands it over

WHISPER_RATE = 16_000
MAX_SECONDS = 30  # longer recordings are cut: a turn is a sentence or two
MIN_SAMPLES = WHISPER_RATE // 10  # under 0.1 s there is nothing to transcribe
MAX_SPEECH_CHARS = 600


class SpeechUnavailable(RuntimeError):
    """Speech is switched on but a package, a model or a voice file is missing."""


class SpeechBackend(Protocol):
    def transcribe(self, samples: npt.NDArray[np.float32]) -> str:
        """Mono float32 samples at 16 kHz in, text out."""
        ...

    def synthesize(self, text: str) -> Audio:
        """Text in, mono int16 audio out."""
        ...


def to_mono_16k(audio: Audio) -> npt.NDArray[np.float32]:
    """What Whisper reads: mono float32 in [-1, 1] at 16 kHz, at most ``MAX_SECONDS`` long."""
    rate, data = audio
    if rate <= 0:
        raise ValueError("the sample rate must be positive")
    samples = np.asarray(data)
    if samples.dtype.kind in "iu":  # scale to [-1, 1] before any averaging
        half = (float(np.iinfo(samples.dtype).max) + 1.0) / (
            2.0 if samples.dtype.kind == "u" else 1
        )
        centre = half if samples.dtype.kind == "u" else 0.0
        floats = (samples.astype(np.float64) - centre) / half
    else:
        floats = samples.astype(np.float64)
    if floats.ndim == 2:  # (samples, channels)
        floats = floats.mean(axis=1)
    mono: npt.NDArray[np.float32] = np.clip(floats, -1.0, 1.0).astype(np.float32)
    if rate != WHISPER_RATE and mono.size:
        target = max(1, round(mono.size * WHISPER_RATE / rate))
        positions = np.linspace(0, mono.size - 1, target)
        mono = np.interp(positions, np.arange(mono.size), mono).astype(np.float32)
    return mono[: MAX_SECONDS * WHISPER_RATE]


async def transcribe(backend: SpeechBackend, audio: Audio | None) -> str:
    """The text of one recording, or ``""`` for none or a recording that is too short."""
    if audio is None:
        return ""
    samples = to_mono_16k(audio)
    if samples.size < MIN_SAMPLES:
        return ""
    text = await asyncio.to_thread(backend.transcribe, samples)
    return text.strip()


async def speak(backend: SpeechBackend, text: str) -> Audio | None:
    """The agent's line as audio, or ``None`` for an empty line."""
    line = text.strip()[:MAX_SPEECH_CHARS]
    if not line:
        return None
    return await asyncio.to_thread(backend.synthesize, line)


class LocalSpeech:
    """faster-whisper (CPU, int8) and Piper. Loading is slow, so build it once."""

    def __init__(self, whisper_model: str, piper_voice: str | None) -> None:
        if not piper_voice:
            raise SpeechUnavailable("Set PIPER_VOICE to the path of a Piper .onnx voice file.")
        try:
            from faster_whisper import WhisperModel
            from piper import PiperVoice
        except ImportError as exc:
            raise SpeechUnavailable(
                "Install the speech extra first: uv sync --extra speech"
            ) from exc
        try:
            self._stt = WhisperModel(whisper_model, device="cpu", compute_type="int8")
            self._voice = PiperVoice.load(piper_voice)
        except Exception as exc:  # a missing file, a failed download, a bad model name
            raise SpeechUnavailable(f"Could not load the speech models: {exc}") from exc

    def transcribe(self, samples: npt.NDArray[np.float32]) -> str:
        segments, _ = self._stt.transcribe(samples, language="en", beam_size=1)
        return " ".join(segment.text.strip() for segment in segments)

    def synthesize(self, text: str) -> Audio:
        chunks = list(self._voice.synthesize(text))
        if not chunks:
            raise SpeechUnavailable("Piper returned no audio.")
        samples = np.concatenate([chunk.audio_int16_array for chunk in chunks])
        return int(chunks[0].sample_rate), samples


def build_speech(settings: Settings) -> SpeechBackend | None:
    """The backend when ``SPEECH_ENABLED`` is on, else ``None``. Raises ``SpeechUnavailable``."""
    if not settings.speech_enabled:
        return None
    return LocalSpeech(settings.whisper_model, settings.piper_voice)
