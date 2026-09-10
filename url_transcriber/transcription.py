"""Local CPU transcription through faster-whisper."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from av.error import FFmpegError
from faster_whisper import WhisperModel

from url_transcriber.errors import TranscriptionError
from url_transcriber.models import TranscriptResult, TranscriptSegment


DEFAULT_MODEL = "small"
_EXPECTED_TRANSCRIPTION_ERRORS = (OSError, RuntimeError, ValueError, FFmpegError)


def transcribe_audio(
    audio_path: Path,
    model_name: str = DEFAULT_MODEL,
) -> TranscriptResult:
    """Transcribe an existing audio file locally without taking ownership of it."""
    source_path = _validate_audio_path(audio_path)

    try:
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
        )
    except _EXPECTED_TRANSCRIPTION_ERRORS as error:
        raise TranscriptionError("Could not load the local Whisper model.") from error

    try:
        whisper_segments, transcription_info = model.transcribe(str(source_path))
    except _EXPECTED_TRANSCRIPTION_ERRORS as error:
        raise TranscriptionError("Could not transcribe the audio file.") from error

    segments = _convert_segments(whisper_segments)
    if not segments:
        raise TranscriptionError("Local Whisper produced no usable transcript text.")

    language = getattr(transcription_info, "language", None)
    if not isinstance(language, str):
        language = ""
    else:
        language = language.strip()

    return TranscriptResult(
        segments=segments,
        language=language,
        source="local Whisper",
        whisper_model=model_name,
    )


def _validate_audio_path(audio_path: Path) -> Path:
    try:
        source_path = audio_path.resolve(strict=True)
    except OSError as error:
        raise TranscriptionError("The audio file does not exist.") from error

    if not source_path.is_file():
        raise TranscriptionError("The audio path is not a regular file.")
    return source_path


def _convert_segments(whisper_segments: Iterable[Any]) -> list[TranscriptSegment]:
    converted: list[TranscriptSegment] = []
    try:
        for segment in whisper_segments:
            text = segment.text.strip()
            if text:
                converted.append(
                    TranscriptSegment(
                        start_seconds=float(segment.start),
                        text=text,
                    )
                )
    except _EXPECTED_TRANSCRIPTION_ERRORS as error:
        raise TranscriptionError("Local Whisper transcription failed.") from error
    return converted
