"""Tests for the initial project foundation."""

import importlib

import pytest

from url_transcriber.errors import (
    AudioDownloadError,
    ExtractionError,
    OutputError,
    SubtitleError,
    TranscriptionError,
    URLTranscriberError,
)
from url_transcriber.models import TranscriptResult, TranscriptSegment, VideoMetadata


def test_core_dataclasses_can_be_constructed() -> None:
    metadata = VideoMetadata(
        title="Example video",
        webpage_url="https://example.com/video",
        uploader="Example channel",
        duration_seconds=123.5,
        manual_subtitles={"en": ("vtt", "srt")},
        automatic_captions={"fr": ("vtt",)},
    )
    segment = TranscriptSegment(start_seconds=1.25, text="Hello world")

    assert metadata.title == "Example video"
    assert metadata.manual_subtitles["en"] == ("vtt", "srt")
    assert segment.start_seconds == 1.25


def test_transcript_result_supports_subtitle_source() -> None:
    result = TranscriptResult(
        segments=[TranscriptSegment(start_seconds=0.0, text="Caption text")],
        language="en",
        source="manual subtitles",
    )

    assert result.source == "manual subtitles"
    assert result.whisper_model is None


def test_transcript_result_supports_whisper_source() -> None:
    result = TranscriptResult(
        segments=[TranscriptSegment(start_seconds=0.0, text="Spoken text")],
        language="en",
        source="local Whisper",
        whisper_model="small",
    )

    assert result.source == "local Whisper"
    assert result.whisper_model == "small"


@pytest.mark.parametrize(
    "error_type",
    [
        ExtractionError,
        SubtitleError,
        AudioDownloadError,
        TranscriptionError,
        OutputError,
    ],
)
def test_project_errors_share_a_common_base(error_type: type[Exception]) -> None:
    assert issubclass(error_type, URLTranscriberError)


@pytest.mark.parametrize(
    "module_name",
    [
        "url_transcriber",
        "url_transcriber.cli",
        "url_transcriber.pipeline",
        "url_transcriber.models",
        "url_transcriber.errors",
        "url_transcriber.extractor",
        "url_transcriber.subtitles",
        "url_transcriber.audio",
        "url_transcriber.transcription",
        "url_transcriber.formatter",
        "url_transcriber.output",
        "main",
    ],
)
def test_project_modules_import(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
