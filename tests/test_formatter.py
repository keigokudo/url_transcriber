"""Tests for pure transcript formatting helpers."""

import pytest

from url_transcriber.formatter import (
    format_duration,
    format_markdown,
    format_timestamp,
    sanitize_filename,
)
from url_transcriber.models import TranscriptResult, TranscriptSegment, VideoMetadata


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00"),
        (18, "00:18"),
        (60, "01:00"),
        (64, "01:04"),
        (3599, "59:59"),
        (3600, "01:00:00"),
        (3661, "01:01:01"),
        (25 * 3600 + 2, "25:00:02"),
        (18.999, "00:18"),
    ],
)
def test_format_timestamp(seconds: int | float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


@pytest.mark.parametrize("seconds", [-1, float("inf"), float("nan")])
def test_format_timestamp_rejects_invalid_values(seconds: float) -> None:
    with pytest.raises(ValueError):
        format_timestamp(seconds)


@pytest.mark.parametrize(
    ("duration_seconds", "expected"),
    [
        (42 * 60 + 31, "42:31"),
        (3600 + 12 * 60 + 5, "01:12:05"),
        (None, "Unknown"),
    ],
)
def test_format_duration(
    duration_seconds: int | float | None,
    expected: str,
) -> None:
    assert format_duration(duration_seconds) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("The Future of AI: What Happens Next?", "the-future-of-ai-what-happens-next"),
        ('bad<name>:"with/invalid\\characters|?*', "bad-name-with-invalid-characters"),
        ("many     spaces", "many-spaces"),
        ("  padded title  ", "padded-title"),
        ("dots...and---separators___", "dots-and-separators"),
        ("日本語のタイトル", "日本語のタイトル"),
        ("< > : ? *", "transcript"),
        ("", "transcript"),
        ("CON", "con-transcript"),
        ("nul", "nul-transcript"),
        ("COM1", "com1-transcript"),
    ],
)
def test_sanitize_filename(title: str, expected: str) -> None:
    assert sanitize_filename(title) == expected


def test_format_markdown_for_subtitles() -> None:
    metadata = VideoMetadata(
        title="Example Video",
        webpage_url="https://www.youtube.com/watch?v=example",
        uploader="Example Channel",
        duration_seconds=42 * 60 + 31,
    )
    transcript = TranscriptResult(
        segments=[
            TranscriptSegment(0, "First segment text."),
            TranscriptSegment(18, "Second segment text."),
        ],
        language="English",
        source="automatic captions",
    )

    markdown = format_markdown(metadata, transcript)

    assert markdown.startswith("# Example Video\n\n")
    assert "Source: https://www.youtube.com/watch?v=example" in markdown
    assert "Channel: Example Channel" in markdown
    assert "Duration: 42:31" in markdown
    assert "Language: English" in markdown
    assert "Transcript source: YouTube auto captions" in markdown
    assert "Whisper model:" not in markdown
    assert "[00:00]\nFirst segment text." in markdown
    assert "[00:18]\nSecond segment text." in markdown
    assert markdown.index("First segment text.") < markdown.index("Second segment text.")


def test_format_markdown_for_whisper() -> None:
    metadata = VideoMetadata(
        title="Long Video",
        webpage_url="https://example.com/long-video",
        uploader="A Channel",
        duration_seconds=3661,
    )
    transcript = TranscriptResult(
        segments=[TranscriptSegment(3600, "An hour later.")],
        language="en",
        source="local Whisper",
        whisper_model="small",
    )

    markdown = format_markdown(metadata, transcript)

    assert "Duration: 01:01:01" in markdown
    assert "Transcript source: Local Whisper" in markdown
    assert "Whisper model: small" in markdown
    assert "[01:00:00]\nAn hour later." in markdown


def test_format_markdown_handles_missing_optional_metadata() -> None:
    metadata = VideoMetadata(
        title=" ",
        webpage_url="https://example.com/video",
        uploader=None,
        duration_seconds=None,
    )
    transcript = TranscriptResult(
        segments=[],
        language="",
        source="manual subtitles",
    )

    markdown = format_markdown(metadata, transcript)

    assert markdown.startswith("# Untitled video\n\n")
    assert "Channel: Unknown" in markdown
    assert "Duration: Unknown" in markdown
    assert "Language: Unknown" in markdown
    assert "Transcript source: YouTube manual subtitles" in markdown
    assert "Whisper model:" not in markdown
