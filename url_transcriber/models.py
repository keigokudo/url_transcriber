"""Core data models shared by URL Transcriber components."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SubtitleTrack:
    """One available subtitle resource in a specific timed-text format."""

    extension: str
    url: str | None = None
    content: str | None = None
    http_headers: dict[str, str] = field(default_factory=dict)


SubtitleTracks = dict[str, tuple[SubtitleTrack, ...]]
TranscriptSource = Literal[
    "manual subtitles",
    "automatic captions",
    "local Whisper",
]


@dataclass
class VideoMetadata:
    """Normalized video information needed by the future pipeline.

    Subtitle mappings use language codes as keys and available subtitle
    resources for that language as values.
    """

    title: str
    webpage_url: str
    uploader: str | None
    duration_seconds: float | None
    manual_subtitles: SubtitleTracks = field(default_factory=dict)
    automatic_captions: SubtitleTracks = field(default_factory=dict)
    original_language: str | None = None


@dataclass
class TranscriptSegment:
    """A piece of transcript text and its start time in seconds."""

    start_seconds: float
    text: str


@dataclass
class TranscriptResult:
    """A completed transcript before it is formatted for output."""

    segments: list[TranscriptSegment]
    language: str
    source: TranscriptSource
    whisper_model: str | None = None
