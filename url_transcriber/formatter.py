"""Pure formatting helpers for transcript output."""

import math

from url_transcriber.models import TranscriptResult, TranscriptSource, VideoMetadata


_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}

_SOURCE_LABELS = {
    "manual subtitles": "YouTube manual subtitles",
    "automatic captions": "YouTube auto captions",
    "local Whisper": "Local Whisper",
}


def format_timestamp(seconds: int | float) -> str:
    """Format non-negative seconds as ``MM:SS`` or ``HH:MM:SS``.

    Fractional seconds are truncated because v0.1 does not display
    milliseconds.
    """
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be a finite, non-negative number")

    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def format_duration(duration_seconds: int | float | None) -> str:
    """Format a known duration, or return a readable missing-value label."""
    if duration_seconds is None:
        return "Unknown"
    return format_timestamp(duration_seconds)


def format_transcript_source(source: TranscriptSource) -> str:
    """Return the stable user-facing label for a transcript source."""
    return _SOURCE_LABELS[source]


def sanitize_filename(title: str) -> str:
    """Return a deterministic, Windows-safe filename stem for ``title``.

    Unicode letters and numbers are retained. Whitespace, punctuation, symbols,
    control characters, and Windows-invalid filename characters become a
    single hyphen separator.
    """
    parts: list[str] = []
    separator_pending = False

    for character in title.strip().lower():
        if character.isalnum():
            if separator_pending and parts:
                parts.append("-")
            parts.append(character)
            separator_pending = False
        else:
            separator_pending = True

    stem = "".join(parts) or "transcript"
    if stem in _WINDOWS_RESERVED_NAMES:
        stem = f"{stem}-transcript"
    return stem


def format_markdown(metadata: VideoMetadata, transcript: TranscriptResult) -> str:
    """Format video metadata and transcript segments as Markdown."""
    title = metadata.title.strip() or "Untitled video"
    webpage_url = metadata.webpage_url.strip() or "Unknown"
    channel = metadata.uploader.strip() if metadata.uploader else "Unknown"
    language = transcript.language.strip() if transcript.language else "Unknown"
    source_label = format_transcript_source(transcript.source)

    sections = [
        f"# {title}",
        f"Source: {webpage_url}",
        f"Channel: {channel}",
        f"Duration: {format_duration(metadata.duration_seconds)}",
        f"Language: {language}",
        f"Transcript source: {source_label}",
    ]

    if transcript.source == "local Whisper" and transcript.whisper_model:
        sections.append(f"Whisper model: {transcript.whisper_model}")

    sections.append("## Transcript")
    sections.extend(
        f"[{format_timestamp(segment.start_seconds)}]\n{segment.text}"
        for segment in transcript.segments
    )
    return "\n\n".join(sections) + "\n"
