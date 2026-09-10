"""Metadata-only video inspection through the yt-dlp Python API."""

import math
from collections.abc import Mapping
from typing import Any

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from url_transcriber.errors import ExtractionError
from url_transcriber.models import SubtitleTracks, VideoMetadata


_YDL_OPTIONS: dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "skip_download": True,
    "writesubtitles": False,
    "writeautomaticsub": False,
    "writethumbnail": False,
}


def extract_metadata(url: str) -> VideoMetadata:
    """Extract normalized metadata for one video without downloading files."""
    try:
        with YoutubeDL(_YDL_OPTIONS.copy()) as downloader:
            info = downloader.extract_info(url, download=False)
    except DownloadError as error:
        raise ExtractionError("Could not extract video metadata.") from error

    if not isinstance(info, Mapping):
        raise ExtractionError("yt-dlp returned no usable video metadata.")

    if info.get("_type") in {"playlist", "multi_video"} or "entries" in info:
        raise ExtractionError("Only single-video URLs are supported.")

    title = _nonempty_string(info.get("title"))
    if title is None:
        raise ExtractionError("Extracted metadata did not include a video title.")

    return VideoMetadata(
        title=title,
        webpage_url=_nonempty_string(info.get("webpage_url")) or url,
        uploader=_nonempty_string(info.get("channel"))
        or _nonempty_string(info.get("uploader")),
        duration_seconds=_normalize_duration(info.get("duration")),
        manual_subtitles=_normalize_subtitle_tracks(info.get("subtitles")),
        automatic_captions=_normalize_subtitle_tracks(
            info.get("automatic_captions")
        ),
    )


def _nonempty_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_duration(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


def _normalize_subtitle_tracks(value: object) -> SubtitleTracks:
    if not isinstance(value, Mapping):
        return {}

    tracks: SubtitleTracks = {}
    for language, entries in value.items():
        if not isinstance(language, str) or not isinstance(entries, list):
            continue

        formats: list[str] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            extension = _nonempty_string(entry.get("ext"))
            if extension is not None and extension not in formats:
                formats.append(extension)

        tracks[language] = tuple(formats)

    return tracks
