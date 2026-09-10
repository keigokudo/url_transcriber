"""Subtitle selection, in-memory retrieval, and conservative VTT parsing."""

import html
import re
from dataclasses import dataclass
from typing import Literal

from yt_dlp import YoutubeDL
from yt_dlp.networking import Request
from yt_dlp.networking.exceptions import RequestError

from url_transcriber.errors import SubtitleError
from url_transcriber.models import (
    SubtitleTrack,
    TranscriptResult,
    TranscriptSegment,
    VideoMetadata,
)


SubtitleSource = Literal["manual subtitles", "automatic captions"]


@dataclass(frozen=True)
class SubtitleSelection:
    """The one supported subtitle resource selected for processing."""

    language: str
    source: SubtitleSource
    track: SubtitleTrack


_LEGACY_LANGUAGE_PRIORITY = ("en", "ja")
_SUPPORTED_EXTENSION = "vtt"
_TIMING_LINE = re.compile(
    r"^\s*(?P<start>(?:\d{2,}:)?\d{2}:\d{2}(?:[.,]\d{1,3})?)"
    r"\s+-->\s+"
    r"(?P<end>(?:\d{2,}:)?\d{2}:\d{2}(?:[.,]\d{1,3})?)"
    r"(?:\s+.*)?$"
)
_VTT_TAG = re.compile(r"<[^>]*>")


def select_subtitle(metadata: VideoMetadata) -> SubtitleSelection | None:
    """Select one retrievable original-language VTT, if available."""
    sources = (
        ("manual subtitles", metadata.manual_subtitles),
        ("automatic captions", metadata.automatic_captions),
    )
    original_family = _language_family(metadata.original_language)
    language_families = (
        (original_family,) if original_family else _LEGACY_LANGUAGE_PRIORITY
    )

    for source, tracks_by_language in sources:
        for language_family in language_families:
            for language in _matching_languages(
                tracks_by_language,
                language_family,
                preferred=metadata.original_language,
            ):
                for track in tracks_by_language[language]:
                    if (
                        track.extension.casefold() == _SUPPORTED_EXTENSION
                        and (track.content is not None or track.url is not None)
                    ):
                        return SubtitleSelection(language, source, track)
    return None


def fetch_subtitle(selection: SubtitleSelection) -> str:
    """Retrieve the selected subtitle into memory, honoring its HTTP headers."""
    if selection.track.content is not None:
        return selection.track.content
    if selection.track.url is None:
        raise SubtitleError("The selected subtitle has no retrievable content.")

    try:
        request = Request(
            selection.track.url,
            headers=selection.track.http_headers,
        )
        with YoutubeDL({"quiet": True, "no_warnings": True}) as downloader:
            with downloader.urlopen(request) as response:
                content = response.read()
    except (RequestError, OSError) as error:
        raise SubtitleError("Could not retrieve the selected subtitle.") from error

    if isinstance(content, str):
        return content
    if not isinstance(content, bytes):
        raise SubtitleError("Subtitle retrieval returned unusable content.")

    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise SubtitleError("The selected subtitle is not valid UTF-8 text.") from error


def parse_vtt(content: str) -> list[TranscriptSegment]:
    """Parse useful cues from a small, YouTube-oriented WebVTT subset."""
    normalized = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    segments: list[TranscriptSegment] = []

    for block in re.split(r"\n\s*\n", normalized):
        lines = block.splitlines()
        if not lines:
            continue

        first_line = lines[0].strip()
        if first_line.startswith("WEBVTT") or first_line.startswith("NOTE"):
            continue
        if first_line in {"STYLE", "REGION"}:
            continue

        timing_index = next(
            (index for index, line in enumerate(lines) if "-->" in line),
            None,
        )
        if timing_index is None:
            continue

        match = _TIMING_LINE.fullmatch(lines[timing_index])
        if match is None:
            error = ValueError(lines[timing_index].strip())
            raise SubtitleError("Could not parse a subtitle cue timestamp.") from error

        try:
            start_seconds = _parse_vtt_timestamp(match.group("start"))
            _parse_vtt_timestamp(match.group("end"))
        except ValueError as error:
            raise SubtitleError("Could not parse a subtitle cue timestamp.") from error

        text = _clean_cue_text("\n".join(lines[timing_index + 1 :]))
        if text:
            segments.append(TranscriptSegment(start_seconds, text))

    if not segments:
        raise SubtitleError("Subtitle content contained no usable VTT cues.")
    return segments


def clean_segments(
    segments: list[TranscriptSegment],
    *,
    remove_automatic_overlap: bool,
) -> list[TranscriptSegment]:
    """Remove exact adjacent duplicates and clear rolling-caption overlap."""
    cleaned: list[TranscriptSegment] = []
    previous_cue_text: str | None = None

    for segment in segments:
        text = segment.text
        if text == previous_cue_text:
            continue

        retained_text = text
        if remove_automatic_overlap and previous_cue_text is not None:
            retained_text = _remove_clear_word_overlap(previous_cue_text, text)

        if retained_text:
            cleaned.append(TranscriptSegment(segment.start_seconds, retained_text))
        previous_cue_text = text

    return cleaned


def process_subtitles(metadata: VideoMetadata) -> TranscriptResult | None:
    """Select and process a subtitle, or return ``None`` when none is usable."""
    selection = select_subtitle(metadata)
    if selection is None:
        return None

    segments = parse_vtt(fetch_subtitle(selection))
    segments = clean_segments(
        segments,
        remove_automatic_overlap=selection.source == "automatic captions",
    )
    if not segments:
        raise SubtitleError("Subtitle content contained no usable transcript text.")

    return TranscriptResult(
        segments=segments,
        language=selection.language,
        source=selection.source,
        whisper_model=None,
    )


def _matching_languages(
    tracks_by_language: dict[str, tuple[SubtitleTrack, ...]],
    family: str,
    *,
    preferred: str | None = None,
) -> list[str]:
    normalized_preferred = _normalize_language_identifier(preferred)
    exact_preferred = [
        language
        for language in tracks_by_language
        if _normalize_language_identifier(language) == normalized_preferred
    ]
    exact_family = [
        language
        for language in tracks_by_language
        if _normalize_language_identifier(language) == family
        and language not in exact_preferred
    ]
    variants = sorted(
        (
            language
            for language in tracks_by_language
            if _language_family(language) == family
            and language not in exact_preferred
            and language not in exact_family
        ),
        key=str.casefold,
    )
    return exact_preferred + exact_family + variants


def _normalize_language_identifier(language: str | None) -> str | None:
    if not isinstance(language, str):
        return None
    normalized = language.strip().casefold().replace("_", "-")
    return normalized or None


def _language_family(language: str | None) -> str | None:
    normalized = _normalize_language_identifier(language)
    if normalized is None:
        return None
    family = normalized.split("-", 1)[0]
    if not re.fullmatch(r"[a-z]{2,3}", family):
        return None
    if family in {"und", "zxx", "mul", "mis"}:
        return None
    return family


def _parse_vtt_timestamp(value: str) -> float:
    time_part, separator, fraction = value.replace(",", ".").partition(".")
    components = [int(component) for component in time_part.split(":")]
    if len(components) == 2:
        hours = 0
        minutes, seconds = components
    elif len(components) == 3:
        hours, minutes, seconds = components
    else:
        raise ValueError(value)

    if minutes >= 60 or seconds >= 60:
        raise ValueError(value)

    milliseconds = int(fraction.ljust(3, "0")) if separator else 0
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def _clean_cue_text(value: str) -> str:
    without_markup = _VTT_TAG.sub("", value)
    return re.sub(r"\s+", " ", html.unescape(without_markup)).strip()


def _remove_clear_word_overlap(previous: str, current: str) -> str:
    previous_words = previous.split()
    current_words = current.split()
    maximum = min(len(previous_words), len(current_words))

    for overlap_size in range(maximum, 1, -1):
        if previous_words[-overlap_size:] == current_words[:overlap_size]:
            return " ".join(current_words[overlap_size:])
    return current
