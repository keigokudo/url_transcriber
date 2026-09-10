"""Network-free tests for yt-dlp metadata extraction."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from yt_dlp.utils import DownloadError

from url_transcriber import extractor
from url_transcriber.errors import ExtractionError
from url_transcriber.models import SubtitleTrack


def _mock_youtube_dl(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: Any = None,
    error: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    downloader = MagicMock()
    if error is None:
        downloader.extract_info.return_value = response
    else:
        downloader.extract_info.side_effect = error

    context_manager = MagicMock()
    context_manager.__enter__.return_value = downloader
    youtube_dl = MagicMock(return_value=context_manager)
    monkeypatch.setattr(extractor, "YoutubeDL", youtube_dl)
    return youtube_dl, downloader


def test_extract_metadata_normalizes_basic_fields_and_uses_no_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied_url = "https://example.com/watch?v=123&list=456"
    youtube_dl, downloader = _mock_youtube_dl(
        monkeypatch,
        response={
            "title": "Example Video",
            "webpage_url": "https://example.com/watch?v=123",
            "channel": "Stable Channel",
            "uploader": "Uploader Fallback",
            "duration": 125,
        },
    )

    metadata = extractor.extract_metadata(supplied_url)

    assert metadata.title == "Example Video"
    assert metadata.webpage_url == "https://example.com/watch?v=123"
    assert metadata.uploader == "Stable Channel"
    assert metadata.duration_seconds == 125.0
    assert metadata.manual_subtitles == {}
    assert metadata.automatic_captions == {}

    options = youtube_dl.call_args.args[0]
    assert options["noplaylist"] is True
    assert options["skip_download"] is True
    assert options["writesubtitles"] is False
    assert options["writeautomaticsub"] is False
    assert options["writethumbnail"] is False
    downloader.extract_info.assert_called_once_with(supplied_url, download=False)


def test_extract_metadata_keeps_manual_and_automatic_tracks_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_youtube_dl(
        monkeypatch,
        response={
            "title": "Multilingual Video",
            "webpage_url": "https://example.com/video",
            "subtitles": {
                "en": [
                    {
                        "ext": "vtt",
                        "url": "https://example.com/en.vtt",
                        "http_headers": {"Referer": "https://example.com/"},
                    },
                    {"ext": "srt", "url": "https://example.com/en.srt"},
                    {"ext": "vtt", "url": "https://example.com/en-duplicate.vtt"},
                ],
                "en-US": [{"ext": "json3", "url": "https://example.com/en-us"}],
            },
            "automatic_captions": {
                "ja": [{"ext": "vtt", "url": "https://example.com/ja.vtt"}],
                "ja-JP": [{"ext": "vtt", "data": "WEBVTT\n"}],
                "fr": [{"ext": "srv3", "url": "https://example.com/fr"}],
            },
        },
    )

    metadata = extractor.extract_metadata("https://example.com/video")

    assert metadata.manual_subtitles == {
        "en": (
            SubtitleTrack(
                extension="vtt",
                url="https://example.com/en.vtt",
                http_headers={"Referer": "https://example.com/"},
            ),
            SubtitleTrack(extension="srt", url="https://example.com/en.srt"),
            SubtitleTrack(
                extension="vtt",
                url="https://example.com/en-duplicate.vtt",
            ),
        ),
        "en-US": (
            SubtitleTrack(extension="json3", url="https://example.com/en-us"),
        ),
    }
    assert metadata.automatic_captions == {
        "ja": (SubtitleTrack(extension="vtt", url="https://example.com/ja.vtt"),),
        "ja-JP": (SubtitleTrack(extension="vtt", content="WEBVTT\n"),),
        "fr": (SubtitleTrack(extension="srv3", url="https://example.com/fr"),),
    }


def test_extract_metadata_uses_optional_field_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied_url = "https://example.com/original"
    _mock_youtube_dl(
        monkeypatch,
        response={
            "title": "Minimal Video",
            "uploader": "Uploader Name",
            "duration": None,
            "subtitles": None,
            "automatic_captions": {},
        },
    )

    metadata = extractor.extract_metadata(supplied_url)

    assert metadata.webpage_url == supplied_url
    assert metadata.uploader == "Uploader Name"
    assert metadata.duration_seconds is None
    assert metadata.manual_subtitles == {}
    assert metadata.automatic_captions == {}


def test_extract_metadata_allows_missing_uploader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_youtube_dl(monkeypatch, response={"title": "Anonymous Video"})

    metadata = extractor.extract_metadata("https://example.com/video")

    assert metadata.uploader is None


def test_extract_metadata_converts_download_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = DownloadError("network failed")
    _mock_youtube_dl(monkeypatch, error=original_error)

    with pytest.raises(ExtractionError) as error_info:
        extractor.extract_metadata("https://example.com/unavailable")

    assert str(error_info.value) == "Could not extract video metadata."
    assert error_info.value.__cause__ is original_error


@pytest.mark.parametrize("result_type", ["playlist", "multi_video"])
def test_extract_metadata_rejects_multi_entry_results(
    monkeypatch: pytest.MonkeyPatch,
    result_type: str,
) -> None:
    _mock_youtube_dl(
        monkeypatch,
        response={
            "_type": result_type,
            "title": "Several Videos",
            "entries": [{"title": "First"}, {"title": "Second"}],
        },
    )

    with pytest.raises(ExtractionError, match="Only single-video URLs"):
        extractor.extract_metadata("https://example.com/playlist")


def test_extract_metadata_rejects_an_entries_collection_on_video_shaped_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_youtube_dl(
        monkeypatch,
        response={
            "title": "Ambiguous Result",
            "entries": [{"title": "Nested Video"}],
        },
    )

    with pytest.raises(ExtractionError, match="Only single-video URLs"):
        extractor.extract_metadata("https://example.com/ambiguous")


def test_extract_metadata_rejects_missing_required_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_youtube_dl(monkeypatch, response={"webpage_url": "https://example.com"})

    with pytest.raises(ExtractionError, match="did not include a video title"):
        extractor.extract_metadata("https://example.com/video")
