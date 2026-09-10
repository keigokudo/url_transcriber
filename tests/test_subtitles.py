"""Tests for subtitle selection, retrieval, parsing, and cleaning."""

from unittest.mock import MagicMock

import pytest
from yt_dlp.networking.exceptions import RequestError

from url_transcriber import subtitles
from url_transcriber.errors import SubtitleError
from url_transcriber.models import SubtitleTrack, TranscriptSegment, VideoMetadata


def _metadata(
    *,
    manual: dict[str, tuple[SubtitleTrack, ...]] | None = None,
    automatic: dict[str, tuple[SubtitleTrack, ...]] | None = None,
) -> VideoMetadata:
    return VideoMetadata(
        title="Example",
        webpage_url="https://example.com/video",
        uploader="Channel",
        duration_seconds=60,
        manual_subtitles=manual or {},
        automatic_captions=automatic or {},
    )


def _track(
    extension: str = "vtt",
    *,
    content: str | None = "WEBVTT",
    url: str | None = None,
) -> SubtitleTrack:
    return SubtitleTrack(extension=extension, content=content, url=url)


def test_selection_prefers_manual_english_over_automatic_english() -> None:
    metadata = _metadata(
        manual={"en": (_track(),)},
        automatic={"en": (_track(),)},
    )

    selection = subtitles.select_subtitle(metadata)

    assert selection is not None
    assert selection.source == "manual subtitles"
    assert selection.language == "en"


def test_selection_prefers_manual_japanese_over_automatic_english() -> None:
    metadata = _metadata(
        manual={"ja": (_track(),)},
        automatic={"en": (_track(),)},
    )

    selection = subtitles.select_subtitle(metadata)

    assert selection is not None
    assert selection.source == "manual subtitles"
    assert selection.language == "ja"


def test_selection_uses_automatic_when_no_supported_manual_track_exists() -> None:
    metadata = _metadata(
        manual={"fr": (_track(),)},
        automatic={"en": (_track(),)},
    )

    selection = subtitles.select_subtitle(metadata)

    assert selection is not None
    assert selection.source == "automatic captions"
    assert selection.language == "en"


@pytest.mark.parametrize("language", ["en", "en-US", "en-GB", "ja", "ja-JP"])
def test_selection_recognizes_supported_language_variants(language: str) -> None:
    selection = subtitles.select_subtitle(_metadata(manual={language: (_track(),)}))

    assert selection is not None
    assert selection.language == language


def test_selection_prefers_exact_root_then_sorted_variants() -> None:
    metadata = _metadata(
        manual={
            "en-US": (_track(),),
            "en-GB": (_track(),),
            "en": (_track(),),
        }
    )

    selection = subtitles.select_subtitle(metadata)
    assert selection is not None
    assert selection.language == "en"

    del metadata.manual_subtitles["en"]
    selection = subtitles.select_subtitle(metadata)
    assert selection is not None
    assert selection.language == "en-GB"


def test_selection_chooses_vtt_and_rejects_unsupported_formats() -> None:
    metadata = _metadata(
        manual={"en": (_track("json3"), _track("vtt"))},
    )

    selection = subtitles.select_subtitle(metadata)

    assert selection is not None
    assert selection.track.extension == "vtt"
    assert subtitles.select_subtitle(
        _metadata(manual={"en": (_track("json3"), _track("srv3"))})
    ) is None


def test_selection_returns_none_for_unsupported_languages() -> None:
    assert subtitles.select_subtitle(_metadata(manual={"fr": (_track(),)})) is None


def test_fetch_subtitle_uses_inline_content_without_http(monkeypatch: pytest.MonkeyPatch) -> None:
    youtube_dl = MagicMock()
    monkeypatch.setattr(subtitles, "YoutubeDL", youtube_dl)
    selection = subtitles.SubtitleSelection(
        "en",
        "manual subtitles",
        SubtitleTrack(extension="vtt", content="WEBVTT\n"),
    )

    assert subtitles.fetch_subtitle(selection) == "WEBVTT\n"
    youtube_dl.assert_not_called()


def test_fetch_subtitle_uses_yt_dlp_http_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.read.return_value = "WEBVTT\n日本語".encode()
    downloader = MagicMock()
    downloader.urlopen.return_value.__enter__.return_value = response
    youtube_dl = MagicMock()
    youtube_dl.return_value.__enter__.return_value = downloader
    monkeypatch.setattr(subtitles, "YoutubeDL", youtube_dl)
    selection = subtitles.SubtitleSelection(
        "ja",
        "manual subtitles",
        SubtitleTrack(
            extension="vtt",
            url="https://example.com/ja.vtt",
            http_headers={"Referer": "https://example.com/"},
        ),
    )

    content = subtitles.fetch_subtitle(selection)

    assert content == "WEBVTT\n日本語"
    request = downloader.urlopen.call_args.args[0]
    assert request.url == "https://example.com/ja.vtt"
    assert request.headers["Referer"] == "https://example.com/"


def test_fetch_subtitle_converts_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = MagicMock()
    original_error = RequestError("unavailable")
    downloader.urlopen.side_effect = original_error
    youtube_dl = MagicMock()
    youtube_dl.return_value.__enter__.return_value = downloader
    monkeypatch.setattr(subtitles, "YoutubeDL", youtube_dl)
    selection = subtitles.SubtitleSelection(
        "en",
        "manual subtitles",
        SubtitleTrack(extension="vtt", url="https://example.com/en.vtt"),
    )

    with pytest.raises(SubtitleError) as error_info:
        subtitles.fetch_subtitle(selection)

    assert error_info.value.__cause__ is original_error


def test_parse_vtt_supports_headers_ids_timestamps_markup_and_empty_cues() -> None:
    content = """WEBVTT

NOTE this block is ignored
even over multiple lines

cue-one
00:01.250 --> 00:03.000 align:start position:0%
<c.green>Hello</c>   &amp; <b>world</b>!

cue-two
01:02:03.500 --> 01:02:05.000
日本語の字幕です。

cue-empty
01:02:06.000 --> 01:02:07.000
<c></c>
"""

    segments = subtitles.parse_vtt(content)

    assert segments == [
        TranscriptSegment(1.25, "Hello & world!"),
        TranscriptSegment(3723.5, "日本語の字幕です。"),
    ]


def test_parse_vtt_parses_multiple_cues_and_blank_lines() -> None:
    content = """WEBVTT

00:00:00.000 --> 00:00:01.000
First cue.


00:00:18.000 --> 00:00:20.000
Second cue.
continues here.
"""

    assert subtitles.parse_vtt(content) == [
        TranscriptSegment(0.0, "First cue."),
        TranscriptSegment(18.0, "Second cue. continues here."),
    ]


def test_parse_vtt_rejects_malformed_or_empty_content() -> None:
    with pytest.raises(SubtitleError, match="timestamp"):
        subtitles.parse_vtt("WEBVTT\n\n00:bad --> 00:02.000\nText")

    with pytest.raises(SubtitleError, match="timestamp"):
        subtitles.parse_vtt("WEBVTT\n\n00:61:00.000 --> 01:02:00.000\nText")

    with pytest.raises(SubtitleError, match="no usable VTT cues"):
        subtitles.parse_vtt("WEBVTT\n\nNOTE no transcript")


def test_clean_segments_removes_exact_adjacent_duplicates() -> None:
    segments = [
        TranscriptSegment(1, "Hello world"),
        TranscriptSegment(2, "Hello world"),
        TranscriptSegment(3, "Hello world again"),
    ]

    assert subtitles.clean_segments(segments, remove_automatic_overlap=False) == [
        TranscriptSegment(1, "Hello world"),
        TranscriptSegment(3, "Hello world again"),
    ]


def test_clean_segments_removes_only_clear_automatic_caption_overlap() -> None:
    segments = [
        TranscriptSegment(1, "Today we're going"),
        TranscriptSegment(2, "Today we're going to talk"),
        TranscriptSegment(3, "we're going to talk about Python"),
    ]

    assert subtitles.clean_segments(segments, remove_automatic_overlap=True) == [
        TranscriptSegment(1, "Today we're going"),
        TranscriptSegment(2, "to talk"),
        TranscriptSegment(3, "about Python"),
    ]


def test_clean_segments_preserves_uncertain_repeated_speech() -> None:
    segments = [
        TranscriptSegment(1, "Yes, yes."),
        TranscriptSegment(2, "Yes, please continue."),
        TranscriptSegment(3, "行く 行く"),
    ]

    assert subtitles.clean_segments(segments, remove_automatic_overlap=True) == segments


def test_process_subtitles_builds_manual_result() -> None:
    content = "WEBVTT\n\n00:00.000 --> 00:01.000\nHello."
    metadata = _metadata(manual={"en": (_track(content=content),)})

    result = subtitles.process_subtitles(metadata)

    assert result is not None
    assert result.source == "manual subtitles"
    assert result.language == "en"
    assert result.whisper_model is None
    assert result.segments == [TranscriptSegment(0, "Hello.")]


def test_process_subtitles_builds_cleaned_automatic_result() -> None:
    content = """WEBVTT

00:01.000 --> 00:02.000
Hello world

00:02.000 --> 00:03.000
Hello world again
"""
    metadata = _metadata(automatic={"ja-JP": (_track(content=content),)})

    result = subtitles.process_subtitles(metadata)

    assert result is not None
    assert result.source == "automatic captions"
    assert result.language == "ja-JP"
    assert result.whisper_model is None
    assert result.segments == [
        TranscriptSegment(1, "Hello world"),
        TranscriptSegment(2, "again"),
    ]


def test_process_subtitles_returns_none_when_no_supported_track_exists() -> None:
    assert subtitles.process_subtitles(_metadata(manual={"fr": (_track(),)})) is None
