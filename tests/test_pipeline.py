"""Orchestration tests for the single-URL core pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from url_transcriber import pipeline
from url_transcriber.errors import (
    AudioDownloadError,
    ExtractionError,
    OutputError,
    SubtitleError,
    TranscriptionError,
)
from url_transcriber.models import (
    SubtitleTrack,
    TranscriptResult,
    TranscriptSegment,
    TranscriptSource,
    VideoMetadata,
)
from url_transcriber.subtitles import process_subtitles


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        title="Example Video",
        webpage_url="https://example.com/video",
        uploader="Example Channel",
        duration_seconds=60,
    )


def _transcript(source: TranscriptSource = "local Whisper") -> TranscriptResult:
    model = "small" if source == "local Whisper" else None
    return TranscriptResult(
        segments=[TranscriptSegment(0, "Transcript text")],
        language="en",
        source=source,
        whisper_model=model,
    )


def _mock_boundaries(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    mocks = {
        "extract": MagicMock(return_value=_metadata()),
        "subtitles": MagicMock(return_value=None),
        "download": MagicMock(),
        "transcribe": MagicMock(return_value=_transcript()),
        "format": MagicMock(return_value="# Rendered transcript\n"),
        "write": MagicMock(return_value=Path("outputs/example-video.md")),
    }
    monkeypatch.setattr(pipeline, "extract_metadata", mocks["extract"])
    monkeypatch.setattr(pipeline, "process_subtitles", mocks["subtitles"])
    monkeypatch.setattr(pipeline, "download_audio", mocks["download"])
    monkeypatch.setattr(pipeline, "transcribe_audio", mocks["transcribe"])
    monkeypatch.setattr(pipeline, "format_markdown", mocks["format"])
    monkeypatch.setattr(pipeline, "write_markdown", mocks["write"])
    return mocks


@pytest.mark.parametrize("source", ["manual subtitles", "automatic captions"])
def test_subtitle_branch_skips_temporary_audio_and_whisper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    source: TranscriptSource,
) -> None:
    mocks = _mock_boundaries(monkeypatch)
    metadata = mocks["extract"].return_value
    transcript = _transcript(source)
    output_path = tmp_path / "outputs" / "example-video.md"
    mocks["subtitles"].return_value = transcript
    mocks["write"].return_value = output_path
    temporary_directory = MagicMock(side_effect=AssertionError("unexpected temp dir"))
    monkeypatch.setattr(pipeline, "TemporaryDirectory", temporary_directory)

    result = pipeline.run_url_pipeline("https://example.com/video", tmp_path / "outputs")

    mocks["extract"].assert_called_once_with("https://example.com/video")
    mocks["subtitles"].assert_called_once_with(metadata)
    mocks["download"].assert_not_called()
    mocks["transcribe"].assert_not_called()
    temporary_directory.assert_not_called()
    mocks["format"].assert_called_once_with(metadata, transcript)
    mocks["write"].assert_called_once_with(
        "# Rendered transcript\n",
        metadata.title,
        tmp_path / "outputs",
    )
    assert result == pipeline.PipelineResult(output_path, metadata, transcript)
    assert result.transcript.source == source
    assert capsys.readouterr().out == ""


def test_no_subtitle_runs_whisper_with_live_temporary_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mocks = _mock_boundaries(monkeypatch)
    output_dir = tmp_path / "outputs"
    state: dict[str, Path] = {}

    def download(url: str, temp_dir: Path) -> Path:
        assert url == "https://example.com/video"
        assert temp_dir.exists() and temp_dir.is_dir()
        assert temp_dir != output_dir
        assert temp_dir.name.startswith("url-transcriber-")
        audio_path = temp_dir / "audio.webm"
        audio_path.write_bytes(b"temporary audio")
        state.update(temp_dir=temp_dir, audio_path=audio_path)
        return audio_path

    def transcribe(audio_path: Path) -> TranscriptResult:
        assert audio_path == state["audio_path"]
        assert audio_path.exists()
        return _transcript()

    mocks["download"].side_effect = download
    mocks["transcribe"].side_effect = transcribe

    result = pipeline.run_url_pipeline("https://example.com/video", output_dir)

    assert result.transcript.source == "local Whisper"
    mocks["download"].assert_called_once()
    mocks["transcribe"].assert_called_once_with(state["audio_path"])
    mocks["format"].assert_called_once_with(_metadata(), result.transcript)
    mocks["write"].assert_called_once_with(
        "# Rendered transcript\n",
        "Example Video",
        output_dir,
    )
    assert not state["audio_path"].exists()
    assert not state["temp_dir"].exists()


def test_non_original_subtitle_falls_back_to_whisper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mocks = _mock_boundaries(monkeypatch)
    metadata = _metadata()
    metadata.original_language = "ja"
    metadata.automatic_captions = {
        "en": (
            SubtitleTrack(
                extension="vtt",
                content="WEBVTT\n\n00:00.000 --> 00:01.000\nTranslated text.",
            ),
        )
    }
    mocks["extract"].return_value = metadata
    mocks["subtitles"].side_effect = process_subtitles

    def download(_url: str, temp_dir: Path) -> Path:
        audio_path = temp_dir / "original-audio.webm"
        audio_path.touch()
        return audio_path

    mocks["download"].side_effect = download

    result = pipeline.run_url_pipeline("https://example.com/video", tmp_path)

    mocks["subtitles"].assert_called_once_with(metadata)
    mocks["download"].assert_called_once()
    mocks["transcribe"].assert_called_once()
    assert result.transcript.source == "local Whisper"


def test_subtitle_error_falls_back_to_whisper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mocks = _mock_boundaries(monkeypatch)
    mocks["subtitles"].side_effect = SubtitleError("subtitle unavailable")

    def download(_url: str, temp_dir: Path) -> Path:
        audio_path = temp_dir / "audio.m4a"
        audio_path.touch()
        return audio_path

    mocks["download"].side_effect = download

    result = pipeline.run_url_pipeline("https://example.com/video", tmp_path)

    assert result.transcript.source == "local Whisper"
    mocks["download"].assert_called_once()
    mocks["transcribe"].assert_called_once()


def test_unexpected_subtitle_error_propagates_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mocks = _mock_boundaries(monkeypatch)
    original_error = RuntimeError("bug")
    mocks["subtitles"].side_effect = original_error

    with pytest.raises(RuntimeError) as error_info:
        pipeline.run_url_pipeline("https://example.com/video", tmp_path)

    assert error_info.value is original_error
    mocks["download"].assert_not_called()
    mocks["transcribe"].assert_not_called()
    mocks["format"].assert_not_called()
    mocks["write"].assert_not_called()


def test_extraction_error_stops_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mocks = _mock_boundaries(monkeypatch)
    original_error = ExtractionError("unavailable")
    mocks["extract"].side_effect = original_error

    with pytest.raises(ExtractionError) as error_info:
        pipeline.run_url_pipeline("https://example.com/video", tmp_path)

    assert error_info.value is original_error
    for boundary in ("subtitles", "download", "transcribe", "format", "write"):
        mocks[boundary].assert_not_called()


def test_audio_error_propagates_without_transcription(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mocks = _mock_boundaries(monkeypatch)
    original_error = AudioDownloadError("download failed")
    state: dict[str, Path] = {}

    def fail_download(_url: str, temp_dir: Path) -> Path:
        state["temp_dir"] = temp_dir
        partial_path = temp_dir / "audio.webm.part"
        partial_path.write_bytes(b"partial audio")
        state["partial_path"] = partial_path
        raise original_error

    mocks["download"].side_effect = fail_download

    with pytest.raises(AudioDownloadError) as error_info:
        pipeline.run_url_pipeline("https://example.com/video", tmp_path)

    assert error_info.value is original_error
    mocks["transcribe"].assert_not_called()
    assert not state["partial_path"].exists()
    assert not state["temp_dir"].exists()


@pytest.mark.parametrize(
    ("failing_boundary", "error"),
    [
        ("transcribe", TranscriptionError("transcription failed")),
        ("format", RuntimeError("formatter bug")),
        ("write", OutputError("output failed")),
    ],
)
def test_whisper_branch_cleans_up_after_later_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failing_boundary: str,
    error: Exception,
) -> None:
    mocks = _mock_boundaries(monkeypatch)
    state: dict[str, Path] = {}

    def download(_url: str, temp_dir: Path) -> Path:
        audio_path = temp_dir / "audio.webm"
        audio_path.write_bytes(b"temporary audio")
        state.update(temp_dir=temp_dir, audio_path=audio_path)
        return audio_path

    mocks["download"].side_effect = download
    mocks[failing_boundary].side_effect = error

    with pytest.raises(type(error)) as error_info:
        pipeline.run_url_pipeline("https://example.com/video", tmp_path / "outputs")

    assert error_info.value is error
    assert not state["audio_path"].exists()
    assert not state["temp_dir"].exists()

    if failing_boundary == "transcribe":
        mocks["format"].assert_not_called()
        mocks["write"].assert_not_called()
    elif failing_boundary == "format":
        mocks["write"].assert_not_called()
