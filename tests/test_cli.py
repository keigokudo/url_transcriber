"""Tests for the thin command-line adapter."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from url_transcriber import cli
from url_transcriber.errors import (
    AudioDownloadError,
    ExtractionError,
    OutputError,
    TranscriptionError,
)
from url_transcriber.models import (
    TranscriptResult,
    TranscriptSegment,
    TranscriptSource,
    VideoMetadata,
)
from url_transcriber.pipeline import PipelineResult


def _pipeline_result(
    source: TranscriptSource,
    *,
    duration_seconds: float | None = 151,
) -> PipelineResult:
    metadata = VideoMetadata(
        title="Example Video",
        webpage_url="https://example.com/video",
        uploader="Example Channel",
        duration_seconds=duration_seconds,
    )
    transcript = TranscriptResult(
        segments=[TranscriptSegment(0, "Transcript text")],
        language="en",
        source=source,
        whisper_model="small" if source == "local Whisper" else None,
    )
    return PipelineResult(
        output_path=Path("outputs") / "example-video.md",
        metadata=metadata,
        transcript=transcript,
    )


@pytest.mark.parametrize(
    ("source", "expected_label"),
    [
        ("manual subtitles", "YouTube manual subtitles"),
        ("automatic captions", "YouTube auto captions"),
        ("local Whisper", "Local Whisper"),
    ],
)
def test_url_command_displays_success_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: TranscriptSource,
    expected_label: str,
) -> None:
    run_pipeline = MagicMock(return_value=_pipeline_result(source))
    monkeypatch.setattr(cli, "run_url_pipeline", run_pipeline)
    supplied_url = "https://example.com/watch?v=abc&list=unchanged"

    exit_code = cli.main(["url", supplied_url])

    assert exit_code == 0
    run_pipeline.assert_called_once_with(supplied_url)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Processing URL..." in captured.out
    assert "Found:\nExample Video" in captured.out
    assert "Duration:\n02:31" in captured.out
    assert f"Saved:\n{Path('outputs') / 'example-video.md'}" in captured.out
    assert f"Transcript source:\n{expected_label}" in captured.out


def test_url_command_displays_unknown_duration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "run_url_pipeline",
        MagicMock(return_value=_pipeline_result("manual subtitles", duration_seconds=None)),
    )

    assert cli.main(["url", "https://example.com/video"]) == 0

    assert "Duration:\nUnknown" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        ExtractionError("Could not extract video metadata."),
        AudioDownloadError("Could not download audio."),
        TranscriptionError("Could not transcribe audio."),
        OutputError("Could not write transcript."),
    ],
)
def test_expected_application_errors_are_printed_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    monkeypatch.setattr(cli, "run_url_pipeline", MagicMock(side_effect=error))

    exit_code = cli.main(["url", "https://example.com/video"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.err == f"Error:\n{error}\n"
    assert "Traceback" not in captured.err


def test_unexpected_pipeline_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = RuntimeError("programming bug")
    monkeypatch.setattr(
        cli,
        "run_url_pipeline",
        MagicMock(side_effect=original_error),
    )

    with pytest.raises(RuntimeError) as error_info:
        cli.main(["url", "https://example.com/video"])

    assert error_info.value is original_error


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["url"],
        ["unknown"],
    ],
)
def test_argparse_rejects_invalid_usage(
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as error_info:
        cli.main(arguments)

    assert error_info.value.code != 0
    assert "usage:" in capsys.readouterr().err.lower()
