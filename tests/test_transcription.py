"""Tests for local faster-whisper result normalization."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from url_transcriber import transcription
from url_transcriber.errors import TranscriptionError
from url_transcriber.models import TranscriptSegment


def _audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "audio.webm"
    path.write_bytes(b"mock audio")
    return path


def _mock_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    segments: object,
    language: object = "en",
) -> tuple[MagicMock, MagicMock]:
    model = MagicMock()
    model.transcribe.return_value = (segments, SimpleNamespace(language=language))
    whisper_model = MagicMock(return_value=model)
    monkeypatch.setattr(transcription, "WhisperModel", whisper_model)
    return whisper_model, model


def test_transcribe_audio_uses_cpu_int8_defaults_and_converts_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = _audio_file(tmp_path)
    whisper_model, model = _mock_model(
        monkeypatch,
        segments=[
            SimpleNamespace(start=0.0, text=" Hello world "),
            SimpleNamespace(start=12.4, text=" Second segment "),
        ],
    )

    result = transcription.transcribe_audio(audio_path)

    whisper_model.assert_called_once_with(
        "small",
        device="cpu",
        compute_type="int8",
    )
    model.transcribe.assert_called_once_with(str(audio_path.resolve()))
    assert result.segments == [
        TranscriptSegment(0.0, "Hello world"),
        TranscriptSegment(12.4, "Second segment"),
    ]
    assert result.language == "en"
    assert result.source == "local Whisper"
    assert result.whisper_model == "small"
    assert audio_path.exists()


@pytest.mark.parametrize("language", ["ja", "fr"])
def test_transcribe_audio_preserves_detected_language(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    language: str,
) -> None:
    audio_path = _audio_file(tmp_path)
    _mock_model(
        monkeypatch,
        segments=[SimpleNamespace(start=1, text="テスト")],
        language=language,
    )

    assert transcription.transcribe_audio(audio_path).language == language


def test_transcribe_audio_uses_unknown_language_representation_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = _audio_file(tmp_path)
    _mock_model(
        monkeypatch,
        segments=[SimpleNamespace(start=0, text="Speech")],
        language=None,
    )

    assert transcription.transcribe_audio(audio_path).language == ""


def test_transcribe_audio_skips_empty_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = _audio_file(tmp_path)
    _mock_model(
        monkeypatch,
        segments=[
            SimpleNamespace(start=0, text=""),
            SimpleNamespace(start=1, text="   "),
            SimpleNamespace(start=2, text=" Kept "),
        ],
    )

    result = transcription.transcribe_audio(audio_path)

    assert result.segments == [TranscriptSegment(2.0, "Kept")]


def test_transcribe_audio_rejects_all_empty_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = _audio_file(tmp_path)
    _mock_model(
        monkeypatch,
        segments=[SimpleNamespace(start=0, text=" ")],
    )

    with pytest.raises(TranscriptionError, match="no usable transcript"):
        transcription.transcribe_audio(audio_path)


def test_transcribe_audio_rejects_missing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    whisper_model = MagicMock()
    monkeypatch.setattr(transcription, "WhisperModel", whisper_model)

    with pytest.raises(TranscriptionError, match="does not exist"):
        transcription.transcribe_audio(tmp_path / "missing.webm")

    whisper_model.assert_not_called()


def test_transcribe_audio_rejects_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    whisper_model = MagicMock()
    monkeypatch.setattr(transcription, "WhisperModel", whisper_model)

    with pytest.raises(TranscriptionError, match="not a regular file"):
        transcription.transcribe_audio(tmp_path)

    whisper_model.assert_not_called()


def test_transcribe_audio_converts_model_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = _audio_file(tmp_path)
    original_error = RuntimeError("model unavailable")
    monkeypatch.setattr(
        transcription,
        "WhisperModel",
        MagicMock(side_effect=original_error),
    )

    with pytest.raises(TranscriptionError, match="load") as error_info:
        transcription.transcribe_audio(audio_path)

    assert error_info.value.__cause__ is original_error
    assert audio_path.exists()


def test_transcribe_audio_converts_initial_transcription_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = _audio_file(tmp_path)
    original_error = RuntimeError("decode failed")
    model = MagicMock()
    model.transcribe.side_effect = original_error
    monkeypatch.setattr(transcription, "WhisperModel", MagicMock(return_value=model))

    with pytest.raises(TranscriptionError, match="transcribe") as error_info:
        transcription.transcribe_audio(audio_path)

    assert error_info.value.__cause__ is original_error
    assert audio_path.exists()


def test_transcribe_audio_converts_lazy_iteration_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = _audio_file(tmp_path)
    original_error = RuntimeError("inference failed during iteration")

    def failing_segments() -> Iterator[object]:
        yield SimpleNamespace(start=0, text="Partial text")
        raise original_error

    _mock_model(monkeypatch, segments=failing_segments())

    with pytest.raises(TranscriptionError, match="transcription failed") as error_info:
        transcription.transcribe_audio(audio_path)

    assert error_info.value.__cause__ is original_error
    assert audio_path.exists()


def test_transcribe_audio_records_nondefault_model_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = _audio_file(tmp_path)
    whisper_model, _ = _mock_model(
        monkeypatch,
        segments=[SimpleNamespace(start=0, text="Text")],
    )

    result = transcription.transcribe_audio(audio_path, model_name="tiny")

    whisper_model.assert_called_once_with("tiny", device="cpu", compute_type="int8")
    assert result.whisper_model == "tiny"
