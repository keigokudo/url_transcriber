"""Network-free tests for temporary audio-only downloads."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock

import pytest
from yt_dlp.utils import DownloadError

from url_transcriber import audio
from url_transcriber.errors import AudioDownloadError


def _mock_download(
    monkeypatch: pytest.MonkeyPatch,
    result: Any,
) -> tuple[MagicMock, MagicMock]:
    downloader = MagicMock()
    downloader.extract_info.return_value = result
    context_manager = MagicMock()
    context_manager.__enter__.return_value = downloader
    youtube_dl = MagicMock(return_value=context_manager)
    monkeypatch.setattr(audio, "YoutubeDL", youtube_dl)
    return youtube_dl, downloader


@pytest.mark.parametrize("extension", ["m4a", "webm"])
def test_download_audio_returns_actual_file_for_different_extensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extension: str,
) -> None:
    downloaded_file = tmp_path / f"audio.{extension}"
    downloaded_file.write_bytes(b"audio bytes")
    youtube_dl, downloader = _mock_download(
        monkeypatch,
        {"requested_downloads": [{"filepath": str(downloaded_file)}]},
    )

    result = audio.download_audio("https://example.com/video", tmp_path)

    assert isinstance(result, Path)
    assert result == downloaded_file.resolve()
    assert result.exists()
    assert result.is_relative_to(tmp_path.resolve())
    downloader.extract_info.assert_called_once_with(
        "https://example.com/video",
        download=True,
    )

    options = youtube_dl.call_args.args[0]
    assert options["format"] == "bestaudio"
    assert "/best" not in options["format"]
    assert options["noplaylist"] is True
    assert options["noprogress"] is True
    assert options["outtmpl"] == str(tmp_path.resolve() / "audio.%(ext)s")
    assert options["paths"] == {
        "home": str(tmp_path.resolve()),
        "temp": str(tmp_path.resolve()),
    }
    assert "postprocessors" not in options


def test_download_audio_uses_official_filename_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloaded_file = tmp_path / "audio.opus"
    downloaded_file.write_bytes(b"opus")
    _, downloader = _mock_download(monkeypatch, {"ext": "opus"})
    downloader.prepare_filename.return_value = str(downloaded_file)

    assert audio.download_audio("https://example.com/video", tmp_path) == (
        downloaded_file.resolve()
    )
    downloader.prepare_filename.assert_called_once_with({"ext": "opus"})


def test_download_audio_converts_yt_dlp_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_error = DownloadError("no audio format")
    downloader = MagicMock()
    downloader.extract_info.side_effect = original_error
    context_manager = MagicMock()
    context_manager.__enter__.return_value = downloader
    monkeypatch.setattr(audio, "YoutubeDL", MagicMock(return_value=context_manager))

    with pytest.raises(AudioDownloadError) as error_info:
        audio.download_audio("https://example.com/video", tmp_path)

    assert error_info.value.__cause__ is original_error


def test_download_audio_rejects_missing_output_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_file = tmp_path / "audio.webm"
    _mock_download(
        monkeypatch,
        {"requested_downloads": [{"filepath": str(missing_file)}]},
    )

    with pytest.raises(AudioDownloadError, match="was not found"):
        audio.download_audio("https://example.com/video", tmp_path)


def test_download_audio_rejects_multiple_reported_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "audio.m4a"
    second = tmp_path / "audio.webm"
    first.touch()
    second.touch()
    _mock_download(
        monkeypatch,
        {
            "requested_downloads": [
                {"filepath": str(first)},
                {"filepath": str(second)},
            ]
        },
    )

    with pytest.raises(AudioDownloadError, match="multiple downloaded files"):
        audio.download_audio("https://example.com/video", tmp_path)


def test_download_audio_rejects_file_outside_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    outside_file = tmp_path / "outside.webm"
    outside_file.touch()
    _mock_download(
        monkeypatch,
        {"requested_downloads": [{"filepath": str(outside_file)}]},
    )

    with pytest.raises(AudioDownloadError, match="outside the temporary directory"):
        audio.download_audio("https://example.com/video", destination)


@pytest.mark.parametrize("destination_kind", ["missing", "file"])
def test_download_audio_rejects_invalid_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    destination_kind: str,
) -> None:
    destination = tmp_path / destination_kind
    if destination_kind == "file":
        destination.touch()
    youtube_dl = MagicMock()
    monkeypatch.setattr(audio, "YoutubeDL", youtube_dl)

    with pytest.raises(AudioDownloadError, match="temporary audio"):
        audio.download_audio("https://example.com/video", destination)

    youtube_dl.assert_not_called()


def test_download_audio_supports_caller_owned_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloader = MagicMock()

    def create_audio(_url: str, *, download: bool) -> dict[str, Any]:
        assert download is True
        output_template = Path(youtube_dl.call_args.args[0]["outtmpl"])
        downloaded_file = output_template.with_name("audio.webm")
        downloaded_file.write_bytes(b"temporary audio")
        return {"requested_downloads": [{"filepath": str(downloaded_file)}]}

    downloader.extract_info.side_effect = create_audio
    context_manager = MagicMock()
    context_manager.__enter__.return_value = downloader
    youtube_dl = MagicMock(return_value=context_manager)
    monkeypatch.setattr(audio, "YoutubeDL", youtube_dl)

    with TemporaryDirectory(dir=tmp_path) as temporary_name:
        temporary_path = Path(temporary_name)
        downloaded_path = audio.download_audio(
            "https://example.com/video",
            temporary_path,
        )
        assert downloaded_path.exists()
        assert downloaded_path.parent == temporary_path.resolve()

    assert not temporary_path.exists()
