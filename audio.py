"""Audio-only downloads within a caller-owned temporary directory."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from url_transcriber.errors import AudioDownloadError


def download_audio(url: str, temp_dir: Path) -> Path:
    """Download one audio-only stream into ``temp_dir`` and return its path.

    The caller owns the directory lifetime and is responsible for removing it
    after the returned file has been consumed.
    """
    destination = _validate_destination(temp_dir)
    options = _download_options(destination)

    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            reported_path = _reported_download_path(info, downloader)
    except DownloadError as error:
        raise AudioDownloadError("Could not download an audio-only format.") from error

    return _validate_downloaded_file(reported_path, destination)


def _validate_destination(temp_dir: Path) -> Path:
    try:
        destination = temp_dir.resolve(strict=True)
    except OSError as error:
        raise AudioDownloadError("The temporary audio directory does not exist.") from error

    if not destination.is_dir():
        raise AudioDownloadError("The temporary audio destination is not a directory.")
    return destination


def _download_options(destination: Path) -> dict[str, Any]:
    return {
        "format": "bestaudio",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "outtmpl": str(destination / "audio.%(ext)s"),
        "paths": {
            "home": str(destination),
            "temp": str(destination),
        },
        "writesubtitles": False,
        "writeautomaticsub": False,
        "writethumbnail": False,
    }


def _reported_download_path(info: object, downloader: YoutubeDL) -> Path:
    if not isinstance(info, Mapping):
        raise AudioDownloadError("yt-dlp returned no usable download information.")
    if info.get("_type") in {"playlist", "multi_video"} or "entries" in info:
        raise AudioDownloadError("Only single-video URLs are supported.")

    requested_downloads = info.get("requested_downloads")
    if isinstance(requested_downloads, list):
        paths = {
            path
            for item in requested_downloads
            if isinstance(item, Mapping)
            if isinstance(path := item.get("filepath"), str) and path
        }
        if len(paths) > 1:
            raise AudioDownloadError("yt-dlp reported multiple downloaded files.")
        if paths:
            return Path(paths.pop())

    filepath = info.get("filepath")
    if isinstance(filepath, str) and filepath:
        return Path(filepath)

    prepared_filename = downloader.prepare_filename(info)
    if not isinstance(prepared_filename, str) or not prepared_filename:
        raise AudioDownloadError("yt-dlp did not report a downloaded audio file.")
    return Path(prepared_filename)


def _validate_downloaded_file(reported_path: Path, destination: Path) -> Path:
    try:
        downloaded_path = reported_path.resolve(strict=True)
    except OSError as error:
        raise AudioDownloadError("Downloaded audio file was not found.") from error

    try:
        downloaded_path.relative_to(destination)
    except ValueError as error:
        raise AudioDownloadError(
            "yt-dlp reported an audio file outside the temporary directory."
        ) from error

    if not downloaded_path.is_file():
        raise AudioDownloadError("The downloaded audio path is not a regular file.")
    return downloaded_path
