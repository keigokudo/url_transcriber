"""Filesystem writing for final Markdown transcripts."""

from pathlib import Path

from url_transcriber.errors import OutputError
from url_transcriber.formatter import sanitize_filename


DEFAULT_OUTPUT_DIRECTORY = Path("outputs")


def write_markdown(
    markdown: str,
    title: str,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    """Write Markdown to the next available title-based path and return it."""
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputError(
            f"Could not create output directory: {output_directory}"
        ) from error

    stem = sanitize_filename(title)
    sequence = 1

    while True:
        suffix = "" if sequence == 1 else f"-{sequence}"
        output_path = output_directory / f"{stem}{suffix}.md"

        try:
            file = output_path.open("x", encoding="utf-8", newline="\n")
        except FileExistsError:
            sequence += 1
            continue
        except OSError as error:
            raise OutputError(f"Could not write transcript: {output_path}") from error

        try:
            with file:
                file.write(markdown)
        except OSError as error:
            _remove_partial_file(output_path)
            raise OutputError(f"Could not write transcript: {output_path}") from error

        return output_path


def _remove_partial_file(path: Path) -> None:
    """Best-effort cleanup of a file created by a failed write."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
