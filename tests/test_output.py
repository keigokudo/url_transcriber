"""Tests for collision-safe Markdown output."""

from pathlib import Path

import pytest

from url_transcriber.errors import OutputError
from url_transcriber.output import write_markdown


def test_write_markdown_creates_directory_and_writes_utf8(tmp_path: Path) -> None:
    output_directory = tmp_path / "nested" / "outputs"
    markdown = "# 日本語\n\nRésumé café.\n"

    output_path = write_markdown(markdown, "日本語 title", output_directory)

    assert output_path == output_directory / "日本語-title.md"
    assert output_path.read_text(encoding="utf-8") == markdown


def test_write_markdown_uses_deterministic_collision_suffixes(tmp_path: Path) -> None:
    output_directory = tmp_path / "outputs"

    first_path = write_markdown("first", "Example Video", output_directory)
    second_path = write_markdown("second", "Example Video", output_directory)
    third_path = write_markdown("third", "Example Video", output_directory)

    assert first_path.name == "example-video.md"
    assert second_path.name == "example-video-2.md"
    assert third_path.name == "example-video-3.md"
    assert first_path.read_text(encoding="utf-8") == "first"
    assert second_path.read_text(encoding="utf-8") == "second"
    assert third_path.read_text(encoding="utf-8") == "third"


def test_write_markdown_wraps_directory_creation_errors(tmp_path: Path) -> None:
    output_directory = tmp_path / "not-a-directory"
    output_directory.write_text("existing file", encoding="utf-8")

    with pytest.raises(OutputError) as error_info:
        write_markdown("content", "Title", output_directory)

    assert isinstance(error_info.value.__cause__, OSError)
