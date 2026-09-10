"""Command-line adapter for URL Transcriber."""

import argparse
import sys
from collections.abc import Sequence

from url_transcriber.errors import URLTranscriberError
from url_transcriber.formatter import format_duration, format_transcript_source
from url_transcriber.pipeline import PipelineResult, run_url_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the selected command, and return a process exit code."""
    arguments = _build_parser().parse_args(argv)

    if arguments.command == "url":
        return _run_url(arguments.video_url)

    raise AssertionError(f"Unhandled command: {arguments.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Markdown transcript from one video URL.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    url_parser = subparsers.add_parser(
        "url",
        help="transcribe one video URL",
    )
    url_parser.add_argument("video_url", metavar="VIDEO_URL")
    return parser


def _run_url(url: str) -> int:
    print("Processing URL...")
    print()

    try:
        result = run_url_pipeline(url)
    except URLTranscriberError as error:
        _print_error(error)
        return 1

    _print_success(result)
    return 0


def _print_success(result: PipelineResult) -> None:
    print("Found:")
    print(result.metadata.title)
    print()
    print("Duration:")
    print(format_duration(result.metadata.duration_seconds))
    print()
    print("Saved:")
    print(result.output_path)
    print()
    print("Transcript source:")
    print(format_transcript_source(result.transcript.source))


def _print_error(error: URLTranscriberError) -> None:
    print("Error:", file=sys.stderr)
    print(error, file=sys.stderr)
