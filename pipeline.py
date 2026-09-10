"""Single-URL orchestration independent of command-line presentation."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from url_transcriber.audio import download_audio
from url_transcriber.errors import SubtitleError
from url_transcriber.extractor import extract_metadata
from url_transcriber.formatter import format_markdown
from url_transcriber.models import TranscriptResult, VideoMetadata
from url_transcriber.output import DEFAULT_OUTPUT_DIRECTORY, write_markdown
from url_transcriber.subtitles import process_subtitles
from url_transcriber.transcription import transcribe_audio


@dataclass(frozen=True)
class PipelineResult:
    """Successful pipeline output needed by non-core entry points."""

    output_path: Path
    metadata: VideoMetadata
    transcript: TranscriptResult


def run_url_pipeline(
    url: str,
    output_dir: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> PipelineResult:
    """Process one URL through subtitles or temporary audio transcription."""
    metadata = extract_metadata(url)

    try:
        transcript = process_subtitles(metadata)
    except SubtitleError:
        transcript = None

    if transcript is not None:
        markdown = format_markdown(metadata, transcript)
        output_path = write_markdown(markdown, metadata.title, output_dir)
        return PipelineResult(output_path, metadata, transcript)

    with TemporaryDirectory(prefix="url-transcriber-") as temp_name:
        audio_path = download_audio(url, Path(temp_name))
        transcript = transcribe_audio(audio_path)
        markdown = format_markdown(metadata, transcript)
        output_path = write_markdown(markdown, metadata.title, output_dir)
        return PipelineResult(output_path, metadata, transcript)
