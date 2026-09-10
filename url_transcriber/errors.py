"""Expected application errors exposed by URL Transcriber components."""


class URLTranscriberError(Exception):
    """Base class for expected URL Transcriber failures."""


class ExtractionError(URLTranscriberError):
    """Raised when video information cannot be retrieved."""


class SubtitleError(URLTranscriberError):
    """Raised when subtitles cannot be processed."""


class AudioDownloadError(URLTranscriberError):
    """Raised when temporary audio cannot be downloaded."""


class TranscriptionError(URLTranscriberError):
    """Raised when local transcription fails."""


class OutputError(URLTranscriberError):
    """Raised when the final transcript cannot be written."""
