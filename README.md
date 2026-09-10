# URL Transcriber

URL Transcriber is a local command-line application that turns one public video
URL into a timestamped Markdown transcript. The resulting text can then be used
as reference material or supplied to ChatGPT or another tool for summarization,
question answering, and important-point extraction. URL Transcriber does not
perform AI summarization itself in v0.1.

## How it works

The processing mode is selected automatically:

```text
URL
→ retrieve metadata
→ 1. prefer manual subtitles
→ 2. otherwise use automatic captions
→ 3. otherwise download temporary audio and transcribe with Local Whisper
→ format Markdown
→ save under outputs/
```

The user does not choose between subtitles and Whisper. An expected failure
while retrieving or parsing a selected subtitle also triggers the Local Whisper
fallback.

## v0.1 scope and validated target

The accepted v0.1 target is a single public YouTube video URL on a Windows 11
command line. Other sites supported by yt-dlp may work, but they are not part of
the v0.1 acceptance guarantee.

The complete subtitle and Whisper paths were validated on:

- Windows 11, NT build 26200, Windows-native Python 3.14.7
- Ubuntu under WSL2, Python 3.12.3

The validated dependency versions were yt-dlp 2026.8.19, faster-whisper 1.2.1,
CTranslate2 4.8.2, and PyAV 18.1.0. These are validation results, not a claim
that every other platform or Python version works.

## Requirements

- Python
- An internet connection for YouTube access and the first Whisper model download
- Sufficient disk space for the reusable `small` Whisper model cache

The application requests the original audio-only container and faster-whisper
decodes it through PyAV. A separately installed system FFmpeg was not required
for the validated YouTube workflow. FFmpeg may still benefit formats or sites
outside the v0.1 acceptance target.

## Windows setup

Use a repository-local Windows virtual environment from PowerShell:

```powershell
py -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the CLI:

```powershell
python main.py --help
python main.py url --help
```

Do not reuse a WSL virtual environment from Windows. Native Windows and WSL
Python installations require separate environments because their executables
and binary dependency wheels differ.

## WSL2 development setup

From Ubuntu under WSL2:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Process one URL:

```powershell
python main.py url "https://www.youtube.com/watch?v=xxxxx"
```

Successful subtitle output resembles:

```text
Processing URL...

Found:
Example Video

Duration:
42:31

Saved:
outputs/example-video.md

Transcript source:
YouTube auto captions
```

When subtitles cannot be used, the final source is reported as `Local Whisper`
instead. The CLI currently reports truthful start and final information rather
than detailed intermediate progress. Expected application failures are printed
concisely to stderr and return exit code 1.

## Markdown output

Only the final transcript is saved persistently under `outputs/`:

```markdown
# Example Video

Source: https://www.youtube.com/watch?v=xxxxx

Channel: Example Channel

Duration: 42:31

Language: en

Transcript source: YouTube auto captions

## Transcript

[00:00]
Today we're going to discuss an example topic.

[00:18]
The first important point is introduced here.
```

Local Whisper transcripts additionally include `Whisper model: small`.

Titles become lowercase, Unicode-preserving, Windows-safe filenames:

```text
The Future of AI: What Happens Next?
→ the-future-of-ai-what-happens-next.md
```

Existing files are never silently overwritten. Repeated runs use deterministic
suffixes such as `example.md`, `example-2.md`, and `example-3.md`.

## Subtitle behavior

URL Transcriber preserves the original spoken language by default. It does not
use translated subtitles as a substitute for original-language captions. When
yt-dlp reports the video's language, subtitle selection uses this priority:

1. Original-language manual VTT subtitles
2. Original-language automatic VTT captions
3. Local Whisper transcription of the original audio

Identifiers such as `en`, `en-US`, `en-GB`, `ja`, `ja-JP`, and underscore
variants are matched by language family. If yt-dlp does not report a usable
video language, the existing English/Japanese subtitle fallback order is
retained rather than guessing a new original language.

Only VTT is parsed; TTML, JSON3, and SRV formats are not v0.1 inputs. The one
selected subtitle is fetched and parsed in memory rather than saved as a `.vtt`
file. Cleaning removes presentation markup, normalizes whitespace, and reduces
only clear adjacent duplication or caption overlap.

## Local Whisper fallback

When no supported subtitle succeeds, yt-dlp downloads an audio-only format into
a Python-managed temporary directory. faster-whisper then transcribes it with:

```text
model: small
device: CPU
compute type: int8
language: automatically detected
task: transcription (not translation)
```

The first Whisper run may download model files into the normal Hugging Face
cache. That reusable model cache is retained intentionally and is distinct from
source video or audio.

## Temporary media and privacy behavior

URL Transcriber does not maintain a media library:

- Video is not intentionally downloaded or stored.
- Audio-only fallback media is not permanently stored.
- Subtitle files are not permanently stored.
- yt-dlp fragments and partial files are constrained to the temporary scope.

The Core Pipeline owns `tempfile.TemporaryDirectory` through audio download,
transcription, Markdown formatting, and output writing. Python removes that
directory during normal completion and exception handling. This is not an
absolute guarantee after an operating-system crash or power loss.

Normal persistent application output is `outputs/*.md` only.

## Testing

With the relevant environment activated, run:

```text
pytest
```

The automated suite mocks network and machine-learning boundaries so it does
not contact YouTube, download models, or perform inference. v0.1 acceptance also
included separate real end-to-end tests for:

- Manual subtitle processing
- Audio-only fallback and real Local Whisper inference
- WSL2 and Windows-native execution
- Temporary-media cleanup after success and failure
- UTF-8 output, Windows-safe filenames, and collision suffixes

Live YouTube tests are deliberately not part of the routine automated suite
because upstream availability and behavior can change.

## Current limitations

v0.1 intentionally has:

- One URL per command
- Public YouTube URLs as the validated target
- English/Japanese subtitle selection only
- VTT subtitle parsing only
- CPU/int8 Local Whisper using the `small` model
- No persistent audio or video library
- No AI summary, transcript search, database, Web UI, HTTP API, or mobile app

YouTube extraction depends on yt-dlp and live YouTube behavior, which can change
independently of this project.

## Project structure

```text
main.py                         Minimal executable entry point
url_transcriber/
  cli.py                        Argument parsing and console presentation
  pipeline.py                   Single-URL orchestration and temp ownership
  extractor.py                  Metadata and subtitle availability via yt-dlp
  subtitles.py                  VTT selection, retrieval, parsing, and cleaning
  audio.py                      Audio-only download into caller temp scope
  transcription.py              CPU faster-whisper transcription
  formatter.py                  Timestamp, filename, and Markdown formatting
  output.py                     UTF-8 and collision-safe Markdown writing
  models.py                     Shared typed data models
  errors.py                     Expected application exception hierarchy
tests/                          Network/model-free automated tests
outputs/                        Ignored generated Markdown transcripts
```

## v0.1 acceptance status

The CLI syntax, metadata extraction, manual subtitle path, automatic-caption
logic, real Local Whisper fallback, timestamped Markdown, collision handling,
Windows-safe output, CPU execution, and normal/exception temporary cleanup have
been validated. WSL2 and Windows-native regression suites both pass.

## Future ideas — not implemented in v0.1

Possible later work includes `--lang`, `--model`, `--output`, and `--overwrite`
options; batch URLs; richer progress reporting; AI summaries; a Web API or UI;
and mobile/iPhone integration. None of these are available in v0.1.
