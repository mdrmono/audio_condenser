# audio-condenser

Python CLI/TUI for condensing audio by extracting spoken segments from `.srt` subtitle timestamps, with silence detection as a fallback when subtitles are unavailable.

## Install

```bash
pip install -e .
```

System dependency:

- `ffmpeg`
- `ffprobe`

## CLI

```bash
condenser check
condenser run input.mp3 --output-dir out
condenser tui
```
