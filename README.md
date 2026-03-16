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
condenser run input.mp3 --output-dir out --mode accurate --ffmpeg-threads 1
condenser tui
```

Performance notes:

- The TUI now defaults to fewer queue workers to avoid starting too many ffmpeg jobs at once.
- `accurate` re-encodes for precise cuts. `fast` stream-copies matching mp3, m4a, or wav outputs and is much faster, but cuts are less exact.
- Use `--ffmpeg-threads` or the TUI setting to cap CPU threads per ffmpeg process on slower machines.
