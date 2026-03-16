from __future__ import annotations

from pathlib import Path

import typer

from .core import AUDIO_INPUT_SUFFIXES, CondenserError, check_dependencies, process_file
from .models import ProcessingConfig
from .tui import CondenserTUI

app = typer.Typer(help="Condense audio using SRT-guided spoken segment extraction.")


@app.command()
def check() -> None:
    """Verify system dependencies and print runtime capabilities."""
    try:
        binaries = check_dependencies()
    except CondenserError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    typer.echo("Dependencies:")
    typer.echo(f"  ffmpeg:  {binaries['ffmpeg']}")
    typer.echo(f"  ffprobe: {binaries['ffprobe']}")
    typer.echo("Modes:")
    typer.echo("  SRT-first extraction with auto-pairing via matching .srt files")
    typer.echo("  Silence detection fallback when subtitles are missing")
    typer.echo("Formats:")
    typer.echo(f"  Input:  {', '.join(sorted(suffix.lstrip('.') for suffix in AUDIO_INPUT_SUFFIXES))}")
    typer.echo("  Output: m4a, mp3, wav")


@app.command()
def run(
    inputs: list[Path] = typer.Argument(..., help="One or more audio files to process."),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", help="Directory for condensed files."),
    format: str = typer.Option("auto", "--format", help="auto, m4a, mp3, or wav."),
    srt: Path | None = typer.Option(None, "--srt", help="Explicit subtitle file for a single input."),
    subtitle_padding: int = typer.Option(120, "--subtitle-padding", help="Padding around subtitle segments in milliseconds."),
    merge_gap: int = typer.Option(220, "--merge-gap", help="Merge adjacent segments separated by at most this many milliseconds."),
    silence_threshold: int = typer.Option(-35, "--silence-threshold", help="Fallback silence threshold in dB."),
    min_silence: float = typer.Option(0.35, "--min-silence", help="Fallback minimum silence duration in seconds."),
) -> None:
    """Process one or more audio files."""
    try:
        check_dependencies()
    except CondenserError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    if srt is not None and len(inputs) != 1:
        typer.echo("--srt can only be used with a single input file.", err=True)
        raise typer.Exit(code=2)

    failures = 0
    for index, input_path in enumerate(inputs):
        expanded_input = input_path.expanduser()
        if not expanded_input.exists():
            typer.echo(f"Missing input file: {expanded_input}", err=True)
            failures += 1
            continue

        config = ProcessingConfig(
            output_dir=output_dir.expanduser(),
            output_format=None if format.strip().lower() == "auto" else format.strip().lower(),
            subtitle_path=srt.expanduser() if srt is not None and index == 0 else None,
            subtitle_padding_ms=subtitle_padding,
            merge_gap_ms=merge_gap,
            silence_threshold_db=silence_threshold,
            min_silence_sec=min_silence,
        )

        try:
            result = process_file(expanded_input, config)
        except CondenserError as exc:
            typer.echo(f"{expanded_input.name}: {exc}", err=True)
            failures += 1
            continue

        reduction = 0.0
        if result.original_duration:
            reduction = ((result.original_duration - result.condensed_duration) / result.original_duration) * 100
        typer.echo(
            f"{expanded_input.name} -> {result.output_path} | "
            f"{result.source} | "
            f"{result.condensed_duration:.2f}s from {result.original_duration:.2f}s | "
            f"{reduction:.2f}% reduced"
        )
        typer.echo(f"  report: {result.report_path}")

    if failures:
        raise typer.Exit(code=1)


@app.command()
def tui() -> None:
    """Launch the interactive TUI."""
    CondenserTUI().run()
