from __future__ import annotations

import json
import shutil
import subprocess
from array import array
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import JobResult, PreviewData, ProcessingConfig, Segment
from .subtitles import auto_pair_subtitle, merge_segments, normalize_subtitle_segments, parse_srt

AUDIO_INPUT_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
OUTPUT_FORMATS = {"mp3", "wav", "m4a"}
AMPLITUDE_CHARS = " .:-=+*#%@"


class CondenserError(RuntimeError):
    pass


def check_dependencies() -> dict[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    missing = [name for name, path in {"ffmpeg": ffmpeg, "ffprobe": ffprobe}.items() if not path]
    if missing:
        missing_text = ", ".join(missing)
        raise CondenserError(
            f"Missing required binaries: {missing_text}. Install ffmpeg so both commands are available."
        )
    return {"ffmpeg": ffmpeg or "", "ffprobe": ffprobe or ""}


def process_file(input_path: Path, config: ProcessingConfig) -> JobResult:
    duration = probe_duration(input_path)
    segments, source = plan_segments(input_path, duration, config)
    if not segments:
        raise CondenserError(f"No spoken segments found for {input_path}")

    output_format = choose_output_format(input_path, config.output_format)
    output_path = build_output_path(input_path, config.output_dir, output_format)
    render_condensed_audio(input_path, output_path, segments, output_format)

    condensed_duration = round(sum(segment.duration for segment in segments), 3)
    report_path = write_report(
        input_path=input_path,
        output_path=output_path,
        duration=duration,
        condensed_duration=condensed_duration,
        source=source,
        config=config,
        segments=segments,
    )

    return JobResult(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        source=source,
        original_duration=duration,
        condensed_duration=condensed_duration,
        segments=segments,
        success=True,
    )


def build_preview(input_path: Path, config: ProcessingConfig, columns: int = 72) -> PreviewData:
    duration = probe_duration(input_path)
    segments, source = plan_segments(input_path, duration, config)
    waveform = sample_waveform(input_path, columns=columns)
    timeline = render_timeline(duration, segments, columns=columns)
    return PreviewData(
        source=source,
        duration=duration,
        waveform=waveform,
        timeline=timeline,
        segments=segments,
    )


def probe_duration(input_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    completed = run_command(command)
    try:
        duration = float(completed.stdout.strip())
    except ValueError as exc:
        raise CondenserError(f"Unable to read duration for {input_path}") from exc
    return round(duration, 3)


def plan_segments(input_path: Path, duration: float, config: ProcessingConfig) -> tuple[list[Segment], str]:
    subtitle_path = resolve_subtitle_path(input_path, config.subtitle_path)
    if subtitle_path is not None:
        subtitle_segments = parse_srt(subtitle_path)
        normalized = normalize_subtitle_segments(
            subtitle_segments,
            duration=duration,
            padding_ms=config.subtitle_padding_ms,
            merge_gap_ms=config.merge_gap_ms,
        )
        if normalized:
            explicit = config.subtitle_path is not None
            source = f"srt:{'explicit' if explicit else 'auto'}"
            return normalized, source

    silence_segments = build_segments_from_silence(
        input_path=input_path,
        duration=duration,
        padding_ms=config.subtitle_padding_ms,
        merge_gap_ms=config.merge_gap_ms,
        silence_threshold_db=config.silence_threshold_db,
        min_silence_sec=config.min_silence_sec,
    )
    return silence_segments, "silence:fallback"


def resolve_subtitle_path(input_path: Path, explicit_subtitle: Path | None) -> Path | None:
    if explicit_subtitle is not None:
        return explicit_subtitle
    return auto_pair_subtitle(input_path)


def build_segments_from_silence(
    input_path: Path,
    duration: float,
    padding_ms: int,
    merge_gap_ms: int,
    silence_threshold_db: int,
    min_silence_sec: float,
) -> list[Segment]:
    silent_ranges = detect_silence(input_path, silence_threshold_db, min_silence_sec)
    if not silent_ranges:
        return [Segment(start=0.0, end=duration)]

    speech_segments: list[Segment] = []
    cursor = 0.0
    for silence_start, silence_end in silent_ranges:
        if silence_start > cursor:
            speech_segments.append(Segment(start=cursor, end=silence_start))
        cursor = max(cursor, silence_end)
    if cursor < duration:
        speech_segments.append(Segment(start=cursor, end=duration))

    padding = max(0, padding_ms) / 1000.0
    padded = [
        Segment(start=max(0.0, segment.start - padding), end=min(duration, segment.end + padding))
        for segment in speech_segments
        if segment.duration > 0
    ]
    merge_gap = max(0, merge_gap_ms) / 1000.0
    return merge_segments(padded, merge_gap=merge_gap, duration=duration)


def detect_silence(
    input_path: Path, silence_threshold_db: int, min_silence_sec: float
) -> list[tuple[float, float]]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(input_path),
        "-af",
        f"silencedetect=noise={silence_threshold_db}dB:d={min_silence_sec}",
        "-f",
        "null",
        "-",
    ]
    completed = run_command(command, check=False)

    silence_starts: list[float] = []
    silent_ranges: list[tuple[float, float]] = []
    for line in completed.stderr.splitlines():
        if "silence_start:" in line:
            value = line.split("silence_start:", 1)[1].strip()
            silence_starts.append(float(value))
        elif "silence_end:" in line:
            value = line.split("silence_end:", 1)[1].split("|", 1)[0].strip()
            if silence_starts:
                start = silence_starts.pop(0)
                silent_ranges.append((max(0.0, start), max(0.0, float(value))))
    return [(start, end) for start, end in silent_ranges if end > start]


def choose_output_format(input_path: Path, requested_format: str | None) -> str:
    if requested_format:
        selected = requested_format.strip().lower()
        if selected not in OUTPUT_FORMATS:
            raise CondenserError(f"Unsupported output format: {requested_format}")
        return selected

    suffix = input_path.suffix.lower().lstrip(".")
    if suffix in OUTPUT_FORMATS:
        return suffix
    return "mp3"


def build_output_path(input_path: Path, output_dir: Path, output_format: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}.condensed.{output_format}"


def render_condensed_audio(
    input_path: Path, output_path: Path, segments: list[Segment], output_format: str
) -> None:
    if len(segments) == 1:
        filter_complex = (
            f"[0:a]atrim=start={segments[0].start:.3f}:end={segments[0].end:.3f},"
            "asetpts=PTS-STARTPTS[out]"
        )
    else:
        trims = []
        labels = []
        for index, segment in enumerate(segments):
            trims.append(
                f"[0:a]atrim=start={segment.start:.3f}:end={segment.end:.3f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
            labels.append(f"[a{index}]")
        concat = "".join(labels) + f"concat=n={len(segments)}:v=0:a=1[out]"
        filter_complex = ";".join(trims + [concat])

    if output_format == "mp3":
        codec_args = ["-c:a", "libmp3lame"]
    elif output_format == "wav":
        codec_args = ["-c:a", "pcm_s16le"]
    else:
        codec_args = ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        *codec_args,
        str(output_path),
    ]
    run_command(command)


def write_report(
    input_path: Path,
    output_path: Path,
    duration: float,
    condensed_duration: float,
    source: str,
    config: ProcessingConfig,
    segments: list[Segment],
) -> Path:
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    reduction_percent = round(((duration - condensed_duration) / duration) * 100, 2) if duration else 0.0
    payload = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source": source,
        "original_duration": duration,
        "condensed_duration": condensed_duration,
        "reduction_percent": reduction_percent,
        "settings": {
            "output_format": output_path.suffix.lstrip("."),
            "subtitle_padding_ms": config.subtitle_padding_ms,
            "merge_gap_ms": config.merge_gap_ms,
            "silence_threshold_db": config.silence_threshold_db,
            "min_silence_sec": config.min_silence_sec,
        },
        "segments": [segment.as_dict() for segment in segments],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def sample_waveform(input_path: Path, columns: int = 72) -> str:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "2000",
        "-t",
        "60",
        "-f",
        "s16le",
        "-",
    ]
    completed = run_command(command, text=False)
    samples = array("h")
    samples.frombytes(completed.stdout)
    if not samples:
        return "." * columns

    bucket_size = max(1, len(samples) // columns)
    buckets: list[str] = []
    for index in range(columns):
        start = index * bucket_size
        end = min(len(samples), start + bucket_size)
        window = samples[start:end]
        if not window:
            buckets.append(" ")
            continue
        level = sum(abs(sample) for sample in window) / len(window)
        mapped = int((level / 32767.0) * (len(AMPLITUDE_CHARS) - 1))
        buckets.append(AMPLITUDE_CHARS[max(0, min(mapped, len(AMPLITUDE_CHARS) - 1))])
    return "".join(buckets)


def render_timeline(duration: float, segments: list[Segment], columns: int = 72) -> str:
    if duration <= 0:
        return "." * columns
    timeline = ["." for _ in range(columns)]
    for segment in segments:
        start_index = max(0, min(columns - 1, int((segment.start / duration) * columns)))
        end_index = max(start_index + 1, min(columns, int((segment.end / duration) * columns) or start_index + 1))
        for index in range(start_index, end_index):
            timeline[index] = "#"
    return "".join(timeline)


def run_command(
    command: list[str], check: bool = True, text: bool = True
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=text,
        check=False,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore") if isinstance(completed.stderr, bytes) else completed.stderr
        raise CondenserError(stderr.strip() or f"Command failed: {' '.join(command)}")
    return completed
