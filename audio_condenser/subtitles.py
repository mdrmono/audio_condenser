from __future__ import annotations

import re
from pathlib import Path

from .models import Segment

TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


def auto_pair_subtitle(audio_path: Path) -> Path | None:
    candidate = audio_path.with_suffix(".srt")
    if candidate.exists():
        return candidate
    return None


def parse_srt(path: Path) -> list[Segment]:
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
    segments: list[Segment] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if TIMESTAMP_RE.match(lines[0]):
            time_line = lines[0]
        elif len(lines) >= 2 and TIMESTAMP_RE.match(lines[1]):
            time_line = lines[1]
        else:
            continue

        match = TIMESTAMP_RE.match(time_line)
        if not match:
            continue
        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        if end > start:
            segments.append(Segment(start=start, end=end))

    if not segments:
        raise ValueError(f"No subtitle timing entries found in {path}")
    return segments


def parse_timestamp(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def normalize_subtitle_segments(
    segments: list[Segment],
    duration: float,
    padding_ms: int,
    merge_gap_ms: int,
) -> list[Segment]:
    padding = max(0, padding_ms) / 1000.0
    merge_gap = max(0, merge_gap_ms) / 1000.0
    padded = [
        Segment(
            start=max(0.0, segment.start - padding),
            end=min(duration, segment.end + padding),
        )
        for segment in segments
    ]
    return merge_segments(padded, merge_gap=merge_gap, duration=duration)


def merge_segments(
    segments: list[Segment], merge_gap: float, duration: float | None = None
) -> list[Segment]:
    ordered = sorted((segment for segment in segments if segment.duration > 0), key=lambda item: item.start)
    if not ordered:
        return []

    merged: list[Segment] = [Segment(start=ordered[0].start, end=ordered[0].end)]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start - previous.end <= merge_gap:
            previous.end = max(previous.end, current.end)
        else:
            merged.append(Segment(start=current.start, end=current.end))

    if duration is not None:
        for segment in merged:
            segment.start = max(0.0, min(segment.start, duration))
            segment.end = max(0.0, min(segment.end, duration))
    return [segment for segment in merged if segment.duration > 0]
