from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def as_dict(self) -> dict[str, float]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
        }


@dataclass(slots=True)
class ProcessingConfig:
    output_dir: Path
    output_format: str | None = None
    subtitle_path: Path | None = None
    subtitle_padding_ms: int = 120
    merge_gap_ms: int = 220
    silence_threshold_db: int = -35
    min_silence_sec: float = 0.35


@dataclass(slots=True)
class PreviewData:
    source: str
    duration: float
    waveform: str
    timeline: str
    segments: list[Segment] = field(default_factory=list)


@dataclass(slots=True)
class JobResult:
    input_path: Path
    output_path: Path | None
    report_path: Path | None
    source: str
    original_duration: float
    condensed_duration: float
    segments: list[Segment]
    success: bool
    error: str | None = None
