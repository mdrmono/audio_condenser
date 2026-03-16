from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.suggester import Suggester
from textual.widgets import Button, Footer, Header, Input, Label, ProgressBar, Static

from .core import (
    AUDIO_INPUT_SUFFIXES,
    CondenserError,
    build_preview,
    check_dependencies,
    process_file,
    resolve_subtitle_path,
)
from .models import JobResult, PreviewData, ProcessingConfig


@dataclass(slots=True)
class QueueItem:
    input_path: Path
    subtitle_override: Path | None = None
    status: str = "pending"
    message: str = ""


def split_completion_value(raw: str) -> tuple[Path | None, str, str]:
    if raw.endswith("/"):
        prefix = raw
        partial = ""
        directory_value = raw
    elif "/" in raw:
        parent, _, partial = raw.rpartition("/")
        prefix = f"{parent}/" if parent else "/"
        directory_value = prefix
    else:
        prefix = ""
        partial = raw
        directory_value = "."

    try:
        directory = Path(directory_value).expanduser()
    except RuntimeError:
        return None, prefix, partial
    return directory, prefix, partial


class PathSuggester(Suggester):
    def __init__(self, directories_only: bool = False) -> None:
        super().__init__(use_cache=False, case_sensitive=False)
        self.directories_only = directories_only

    async def get_suggestion(self, value: str) -> str | None:
        raw = value.strip()
        if not raw:
            return None
        if raw == "~":
            return "~/"
        if raw.startswith("~") and "/" not in raw:
            return None

        directory, prefix, partial = split_completion_value(raw)
        if directory is None or not directory.exists() or not directory.is_dir():
            return None

        try:
            entries = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            return None

        partial_lower = partial.lower()
        for entry in entries:
            if self.directories_only and not entry.is_dir():
                continue
            if partial and not entry.name.lower().startswith(partial_lower):
                continue
            suffix = "/" if entry.is_dir() else ""
            return f"{prefix}{entry.name}{suffix}"
        return None


class CondenserTUI(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
    }

    #queue-pane, #settings-pane {
        width: 1fr;
        padding: 1;
    }

    #queue-pane {
        min-width: 46;
    }

    #settings-pane {
        min-width: 56;
        overflow-y: auto;
    }

    Input {
        margin: 0 0 1 0;
    }

    .button-row {
        height: auto;
        margin: 0 0 1 0;
        width: 100%;
    }

    .button-row Button {
        width: 1fr;
        min-width: 14;
        height: 3;
        margin: 0 1 0 0;
    }

    .section-title {
        margin: 1 0 0 0;
        text-style: bold;
    }

    .hint {
        color: $text-muted;
        margin: -1 0 1 0;
    }

    #audio-preview {
        border: round $accent;
        height: 10;
        padding: 1;
        margin-bottom: 1;
        overflow-y: auto;
    }

    #queue-view, #preview, #log-view {
        border: round $accent;
        padding: 1;
    }

    #queue-view {
        height: 1fr;
    }

    #preview {
        height: 12;
        margin-top: 1;
    }

    #log-view {
        height: 8;
        margin-top: 1;
    }

    #status {
        margin-top: 1;
    }

    #queue-progress {
        margin: 0 0 1 0;
    }

    CommandPalette > Vertical {
        margin-top: 0;
        padding-top: 0;
    }

    CommandPalette #--results {
        overlay: none;
        height: auto;
        margin-top: -1;
        padding-top: 0;
    }

    CommandPalette CommandList {
        margin-top: 0;
        padding-top: 0;
    }

    CommandPalette Input {
        margin-bottom: -1;
    }

    CommandPalette CommandInput {
        margin-bottom: -1;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self.queue: list[QueueItem] = []
        self.selected_index = 0
        self.log_lines: list[str] = []
        self.running = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="queue-pane"):
                yield Label("Input", classes="section-title")
                yield Input(
                    placeholder="Audio file or directory path",
                    id="audio-path",
                    suggester=PathSuggester(),
                )
                yield Input(
                    placeholder="Optional .srt override for single-file adds",
                    id="srt-path",
                    suggester=PathSuggester(),
                )
                with Horizontal(id="queue-actions", classes="button-row"):
                    yield Button("Add File", id="add-file")
                    yield Button("Add Directory", id="add-directory")
                    yield Button("Remove", id="remove")
                    yield Button("Prev", id="prev")
                    yield Button("Next", id="next")
                yield Label("Audio path preview", classes="section-title")
                yield Static("\n".join(self._format_directory_contents(Path.cwd())), id="audio-preview")
                yield Label("Queue", classes="section-title")
                yield Static("Queue is empty.", id="queue-view")
            with Vertical(id="settings-pane"):
                yield Label("Settings", classes="section-title")
                with Horizontal(id="run-actions", classes="button-row"):
                    yield Button("Preview Input", id="preview-button")
                    yield Button("Run Queue", id="run-button")
                yield ProgressBar(total=1, id="queue-progress", show_eta=False)
                yield Label("Queue workers")
                yield Input(
                    value=str(self._default_worker_count()),
                    id="workers",
                    placeholder="Files to process in parallel",
                )
                yield Static("How many audio files to process at the same time. Higher values use more CPU.", classes="hint")
                yield Label("Output directory")
                yield Input(
                    value=str(Path.cwd()),
                    id="output-dir",
                    placeholder="Output directory",
                    suggester=PathSuggester(directories_only=True),
                )
                yield Static("Where condensed files and JSON reports are written.", classes="hint")
                yield Label("Subtitle directory")
                yield Input(
                    value="",
                    id="subtitle-dir",
                    placeholder="Directory of stem-matched .srt files",
                    suggester=PathSuggester(directories_only=True),
                )
                yield Static(
                    "If set, lesson01.m4a will first try subtitle-dir/lesson01.srt before same-folder auto-pairing.",
                    classes="hint",
                )
                yield Label("Export format")
                yield Input(value="auto", id="format", placeholder="auto, m4a, mp3, or wav")
                yield Static("Use auto to keep m4a or wav when possible and otherwise export mp3.", classes="hint")
                yield Label("Subtitle padding (ms)")
                yield Input(value="120", id="subtitle-padding", placeholder="Milliseconds added around each subtitle segment")
                yield Static("Adds a little context before and after each subtitle timestamp.", classes="hint")
                yield Label("Merge gap (ms)")
                yield Input(value="220", id="merge-gap", placeholder="Merge nearby speech segments")
                yield Static("Bridges short pauses so consecutive subtitle entries stay together.", classes="hint")
                yield Label("Fallback silence threshold (dB)")
                yield Input(value="-35", id="silence-threshold", placeholder="Silence detection threshold")
                yield Static("Used only when no usable .srt is available. Lower values keep more quiet audio.", classes="hint")
                yield Label("Fallback minimum silence (sec)")
                yield Input(value="0.35", id="min-silence", placeholder="Shortest silence to remove")
                yield Static("Used only for silence fallback. Higher values remove fewer short pauses.", classes="hint")
                yield Static("Preview the typed input or the selected queue item.", id="preview")
                yield Static("Status: idle", id="status")
                yield Static("", id="log-view")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_queue_view()
        self._refresh_audio_preview()
        self._reset_progress(0)
        self._set_status("idle")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "add-file":
            self._add_file_from_input()
        elif button_id == "add-directory":
            self._add_directory_from_input()
        elif button_id == "remove":
            self._remove_selected_item()
        elif button_id == "prev":
            self._move_selection(-1)
        elif button_id == "next":
            self._move_selection(1)
        elif button_id == "preview-button":
            self._preview_current_target()
        elif button_id == "run-button":
            self._run_queue()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in {"audio-path", "srt-path"}:
            self._refresh_audio_preview()
        elif event.input.id == "subtitle-dir":
            self._refresh_queue_view()

    async def on_key(self, event: events.Key) -> None:
        if event.key != "ctrl+tab":
            return
        focused = self.focused
        if not isinstance(focused, Input):
            return
        if focused.id not in {"audio-path", "srt-path", "output-dir", "subtitle-dir"}:
            return
        suggester = focused.suggester
        if not isinstance(suggester, PathSuggester):
            return

        suggestion = await suggester.get_suggestion(focused.value)
        if suggestion and suggestion != focused.value:
            focused.value = suggestion
            focused.cursor_position = len(suggestion)
            event.prevent_default()
            event.stop()

    def _current_config(self, subtitle_override: Path | None = None) -> ProcessingConfig:
        output_dir_value = self.query_one("#output-dir", Input).value.strip()
        output_dir = self._path_from_value(output_dir_value, strict=True) if output_dir_value else Path.cwd()
        output_format_value = self.query_one("#format", Input).value.strip().lower()
        output_format = None if output_format_value in {"", "auto"} else output_format_value
        return ProcessingConfig(
            output_dir=output_dir,
            output_format=output_format,
            subtitle_path=subtitle_override,
            subtitle_padding_ms=int(self.query_one("#subtitle-padding", Input).value.strip()),
            merge_gap_ms=int(self.query_one("#merge-gap", Input).value.strip()),
            silence_threshold_db=int(self.query_one("#silence-threshold", Input).value.strip()),
            min_silence_sec=float(self.query_one("#min-silence", Input).value.strip()),
        )

    @staticmethod
    def _default_worker_count() -> int:
        cpu_count = os.cpu_count() or 2
        return max(1, min(4, cpu_count // 2 or 1))

    def _current_worker_count(self) -> int:
        raw_value = self.query_one("#workers", Input).value.strip()
        try:
            worker_count = int(raw_value)
        except ValueError as exc:
            raise ValueError("Queue workers must be an integer.") from exc
        if worker_count < 1:
            raise ValueError("Queue workers must be at least 1.")
        return worker_count

    def _current_subtitle_directory(self, validate: bool = False) -> Path | None:
        subtitle_dir_value = self.query_one("#subtitle-dir", Input).value.strip()
        if not subtitle_dir_value:
            return None

        subtitle_dir = self._path_from_value(subtitle_dir_value, strict=validate)
        if validate and not subtitle_dir.exists():
            raise ValueError(f"Subtitle directory does not exist: {subtitle_dir}")
        if validate and not subtitle_dir.is_dir():
            raise ValueError(f"Subtitle directory is not a directory: {subtitle_dir}")
        return subtitle_dir

    def _processing_config_for_path(
        self, input_path: Path, manual_override: Path | None = None
    ) -> ProcessingConfig:
        subtitle_dir = self._current_subtitle_directory(validate=True)
        effective_subtitle = self._resolve_effective_subtitle(
            input_path, manual_override, subtitle_dir=subtitle_dir
        )
        return self._current_config(subtitle_override=effective_subtitle)

    def _typed_audio_path(self) -> Path | None:
        audio_value = self.query_one("#audio-path", Input).value.strip()
        if not audio_value:
            return None
        return self._path_from_value(audio_value, strict=False)

    def _typed_subtitle_override(self) -> Path | None:
        subtitle_value = self.query_one("#srt-path", Input).value.strip()
        if not subtitle_value:
            return None
        return self._path_from_value(subtitle_value, strict=False)

    def _path_from_value(self, value: str, strict: bool) -> Path:
        path = Path(value)
        try:
            return path.expanduser()
        except RuntimeError as exc:
            if strict:
                raise ValueError(f"Invalid path shorthand: {value}") from exc
            return path

    def _resolve_effective_subtitle(
        self,
        input_path: Path,
        manual_override: Path | None,
        subtitle_dir: Path | None = None,
    ) -> Path | None:
        if manual_override is not None:
            return manual_override

        if subtitle_dir is None:
            subtitle_dir = self._current_subtitle_directory(validate=False)
        if subtitle_dir is not None and subtitle_dir.exists() and subtitle_dir.is_dir():
            candidate = subtitle_dir / f"{input_path.stem}.srt"
            if candidate.exists():
                return candidate

        return resolve_subtitle_path(input_path, None)

    def _subtitle_status(self, input_path: Path, manual_override: Path | None) -> str:
        if manual_override is not None:
            return f"manual:{manual_override.name}"

        subtitle_dir = self._current_subtitle_directory(validate=False)
        if subtitle_dir is not None and subtitle_dir.exists() and subtitle_dir.is_dir():
            candidate = subtitle_dir / f"{input_path.stem}.srt"
            if candidate.exists():
                return f"subtitle-dir:{candidate.name}"

        paired = resolve_subtitle_path(input_path, None)
        if paired is not None:
            return f"auto:{paired.name}"
        return "fallback:silence"

    def _refresh_audio_preview(self) -> None:
        widget = self.query_one("#audio-preview", Static)
        input_path = self._typed_audio_path()
        raw_value = self.query_one("#audio-path", Input).value.strip()

        if not raw_value:
            widget.update("\n".join(self._format_directory_contents(Path.cwd())))
            return

        directory, _, partial = split_completion_value(raw_value)
        if directory is None or not directory.exists() or not directory.is_dir():
            if input_path is None:
                widget.update("(no completion preview)")
                return
            directory = self._preview_directory_for_path(input_path)
            target_name = input_path.name if not input_path.is_dir() else None
            widget.update(
                "\n".join(
                    self._format_directory_contents(
                        directory, selected_name=target_name, prefix_filter=partial or None
                    )
                )
            )
            return

        if input_path is not None and input_path.exists() and input_path.is_dir():
            widget.update("\n".join(self._format_directory_contents(input_path)))
            return

        selected_name = None
        prefix_filter = partial or None
        if input_path is not None and input_path.exists() and not input_path.is_dir():
            selected_name = input_path.name
            prefix_filter = None

        widget.update(
            "\n".join(
                self._format_directory_contents(
                    directory, selected_name=selected_name, prefix_filter=prefix_filter
                )
            )
        )

    def _preview_directory_for_path(self, input_path: Path) -> Path:
        candidate = input_path if input_path.is_dir() else input_path.parent
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate if candidate.exists() else Path.cwd()

    def _format_directory_contents(
        self,
        directory: Path,
        selected_name: str | None = None,
        prefix_filter: str | None = None,
    ) -> list[str]:
        try:
            entries = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError as exc:
            return [f"(unable to read directory: {exc})"]

        if prefix_filter:
            lowered = prefix_filter.lower()
            entries = [entry for entry in entries if entry.name.lower().startswith(lowered)]

        if not entries:
            return ["(no matches)"]

        lines: list[str] = []
        for entry in entries:
            name = f"{entry.name}/" if entry.is_dir() else entry.name
            marker = "> " if selected_name is not None and entry.name == selected_name else ""
            lines.append(f"{marker}{name}")
        return lines

    def _queue_contains(self, input_path: Path) -> bool:
        return any(item.input_path == input_path for item in self.queue)

    def _add_file_from_input(self) -> None:
        input_path = self._typed_audio_path()
        subtitle_override = self._typed_subtitle_override()
        if input_path is None:
            self._append_log("Audio path is required.")
            return
        if not input_path.exists():
            self._append_log(f"Missing audio file: {input_path}")
            return
        if input_path.is_dir():
            self._append_log("Use Add Directory for folders.")
            return
        if input_path.suffix.lower() not in AUDIO_INPUT_SUFFIXES:
            self._append_log(f"Unsupported audio format: {input_path.suffix or '(none)'}")
            return
        if subtitle_override is not None and not subtitle_override.exists():
            self._append_log(f"Missing subtitle file: {subtitle_override}")
            return
        if self._queue_contains(input_path):
            self._append_log(f"Already queued: {input_path.name}")
            return

        self.queue.append(QueueItem(input_path=input_path, subtitle_override=subtitle_override))
        self.selected_index = len(self.queue) - 1
        self.query_one("#audio-path", Input).value = ""
        self.query_one("#srt-path", Input).value = ""
        self._append_log(f"Queued {input_path.name}")
        self._refresh_audio_preview()
        self._refresh_queue_view()

    def _add_directory_from_input(self) -> None:
        directory = self._typed_audio_path()
        if directory is None:
            self._append_log("Directory path is required.")
            return
        if not directory.exists():
            self._append_log(f"Missing directory: {directory}")
            return
        if not directory.is_dir():
            self._append_log("Add Directory expects a directory path.")
            return

        subtitle_override = self._typed_subtitle_override()
        if subtitle_override is not None:
            self._append_log("Ignoring single-file .srt override while adding a directory.")

        audio_files = sorted(
            [
                entry
                for entry in directory.iterdir()
                if entry.is_file() and entry.suffix.lower() in AUDIO_INPUT_SUFFIXES
            ],
            key=lambda entry: entry.name.lower(),
        )
        if not audio_files:
            self._append_log(f"No supported audio files found in {directory}")
            return

        queued = 0
        skipped = 0
        for audio_file in audio_files:
            if self._queue_contains(audio_file):
                skipped += 1
                continue
            self.queue.append(QueueItem(input_path=audio_file))
            queued += 1

        if queued:
            self.selected_index = max(0, len(self.queue) - queued)
        self.query_one("#audio-path", Input).value = ""
        self.query_one("#srt-path", Input).value = ""
        self._append_log(
            f"Queued {queued} audio files from {directory}"
            + (f" ({skipped} skipped as duplicates)" if skipped else "")
        )
        self._refresh_audio_preview()
        self._refresh_queue_view()

    def _remove_selected_item(self) -> None:
        item = self._selected_item()
        if item is None:
            self._append_log("Queue is empty.")
            return
        removed = self.queue.pop(self.selected_index)
        self.selected_index = max(0, min(self.selected_index, len(self.queue) - 1))
        self._append_log(f"Removed {removed.input_path.name}")
        self._refresh_queue_view()

    def _move_selection(self, delta: int) -> None:
        if not self.queue:
            return
        self.selected_index = (self.selected_index + delta) % len(self.queue)
        self._refresh_queue_view()

    def _preview_current_target(self) -> None:
        input_path = self._typed_audio_path()
        manual_override = self._typed_subtitle_override()

        if input_path is None:
            item = self._selected_item()
            if item is None:
                self._append_log("Queue is empty.")
                return
            input_path = item.input_path
            manual_override = item.subtitle_override

        if not input_path.exists():
            self._append_log(f"Missing audio file: {input_path}")
            return
        if input_path.is_dir():
            self._append_log("Preview Input works on one audio file. Add the directory to queue to process it.")
            return
        if manual_override is not None and not manual_override.exists():
            self._append_log(f"Missing subtitle file: {manual_override}")
            return

        try:
            config = self._processing_config_for_path(input_path, manual_override)
        except ValueError as exc:
            self._append_log(f"Invalid settings: {exc}")
            return
        self._set_status(f"previewing {input_path.name}")
        self._build_preview_worker(input_path, config)

    def _run_queue(self) -> None:
        if self.running:
            self._append_log("Queue is already running.")
            return
        if not self.queue:
            self._append_log("Queue is empty.")
            return
        try:
            check_dependencies()
            worker_count = self._current_worker_count()
            configs = [
                self._processing_config_for_path(item.input_path, item.subtitle_override)
                for item in self.queue
            ]
        except (CondenserError, ValueError) as exc:
            self._append_log(str(exc))
            self._set_status("blocked")
            return
        self.running = True
        self._reset_progress(len(self.queue))
        self._process_queue_worker(list(zip(self.queue, configs)), worker_count)

    def _selected_item(self) -> QueueItem | None:
        if not self.queue:
            return None
        return self.queue[self.selected_index]

    def _refresh_queue_view(self) -> None:
        widget = self.query_one("#queue-view", Static)
        if not self.queue:
            widget.update("Queue is empty.")
            return

        lines = []
        for index, item in enumerate(self.queue):
            marker = ">" if index == self.selected_index else " "
            subtitle_text = self._subtitle_status(item.input_path, item.subtitle_override)
            line = f"{marker} [{item.status}] {item.input_path.name} | {subtitle_text}"
            if item.message:
                line += f" | {item.message}"
            lines.append(line)
        widget.update("\n".join(lines))

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(f"Status: {text}")

    def _reset_progress(self, total: int) -> None:
        progress = self.query_one("#queue-progress", ProgressBar)
        progress.update(total=max(1, total), progress=0)

    def _advance_progress(self) -> None:
        self.query_one("#queue-progress", ProgressBar).advance(1)

    def _append_log(self, text: str) -> None:
        self.log_lines.append(text)
        if len(self.log_lines) > 12:
            self.log_lines = self.log_lines[-12:]
        self.query_one("#log-view", Static).update("\n".join(self.log_lines))

    def _render_preview(self, input_path: Path, preview: PreviewData) -> None:
        segment_count = len(preview.segments)
        preview_text = "\n".join(
            [
                f"File: {input_path.name}",
                f"Source: {preview.source}",
                f"Duration: {preview.duration:.2f}s",
                f"Segments kept: {segment_count}",
                f"Waveform: {preview.waveform}",
                f"Keep map: {preview.timeline}",
            ]
        )
        self.query_one("#preview", Static).update(preview_text)
        self._set_status(f"preview ready for {input_path.name}")

    def _finish_job(self, index: int, result: JobResult) -> None:
        item = self.queue[index]
        if result.success:
            item.status = "done"
            item.message = f"{result.condensed_duration:.2f}s from {result.original_duration:.2f}s"
            self._append_log(
                f"{item.input_path.name} -> {result.output_path} ({result.source}, {len(result.segments)} segments)"
            )
        else:
            item.status = "failed"
            item.message = result.error or "unknown error"
            self._append_log(f"{item.input_path.name} failed: {item.message}")
        self._refresh_queue_view()

    @work(thread=True, exclusive=True)
    def _build_preview_worker(self, input_path: Path, config: ProcessingConfig) -> None:
        try:
            preview = build_preview(input_path, config)
        except Exception as exc:
            self.call_from_thread(self._append_log, f"Preview failed: {exc}")
            self.call_from_thread(self._set_status, "preview failed")
            return
        self.call_from_thread(self._render_preview, input_path, preview)

    @work(thread=True, exclusive=True)
    def _process_queue_worker(
        self, jobs: list[tuple[QueueItem, ProcessingConfig]], worker_count: int
    ) -> None:
        if not jobs:
            self.call_from_thread(self._set_status, "idle")
            self.running = False
            return

        max_workers = max(1, min(worker_count, len(jobs)))
        pending_index = 0
        futures: dict = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pending_index < len(jobs) or futures:
                while pending_index < len(jobs) and len(futures) < max_workers:
                    item, config = jobs[pending_index]
                    future = executor.submit(process_file, item.input_path, config)
                    futures[future] = pending_index
                    self.call_from_thread(self._mark_running, pending_index)
                    self.call_from_thread(
                        self._set_status,
                        f"processing {item.input_path.name} ({pending_index + 1}/{len(jobs)} started)",
                    )
                    pending_index += 1

                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    index = futures.pop(future)
                    item, _ = jobs[index]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = JobResult(
                            input_path=item.input_path,
                            output_path=None,
                            report_path=None,
                            source="error",
                            original_duration=0.0,
                            condensed_duration=0.0,
                            segments=[],
                            success=False,
                            error=str(exc),
                        )
                    self.call_from_thread(self._finish_job, index, result)
                    self.call_from_thread(self._advance_progress)

        self.call_from_thread(self._set_status, "idle")
        self.call_from_thread(self._append_log, "Queue finished.")
        self.running = False

    def _mark_running(self, index: int) -> None:
        self.queue[index].status = "running"
        self.queue[index].message = ""
        self._refresh_queue_view()
