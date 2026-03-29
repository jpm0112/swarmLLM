from __future__ import annotations

"""Live telemetry, file logging, and Rich dashboard support for swarm runs."""

import io
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO

try:  # pragma: no cover - platform-specific import
    import select
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    select = None
    termios = None
    tty = None

import psutil
from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from swarmllm.tracking.token_tracker import TokenUsage


DashboardRequest = Literal["auto", "plain", "tui"]
DashboardMode = Literal["plain", "tui"]
EventLevel = Literal["debug", "info", "warning", "error"]
DashboardView = Literal["overview", "agents", "processes", "detail"]
DashboardDetailKind = Literal["agent", "process"]


def resolve_dashboard_mode(requested: DashboardRequest, is_tty: bool) -> DashboardMode:
    """Resolve dashboard mode from user request and stdout capabilities."""
    if requested == "auto":
        return "tui" if is_tty else "plain"
    if requested == "tui" and not is_tty:
        return "plain"
    return requested


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: max(limit - 1, 0)]}..."


def _safe_json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(_safe_json_dump(payload), encoding="utf-8")
    os.replace(tmp_path, path)


@dataclass
class TelemetryEvent:
    """Serializable lifecycle event."""

    event_type: str
    timestamp: str
    level: EventLevel = "info"
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Current live state for a single worker agent."""

    agent_id: int
    iteration: int = 0
    mode: str = "explore"
    status: str = "queued"
    phase: str = "queued"
    endpoint_label: str = ""
    direction: str = ""
    elapsed_seconds: float = 0.0
    latest_score: float | None = None
    retry_count: int = 0
    failure_reason: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    llm_time_seconds: float = 0.0
    exec_time_seconds: float = 0.0
    last_model: str = ""
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_thinking_tokens: int = 0
    last_total_tokens: int = 0
    last_llm_duration_seconds: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_thinking_tokens: int = 0
    total_tokens: int = 0
    llm_call_count: int = 0
    last_instance_scores: dict[str, float] = field(default_factory=dict)
    last_instance_errors: dict[str, str] = field(default_factory=dict)
    last_approach: str = ""


@dataclass
class ProcessState:
    """Tracked process information for the dashboard."""

    pid: int
    label: str
    kind: str
    status: str = "running"
    role: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_utc_now_iso)
    age_seconds: float = 0.0
    cpu_percent: float = 0.0
    rss_mb: float = 0.0
    command: str = ""
    cwd: str = ""


@dataclass
class IterationState:
    """Live summary for a swarm iteration."""

    iteration: int
    successful: int = 0
    failed: int = 0
    best_score_this_iter: float | None = None
    best_score_overall: float | None = None
    wall_time_seconds: float = 0.0
    avg_llm_time_seconds: float = 0.0
    avg_exec_time_seconds: float = 0.0
    failure_counts: dict[str, int] = field(default_factory=dict)
    llm_total_seconds: float = 0.0
    llm_samples: int = 0
    exec_total_seconds: float = 0.0
    exec_samples: int = 0


@dataclass
class ThroughputSnapshot:
    """Aggregated throughput metrics."""

    rolling_tps: float = 0.0
    lifetime_tps: float = 0.0
    rolling_agent_tps: float = 0.0
    lifetime_agent_tps: float = 0.0
    rolling_coordinator_tps: float = 0.0
    lifetime_coordinator_tps: float = 0.0


@dataclass
class DashboardUIState:
    """Interactive TUI state such as active view and current selection."""

    active_view: DashboardView = "overview"
    detail_kind: DashboardDetailKind = "agent"
    selected_agent_id: int | None = None
    selected_process_pid: int | None = None
    last_key: str = ""


@dataclass
class LiveRunState:
    """Full live snapshot written to disk and rendered in the dashboard."""

    status: str = "starting"
    dashboard_mode: DashboardMode = "plain"
    started_at: str = field(default_factory=_utc_now_iso)
    ended_at: str | None = None
    elapsed_seconds: float = 0.0
    current_iteration: int = 0
    total_iterations: int = 0
    total_agents: int = 0
    current_stage: str = "starting"
    best_score: float | None = None
    best_baseline: float | None = None
    gap_vs_baseline_percent: float | None = None
    total_tokens: int = 0
    throughput: ThroughputSnapshot = field(default_factory=ThroughputSnapshot)
    agent_counts: dict[str, int] = field(
        default_factory=lambda: {"queued": 0, "running": 0, "done": 0, "failed": 0}
    )
    agents: dict[int, AgentState] = field(default_factory=dict)
    iterations: dict[int, IterationState] = field(default_factory=dict)
    processes: dict[int, ProcessState] = field(default_factory=dict)
    recent_events: list[TelemetryEvent] = field(default_factory=list)
    recent_logs: list[str] = field(default_factory=list)
    output_files: dict[str, str] = field(default_factory=dict)
    ui: DashboardUIState = field(default_factory=DashboardUIState)


class TelemetrySink(Protocol):
    """Typed interface exposed to the orchestration layer."""

    def emit_event(self, event_type: str, message: str = "", level: EventLevel = "info", **data: Any) -> None:
        """Record a generic telemetry event."""

    def queue_agent(
        self,
        agent_id: int,
        iteration: int,
        mode: str,
        direction: str,
        endpoint_label: str,
    ) -> None:
        """Track a queued agent."""

    def set_agent_phase(
        self,
        agent_id: int,
        iteration: int,
        phase: str,
        status: str,
        retry_count: int | None = None,
        endpoint_label: str | None = None,
        direction: str | None = None,
        mode: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """Update an agent's live phase."""

    def complete_agent(
        self,
        agent_id: int,
        iteration: int,
        success: bool,
        score: float | None,
        llm_time_seconds: float,
        exec_time_seconds: float,
        runtime_seconds: float,
        failure_reason: str | None = None,
    ) -> None:
        """Record completion for an agent run."""

    def record_llm_call(
        self,
        role: str,
        iteration: int,
        agent_id: int | None,
        model: str,
        duration_seconds: float,
        usage: TokenUsage | None,
        endpoint_label: str = "",
    ) -> None:
        """Record token/duration metrics for one completed LLM call."""

    def register_process(
        self,
        pid: int,
        label: str,
        kind: str,
        role: str = "",
        metadata: dict[str, Any] | None = None,
        command: str = "",
        cwd: str = "",
    ) -> None:
        """Track a live process."""

    def unregister_process(self, pid: int) -> None:
        """Remove a process from active tracking."""


class ThroughputTracker:
    """Rolling and lifetime tokens-per-second metrics from completed calls."""

    def __init__(self, window_seconds: float = 30.0):
        self.window_seconds = window_seconds
        self._calls: deque[dict[str, Any]] = deque()
        self._total_tokens = 0
        self._total_duration = 0.0
        self._role_tokens = {"agent": 0, "coordinator": 0}
        self._role_duration = {"agent": 0.0, "coordinator": 0.0}

    def record(
        self,
        role: str,
        total_tokens: int,
        duration_seconds: float,
        completed_at: float | None = None,
    ) -> None:
        if total_tokens <= 0 or duration_seconds <= 0:
            return
        ts = time.time() if completed_at is None else completed_at
        role_key = "coordinator" if role == "coordinator" else "agent"
        self._calls.append({
            "completed_at": ts,
            "role": role_key,
            "total_tokens": total_tokens,
            "duration_seconds": duration_seconds,
        })
        self._total_tokens += total_tokens
        self._total_duration += duration_seconds
        self._role_tokens[role_key] += total_tokens
        self._role_duration[role_key] += duration_seconds
        self._prune(ts)

    def _prune(self, now: float) -> None:
        threshold = now - self.window_seconds
        while self._calls and self._calls[0]["completed_at"] < threshold:
            self._calls.popleft()

    def snapshot(self, now: float | None = None) -> ThroughputSnapshot:
        current = time.time() if now is None else now
        self._prune(current)
        rolling_tokens = sum(item["total_tokens"] for item in self._calls)
        rolling_duration = sum(item["duration_seconds"] for item in self._calls)
        rolling_agent_tokens = sum(item["total_tokens"] for item in self._calls if item["role"] == "agent")
        rolling_agent_duration = sum(item["duration_seconds"] for item in self._calls if item["role"] == "agent")
        rolling_coord_tokens = sum(item["total_tokens"] for item in self._calls if item["role"] == "coordinator")
        rolling_coord_duration = sum(
            item["duration_seconds"] for item in self._calls if item["role"] == "coordinator"
        )
        return ThroughputSnapshot(
            rolling_tps=(rolling_tokens / rolling_duration) if rolling_duration else 0.0,
            lifetime_tps=(self._total_tokens / self._total_duration) if self._total_duration else 0.0,
            rolling_agent_tps=(rolling_agent_tokens / rolling_agent_duration) if rolling_agent_duration else 0.0,
            lifetime_agent_tps=(
                self._role_tokens["agent"] / self._role_duration["agent"]
                if self._role_duration["agent"]
                else 0.0
            ),
            rolling_coordinator_tps=(
                rolling_coord_tokens / rolling_coord_duration if rolling_coord_duration else 0.0
            ),
            lifetime_coordinator_tps=(
                self._role_tokens["coordinator"] / self._role_duration["coordinator"]
                if self._role_duration["coordinator"]
                else 0.0
            ),
        )


class StdoutMirror(io.TextIOBase):
    """Redirected stdout/stderr sink that mirrors lines into telemetry."""

    def __init__(self, telemetry: "RunTelemetry", stream_name: str):
        self._telemetry = telemetry
        self._stream_name = stream_name
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._telemetry.capture_log_line(line, self._stream_name)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._telemetry.capture_log_line(self._buffer, self._stream_name)
            self._buffer = ""

    def isatty(self) -> bool:
        return False


class DashboardRenderer:
    """Rich layout renderer for live run state."""

    def render(self, state: LiveRunState) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="summary", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["summary"].update(self._summary_panel(state))
        self._render_main(layout["main"], state)
        layout["footer"].update(self._footer_panel(state))
        return layout

    def _render_main(self, layout: Layout, state: LiveRunState) -> None:
        view = state.ui.active_view
        if view == "agents":
            layout.split_column(
                Layout(name="content", ratio=4),
                Layout(name="events", ratio=2),
            )
            layout["content"].split_row(Layout(name="table", ratio=3), Layout(name="detail", ratio=2))
            layout["table"].update(self._agent_panel(state, max_rows=26, compact=False))
            layout["detail"].update(self._detail_panel(state))
            layout["events"].update(self._events_panel(state))
            return
        if view == "processes":
            layout.split_column(
                Layout(name="content", ratio=4),
                Layout(name="events", ratio=2),
            )
            layout["content"].split_row(Layout(name="table", ratio=3), Layout(name="detail", ratio=2))
            layout["table"].update(self._process_panel(state, max_rows=18, compact=False))
            layout["detail"].update(self._detail_panel(state))
            layout["events"].update(self._events_panel(state))
            return
        if view == "detail":
            layout.split_row(Layout(name="detail", ratio=3), Layout(name="events", ratio=2))
            layout["detail"].update(self._detail_panel(state, expanded=True))
            layout["events"].update(self._events_panel(state))
            return

        layout.split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
        layout["left"].split_column(Layout(name="agents", ratio=3), Layout(name="iterations", ratio=2))
        layout["right"].split_column(Layout(name="processes", ratio=2), Layout(name="events", ratio=3))
        layout["agents"].update(self._agent_panel(state, max_rows=12, compact=True))
        layout["iterations"].update(self._iteration_panel(state))
        layout["processes"].update(self._process_panel(state, max_rows=8, compact=True))
        layout["events"].update(self._events_panel(state))

    def _summary_panel(self, state: LiveRunState) -> Panel:
        summary = Table.grid(expand=True)
        summary.add_column(justify="left")
        summary.add_column(justify="center")
        summary.add_column(justify="center")
        summary.add_column(justify="center")
        summary.add_column(justify="center")
        gap_text = "n/a"
        if state.gap_vs_baseline_percent is not None:
            gap = state.gap_vs_baseline_percent
            gap_text = f"{gap:+.1f}%"
        counts = state.agent_counts
        summary.add_row(
            f"[bold cyan]Iteration[/] {state.current_iteration}/{state.total_iterations}  "
            f"[bold white]{state.current_stage}[/]",
            (
                f"[bold green]Agents[/] queued={counts['queued']} "
                f"running={counts['running']} done={counts['done']} failed={counts['failed']}"
            ),
            f"[bold magenta]Best[/] {state.best_score if state.best_score is not None else 'n/a'}",
            f"[bold yellow]Gap[/] {gap_text}",
            (
                f"[bold blue]Tokens[/] {state.total_tokens:,}  "
                f"[bold blue]TPS[/] {state.throughput.rolling_tps:.1f} rolling / "
                f"{state.throughput.lifetime_tps:.1f} life"
            ),
        )
        summary.add_row(
            f"[bold white]Wall[/] {state.elapsed_seconds:.1f}s",
            f"[bold white]Agent TPS[/] {state.throughput.rolling_agent_tps:.1f} rolling",
            f"[bold white]Coord TPS[/] {state.throughput.rolling_coordinator_tps:.1f} rolling",
            f"[bold white]Baseline[/] {state.best_baseline if state.best_baseline is not None else 'n/a'}",
            f"[bold white]View[/] {state.ui.active_view}",
        )
        return Panel(summary, title="Swarm Monitor", border_style="cyan")

    def _agent_panel(self, state: LiveRunState, *, max_rows: int, compact: bool) -> Panel:
        table = Table(expand=True, box=box.SIMPLE_HEAVY)
        table.add_column("Agent", justify="right", width=5)
        table.add_column("Mode", width=8)
        table.add_column("Endpoint", width=16, overflow="fold")
        table.add_column("Phase", width=10)
        table.add_column("Elapsed", justify="right", width=7)
        table.add_column("Score", justify="right", width=10)
        table.add_column("Retry", justify="right", width=5)
        if not compact:
            table.add_column("LLM", justify="right", width=5)
            table.add_column("Tokens", justify="right", width=10)
            table.add_column("Last Call", justify="right", width=10)
        table.add_column("Direction / Failure", overflow="fold")

        visible_agents, start, end = self._windowed_agents(state, max_rows)
        for agent_id in visible_agents:
            agent = state.agents[agent_id]
            descriptor = agent.direction
            if agent.failure_reason:
                descriptor = f"{agent.direction}\n[red]{agent.failure_reason}[/red]"
            score = "n/a" if agent.latest_score is None else str(agent.latest_score)
            row = [
                str(agent.agent_id),
                agent.mode,
                _truncate(agent.endpoint_label or "n/a", 16),
                agent.phase,
                f"{agent.elapsed_seconds:.1f}s",
                score,
                str(agent.retry_count),
            ]
            if not compact:
                row.extend([
                    str(agent.llm_call_count),
                    f"{agent.total_tokens:,}",
                    f"{agent.last_total_tokens:,}" if agent.last_total_tokens else "n/a",
                ])
            row.append(_truncate(descriptor, 180 if compact else 140))
            row_style = "bold reverse" if agent.agent_id == state.ui.selected_agent_id else ""
            table.add_row(*row, style=row_style)

        if not state.agents:
            empty = ["-", "-", "-", "-", "-", "-", "-"]
            if not compact:
                empty.extend(["-", "-", "-"])
            empty.append("No agents yet")
            table.add_row(*empty)
        title = "Agents"
        if state.agents:
            title = f"Agents ({start + 1}-{end} of {len(state.agents)})"
        return Panel(table, title=title, border_style="green")

    def _iteration_panel(self, state: LiveRunState) -> Panel:
        table = Table(expand=True, box=box.SIMPLE)
        table.add_column("Iter", justify="right", width=4)
        table.add_column("Success", justify="right", width=7)
        table.add_column("Failed", justify="right", width=6)
        table.add_column("Best", justify="right", width=10)
        table.add_column("Overall", justify="right", width=10)
        table.add_column("Avg LLM", justify="right", width=8)
        table.add_column("Avg Exec", justify="right", width=8)
        table.add_column("Failures", overflow="fold")

        for iteration in sorted(state.iterations):
            item = state.iterations[iteration]
            failure_bits = ", ".join(f"{reason}: {count}" for reason, count in item.failure_counts.items()) or "-"
            table.add_row(
                str(iteration),
                str(item.successful),
                str(item.failed),
                str(item.best_score_this_iter) if item.best_score_this_iter is not None else "n/a",
                str(item.best_score_overall) if item.best_score_overall is not None else "n/a",
                f"{item.avg_llm_time_seconds:.1f}s",
                f"{item.avg_exec_time_seconds:.1f}s",
                _truncate(failure_bits, 80),
            )

        if not state.iterations:
            table.add_row("-", "-", "-", "-", "-", "-", "-", "No iterations yet")
        return Panel(table, title="Iterations", border_style="yellow")

    def _process_panel(self, state: LiveRunState, *, max_rows: int, compact: bool) -> Panel:
        table = Table(expand=True, box=box.SIMPLE)
        table.add_column("Label", overflow="fold")
        table.add_column("PID", justify="right", width=7)
        table.add_column("CPU%", justify="right", width=7)
        table.add_column("RSS", justify="right", width=8)
        table.add_column("Age", justify="right", width=7)
        table.add_column("Kind", width=10)
        table.add_column("Status", width=8)
        if not compact:
            table.add_column("Context", overflow="fold")

        visible, start, end = self._windowed_processes(state, max_rows)
        for proc in visible:
            context = self._format_process_context(proc)
            row = [
                _truncate(proc.label, 36),
                str(proc.pid),
                f"{proc.cpu_percent:.1f}",
                f"{proc.rss_mb:.1f}MB",
                f"{proc.age_seconds:.1f}s",
                proc.kind,
                proc.status,
            ]
            if not compact:
                row.append(_truncate(context, 56))
            row_style = "bold reverse" if proc.pid == state.ui.selected_process_pid else ""
            table.add_row(*row, style=row_style)

        if not state.processes:
            empty = ["-", "-", "-", "-", "-", "-", "-"]
            if not compact:
                empty.append("No active processes")
            else:
                empty[-1] = "No active processes"
            table.add_row(*empty)
        title = "Processes"
        if state.processes:
            title = f"Processes ({start + 1}-{end} of {len(state.processes)})"
        return Panel(table, title=title, border_style="magenta")

    def _events_panel(self, state: LiveRunState) -> Panel:
        lines: list[Text] = []
        for event in state.recent_events[-8:]:
            color = {"warning": "yellow", "error": "red"}.get(event.level, "cyan")
            stamp = event.timestamp[11:19]
            lines.append(Text.from_markup(f"[{color}]{stamp} {event.event_type}:[/] {event.message}"))

        if state.recent_logs:
            lines.append(Text(""))
            lines.append(Text("Recent log output", style="bold white"))
            for entry in state.recent_logs[-10:]:
                lines.append(Text(entry))

        if not lines:
            lines.append(Text("Waiting for telemetry..."))
        return Panel(Group(*lines), title="Events / Logs", border_style="blue")

    def _detail_panel(self, state: LiveRunState, *, expanded: bool = False) -> Panel:
        if state.ui.detail_kind == "process":
            return self._process_detail_panel(state, expanded=expanded)
        return self._agent_detail_panel(state, expanded=expanded)

    def _agent_detail_panel(self, state: LiveRunState, *, expanded: bool) -> Panel:
        agent = self._selected_agent(state)
        lines: list[Text] = []
        if agent is None:
            lines.append(Text("No agent selected. Use j/k in overview or agents view."))
            return Panel(Group(*lines), title="Agent Detail", border_style="green")

        lines.extend([
            Text(f"Agent {agent.agent_id}  iter={agent.iteration}  mode={agent.mode}  phase={agent.phase}"),
            Text(f"Endpoint: {agent.endpoint_label or 'n/a'}"),
            Text(
                "Elapsed: "
                f"{agent.elapsed_seconds:.1f}s  LLM: {agent.llm_time_seconds:.1f}s  "
                f"Exec: {agent.exec_time_seconds:.1f}s  Retry: {agent.retry_count}"
            ),
            Text(
                "Tokens: "
                f"{agent.total_tokens:,} total  "
                f"({agent.total_prompt_tokens:,} prompt + {agent.total_completion_tokens:,} completion"
                + (f", {agent.total_thinking_tokens:,} thinking" if agent.total_thinking_tokens else "")
                + ")"
            ),
            Text(
                "Last call: "
                f"{agent.last_total_tokens:,} tokens in {agent.last_llm_duration_seconds:.1f}s"
                if agent.last_total_tokens
                else "Last call: n/a"
            ),
        ])
        if agent.last_model:
            lines.append(Text(f"Model: {agent.last_model}"))
        if agent.latest_score is not None:
            lines.append(Text(f"Latest score: {agent.latest_score}"))
        if agent.last_approach:
            lines.append(Text(f"Approach: {_truncate(agent.last_approach, 160 if expanded else 90)}"))
        lines.append(Text(""))
        lines.append(Text("Direction", style="bold white"))
        lines.append(Text(agent.direction or "n/a"))
        if agent.failure_reason:
            lines.append(Text(""))
            lines.append(Text("Failure", style="bold white"))
            lines.append(Text(agent.failure_reason, style="red"))
        if agent.last_instance_scores or agent.last_instance_errors:
            lines.append(Text(""))
            lines.append(Text("Instance Results", style="bold white"))
            for name, score in sorted(agent.last_instance_scores.items()):
                lines.append(Text(f"{name}: {score}"))
            for name, error in sorted(agent.last_instance_errors.items()):
                lines.append(Text(f"{name}: {_truncate(error, 160 if expanded else 90)}", style="red"))

        related = self._related_process_lines(state, agent.agent_id)
        if related:
            lines.append(Text(""))
            lines.append(Text("Related Processes", style="bold white"))
            lines.extend(related)
        return Panel(Group(*lines), title=f"Agent Detail ({agent.agent_id})", border_style="green")

    def _process_detail_panel(self, state: LiveRunState, *, expanded: bool) -> Panel:
        proc = self._selected_process(state)
        lines: list[Text] = []
        if proc is None:
            lines.append(Text("No process selected. Switch to process view and use j/k."))
            return Panel(Group(*lines), title="Process Detail", border_style="magenta")

        lines.extend([
            Text(f"{proc.label}  pid={proc.pid}"),
            Text(f"Kind: {proc.kind}  Role: {proc.role or 'n/a'}  Status: {proc.status}"),
            Text(f"CPU: {proc.cpu_percent:.1f}%  RSS: {proc.rss_mb:.1f}MB  Age: {proc.age_seconds:.1f}s"),
        ])
        if proc.command:
            lines.append(Text(f"Command: {_truncate(proc.command, 220 if expanded else 120)}"))
        if proc.cwd:
            lines.append(Text(f"CWD: {proc.cwd}"))
        if proc.metadata:
            lines.append(Text(""))
            lines.append(Text("Metadata", style="bold white"))
            for key, value in sorted(proc.metadata.items()):
                lines.append(Text(f"{key}: {_truncate(str(value), 220 if expanded else 100)}"))
        agent_id = proc.metadata.get("agent_id") if proc.metadata else None
        if agent_id is not None and agent_id in state.agents:
            agent = state.agents[agent_id]
            lines.append(Text(""))
            lines.append(Text("Related Agent", style="bold white"))
            lines.append(Text(f"Agent {agent.agent_id}  phase={agent.phase}  tokens={agent.total_tokens:,}"))
            lines.append(Text(_truncate(agent.direction, 220 if expanded else 120)))
        return Panel(Group(*lines), title=f"Process Detail ({proc.pid})", border_style="magenta")

    def _footer_panel(self, state: LiveRunState) -> Panel:
        selected = (
            f"agent={state.ui.selected_agent_id if state.ui.selected_agent_id is not None else '-'}  "
            f"process={state.ui.selected_process_pid if state.ui.selected_process_pid is not None else '-'}"
        )
        footer = Table.grid(expand=True)
        footer.add_column(ratio=4)
        footer.add_column(justify="right", ratio=2)
        footer.add_row(
            "Views: o overview  a agents  p processes  d detail  tab cycle  "
            "j/k move  g/G first/last  [ / ] detail target",
            f"Detail={state.ui.detail_kind}  Selected {selected}",
        )
        return Panel(footer, border_style="cyan")

    def _windowed_agents(self, state: LiveRunState, max_rows: int) -> tuple[list[int], int, int]:
        agent_ids = sorted(state.agents)
        return self._window_values(agent_ids, state.ui.selected_agent_id, max_rows)

    def _windowed_processes(self, state: LiveRunState, max_rows: int) -> tuple[list[ProcessState], int, int]:
        processes = sorted(state.processes.values(), key=lambda item: (item.kind, item.label, item.pid))
        selected_pid = state.ui.selected_process_pid
        values, start, end = self._window_values(processes, selected_pid, max_rows, key=lambda item: item.pid)
        return values, start, end

    def _window_values(
        self,
        values: list[Any],
        selected_value: Any,
        max_rows: int,
        *,
        key=None,
    ) -> tuple[list[Any], int, int]:
        if not values:
            return [], 0, 0
        key = key or (lambda value: value)
        try:
            selected_index = next(idx for idx, value in enumerate(values) if key(value) == selected_value)
        except StopIteration:
            selected_index = 0
        max_rows = max(1, max_rows)
        start = max(0, min(len(values) - max_rows, selected_index - max_rows // 2))
        end = min(len(values), start + max_rows)
        return values[start:end], start, end

    def _selected_agent(self, state: LiveRunState) -> AgentState | None:
        if not state.agents:
            return None
        if state.ui.selected_agent_id in state.agents:
            return state.agents[state.ui.selected_agent_id]
        return state.agents[sorted(state.agents)[0]]

    def _selected_process(self, state: LiveRunState) -> ProcessState | None:
        if not state.processes:
            return None
        if state.ui.selected_process_pid in state.processes:
            return state.processes[state.ui.selected_process_pid]
        pid = sorted(state.processes)[0]
        return state.processes[pid]

    def _related_process_lines(self, state: LiveRunState, agent_id: int) -> list[Text]:
        lines: list[Text] = []
        for proc in sorted(state.processes.values(), key=lambda item: item.pid):
            if proc.metadata.get("agent_id") == agent_id:
                lines.append(
                    Text(
                        f"{proc.pid} {proc.label}  cpu={proc.cpu_percent:.1f}%  "
                        f"rss={proc.rss_mb:.1f}MB  {self._format_process_context(proc)}"
                    )
                )
        return lines

    def _format_process_context(self, proc: ProcessState) -> str:
        pieces: list[str] = []
        for key in ["agent_id", "iteration", "instance", "stage", "package", "kind"]:
            if key in proc.metadata:
                pieces.append(f"{key}={proc.metadata[key]}")
        if "log_path" in proc.metadata:
            pieces.append(f"log={proc.metadata['log_path']}")
        return ", ".join(pieces) if pieces else "n/a"


class RunTelemetry:
    """Thread-safe run telemetry with file sinks and optional Rich dashboard."""

    def __init__(
        self,
        output_dir: str,
        requested_mode: DashboardRequest = "auto",
        *,
        is_tty: bool | None = None,
        start_thread: bool = True,
        refresh_interval_seconds: float = 0.5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.requested_mode = requested_mode
        self.is_tty = sys.stdout.isatty() if is_tty is None else is_tty
        self.mode = resolve_dashboard_mode(requested_mode, self.is_tty)
        self.refresh_interval_seconds = refresh_interval_seconds
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._renderer = DashboardRenderer()
        self._dashboard_failed = False
        self._real_stdout: TextIO = sys.stdout
        self._real_stderr: TextIO = sys.stderr
        self._real_stdin: TextIO = sys.stdin
        self._console = Console(file=self._real_stdout, force_terminal=self.is_tty) if self.mode == "tui" else None
        self._live: Live | None = None
        self._input_thread: threading.Thread | None = None
        self._input_enabled = False
        self._stdin_fd: int | None = None
        self._stdin_attrs: list[Any] | None = None
        self._echo_output = self.mode == "plain"
        self._run_log_handle = (self.output_dir / "run.log").open("w", encoding="utf-8", buffering=1)
        self._events_path = self.output_dir / "events.jsonl"
        self._live_state_path = self.output_dir / "live_state.json"
        self._events_handle = self._events_path.open("a", encoding="utf-8", buffering=1)
        self._throughput = ThroughputTracker()
        self._recent_logs: deque[str] = deque(maxlen=200)
        self._recent_events: deque[TelemetryEvent] = deque(maxlen=64)
        self._process_handles: dict[int, psutil.Process] = {}
        self.state = LiveRunState(
            dashboard_mode=self.mode,
            output_files={
                "events_jsonl": str(self._events_path),
                "live_state_json": str(self._live_state_path),
                "run_log": str(self.output_dir / "run.log"),
            },
        )
        self.state.processes[os.getpid()] = ProcessState(
            pid=os.getpid(),
            label="swarmllm main",
            kind="python",
            role="main",
            metadata={"path": sys.executable},
        )
        self._process_handles[os.getpid()] = psutil.Process(os.getpid())
        self._prime_cpu_counter(os.getpid())
        self._ensure_selections_locked()
        self._init_dashboard()
        self.refresh_now()
        if start_thread:
            self._thread = threading.Thread(target=self._refresh_loop, name="swarmllm-telemetry", daemon=True)
            self._thread.start()

    def _init_dashboard(self) -> None:
        if self.mode != "tui" or self._dashboard_failed:
            return
        try:
            assert self._console is not None
            self._live = Live(
                self._renderer.render(self.state),
                console=self._console,
                screen=True,
                auto_refresh=False,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start()
            self._start_input_listener()
        except Exception as exc:  # pragma: no cover - defensive fail-open path
            self._dashboard_failed = True
            self._echo_output = True
            self.mode = "plain"
            self.state.dashboard_mode = "plain"
            self._real_stderr.write(f"Dashboard disabled: {exc}\n")
            self._real_stderr.flush()

    def stdout_mirror(self) -> StdoutMirror:
        return StdoutMirror(self, "stdout")

    def stderr_mirror(self) -> StdoutMirror:
        return StdoutMirror(self, "stderr")

    def set_run_metadata(self, *, total_iterations: int, total_agents: int, current_stage: str) -> None:
        with self._lock:
            self.state.total_iterations = total_iterations
            self.state.total_agents = total_agents
            self.state.current_stage = current_stage
        self.refresh_now()

    def set_baseline(self, best_baseline: float) -> None:
        with self._lock:
            self.state.best_baseline = best_baseline
            self._recompute_gap_locked()
        self.refresh_now()

    def set_best_score(self, best_score: float | None) -> None:
        with self._lock:
            self.state.best_score = best_score
            self._recompute_gap_locked()
        self.refresh_now()

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self.state.current_stage = stage
        self.refresh_now()

    def start_iteration(self, iteration: int) -> None:
        with self._lock:
            self.state.current_iteration = iteration
            self.state.iterations.setdefault(iteration, IterationState(iteration=iteration))
            self.state.current_stage = f"iteration {iteration}"
        self.refresh_now()

    def finish_iteration(
        self,
        iteration: int,
        *,
        successful: int,
        failed: int,
        best_score_this_iter: float | None,
        best_score_overall: float | None,
        wall_time_seconds: float,
        avg_llm_time_seconds: float,
        avg_exec_time_seconds: float,
        failure_counts: dict[str, int],
    ) -> None:
        with self._lock:
            item = self.state.iterations.setdefault(iteration, IterationState(iteration=iteration))
            item.successful = successful
            item.failed = failed
            item.best_score_this_iter = best_score_this_iter
            item.best_score_overall = best_score_overall
            item.wall_time_seconds = wall_time_seconds
            item.avg_llm_time_seconds = avg_llm_time_seconds
            item.avg_exec_time_seconds = avg_exec_time_seconds
            item.failure_counts = dict(failure_counts)
        self.refresh_now()

    def emit_event(self, event_type: str, message: str = "", level: EventLevel = "info", **data: Any) -> None:
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=_utc_now_iso(),
            level=level,
            message=message,
            data=data,
        )
        with self._lock:
            self._recent_events.append(event)
            self.state.recent_events = list(self._recent_events)
            self._events_handle.write(_safe_json_dump(asdict(event)))
            self._events_handle.write("\n")
        self.refresh_now()

    def capture_log_line(self, line: str, stream_name: str = "stdout") -> None:
        clean = line.rstrip("\r")
        with self._lock:
            self._recent_logs.append(clean)
            self.state.recent_logs = list(self._recent_logs)[-50:]
            self._run_log_handle.write(f"{clean}\n")
            if self._echo_output:
                target = self._real_stderr if stream_name == "stderr" else self._real_stdout
                target.write(f"{clean}\n")
                target.flush()
        self.refresh_now()

    def queue_agent(
        self,
        agent_id: int,
        iteration: int,
        mode: str,
        direction: str,
        endpoint_label: str,
    ) -> None:
        with self._lock:
            state = self.state.agents.setdefault(agent_id, AgentState(agent_id=agent_id))
            state.iteration = iteration
            state.mode = mode
            state.status = "queued"
            state.phase = "queued"
            state.direction = direction
            state.endpoint_label = endpoint_label
            state.failure_reason = ""
            state.latest_score = None
            state.retry_count = 0
            state.elapsed_seconds = 0.0
            state.started_at = None
            state.completed_at = None
            state.last_instance_scores = {}
            state.last_instance_errors = {}
            state.last_approach = ""
            self.state.current_iteration = iteration
            self.state.iterations.setdefault(iteration, IterationState(iteration=iteration))
            self._recompute_agent_counts_locked()
            self._ensure_selections_locked()
        self.emit_event(
            "agent_queued",
            message=f"Agent {agent_id} queued on {endpoint_label}",
            agent_id=agent_id,
            iteration=iteration,
            mode=mode,
            endpoint_label=endpoint_label,
            direction=direction,
        )

    def set_agent_phase(
        self,
        agent_id: int,
        iteration: int,
        phase: str,
        status: str,
        retry_count: int | None = None,
        endpoint_label: str | None = None,
        direction: str | None = None,
        mode: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        with self._lock:
            state = self.state.agents.setdefault(agent_id, AgentState(agent_id=agent_id))
            state.iteration = iteration
            state.phase = phase
            state.status = status
            if retry_count is not None:
                state.retry_count = retry_count
            if endpoint_label is not None:
                state.endpoint_label = endpoint_label
            if direction is not None:
                state.direction = direction
            if mode is not None:
                state.mode = mode
            if failure_reason is not None:
                state.failure_reason = failure_reason
            if state.started_at is None and status == "running":
                state.started_at = _utc_now_iso()
            self._recompute_agent_counts_locked()
            self._ensure_selections_locked()
        self.refresh_now()

    def complete_agent(
        self,
        agent_id: int,
        iteration: int,
        success: bool,
        score: float | None,
        llm_time_seconds: float,
        exec_time_seconds: float,
        runtime_seconds: float,
        failure_reason: str | None = None,
        instance_scores: dict[str, float] | None = None,
        instance_errors: dict[str, str] | None = None,
        approach: str | None = None,
    ) -> None:
        with self._lock:
            state = self.state.agents.setdefault(agent_id, AgentState(agent_id=agent_id))
            state.iteration = iteration
            state.status = "done" if success else "failed"
            state.phase = "done" if success else "failed"
            state.latest_score = score
            state.elapsed_seconds = runtime_seconds
            state.llm_time_seconds = llm_time_seconds
            state.exec_time_seconds = exec_time_seconds
            state.failure_reason = failure_reason or ""
            state.completed_at = _utc_now_iso()
            state.last_instance_scores = dict(instance_scores or {})
            state.last_instance_errors = dict(instance_errors or {})
            state.last_approach = approach or state.last_approach
            item = self.state.iterations.setdefault(iteration, IterationState(iteration=iteration))
            if success:
                item.successful += 1
                if score is not None and (item.best_score_this_iter is None or score < item.best_score_this_iter):
                    item.best_score_this_iter = score
            else:
                item.failed += 1
                if failure_reason:
                    item.failure_counts[failure_reason] = item.failure_counts.get(failure_reason, 0) + 1
            if llm_time_seconds > 0:
                item.llm_total_seconds += llm_time_seconds
                item.llm_samples += 1
                item.avg_llm_time_seconds = item.llm_total_seconds / item.llm_samples
            if exec_time_seconds > 0:
                item.exec_total_seconds += exec_time_seconds
                item.exec_samples += 1
                item.avg_exec_time_seconds = item.exec_total_seconds / item.exec_samples
            if self.state.best_score is not None:
                item.best_score_overall = self.state.best_score
            self._recompute_agent_counts_locked()
            self._ensure_selections_locked()
        self.refresh_now()

    def record_new_best(self, iteration: int, agent_id: int, score: float, approach: str) -> None:
        with self._lock:
            self.state.best_score = score
            item = self.state.iterations.setdefault(iteration, IterationState(iteration=iteration))
            item.best_score_overall = score
            self._recompute_gap_locked()
        self.emit_event(
            "new_best_score",
            message=f"Agent {agent_id} reached {score}",
            iteration=iteration,
            agent_id=agent_id,
            score=score,
            approach=_truncate(approach, 140),
        )

    def record_llm_call(
        self,
        role: str,
        iteration: int,
        agent_id: int | None,
        model: str,
        duration_seconds: float,
        usage: TokenUsage | None,
        endpoint_label: str = "",
    ) -> None:
        total_tokens = usage.total_tokens if usage is not None else 0
        with self._lock:
            self._throughput.record(role, total_tokens, duration_seconds)
            self.state.total_tokens += total_tokens
            self.state.throughput = self._throughput.snapshot()
            if role != "coordinator" and agent_id is not None:
                agent = self.state.agents.setdefault(agent_id, AgentState(agent_id=agent_id))
                agent.last_model = model
                agent.last_llm_duration_seconds = duration_seconds
                agent.llm_call_count += 1
                if usage is not None:
                    agent.last_prompt_tokens = usage.prompt_tokens
                    agent.last_completion_tokens = usage.completion_tokens
                    agent.last_thinking_tokens = usage.thinking_tokens
                    agent.last_total_tokens = usage.total_tokens
                    agent.total_prompt_tokens += usage.prompt_tokens
                    agent.total_completion_tokens += usage.completion_tokens
                    agent.total_thinking_tokens += usage.thinking_tokens
                    agent.total_tokens += usage.total_tokens
        self.emit_event(
            "llm_call_completed",
            message=f"{role} call completed in {duration_seconds:.1f}s",
            role=role,
            iteration=iteration,
            agent_id=agent_id,
            model=model,
            endpoint_label=endpoint_label,
            duration_seconds=duration_seconds,
            total_tokens=total_tokens,
            prompt_tokens=usage.prompt_tokens if usage is not None else 0,
            completion_tokens=usage.completion_tokens if usage is not None else 0,
        )

    def register_process(
        self,
        pid: int,
        label: str,
        kind: str,
        role: str = "",
        metadata: dict[str, Any] | None = None,
        command: str = "",
        cwd: str = "",
    ) -> None:
        try:
            process = psutil.Process(pid)
            self._prime_cpu_counter(pid, process=process)
        except psutil.Error:
            return

        with self._lock:
            self._process_handles[pid] = process
            self.state.processes[pid] = ProcessState(
                pid=pid,
                label=label,
                kind=kind,
                role=role,
                metadata=metadata or {},
                command=command,
                cwd=cwd,
            )
            self._ensure_selections_locked()
        self.refresh_now()

    def unregister_process(self, pid: int) -> None:
        with self._lock:
            self._process_handles.pop(pid, None)
            if pid in self.state.processes and pid != os.getpid():
                self.state.processes.pop(pid, None)
            self._ensure_selections_locked()
        self.refresh_now()

    def refresh_now(self) -> None:
        with self._lock:
            self._update_process_metrics_locked()
            self.state.elapsed_seconds = max(0.0, time.time() - self._started_at_epoch())
            self.state.recent_events = list(self._recent_events)[-20:]
            self.state.recent_logs = list(self._recent_logs)[-50:]
            self.state.throughput = self._throughput.snapshot()
            _atomic_write_json(self._live_state_path, self._serialize_state_locked())
            if self._live is not None and not self._dashboard_failed:
                try:
                    self._live.update(self._renderer.render(self.state), refresh=True)
                except Exception as exc:  # pragma: no cover - defensive fail-open path
                    self._disable_dashboard_locked(exc)

    def close(self, status: str = "completed") -> None:
        self.emit_event("run_finished", message=f"Run {status}", status=status)
        with self._lock:
            self.state.status = status
            self.state.ended_at = _utc_now_iso()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.refresh_interval_seconds * 4, 1.0))
        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)
        self._restore_terminal()
        self.refresh_now()
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
        self._events_handle.close()
        self._run_log_handle.close()

    def _disable_dashboard_locked(self, exc: Exception) -> None:
        self._dashboard_failed = True
        self._echo_output = True
        self.mode = "plain"
        self.state.dashboard_mode = "plain"
        self._input_enabled = False
        self._restore_terminal()
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        self._real_stderr.write(f"Dashboard disabled: {exc}\n")
        self._real_stderr.flush()

    def handle_input_key(self, key: str) -> None:
        """Handle a single dashboard navigation key."""
        if not key:
            return
        with self._lock:
            ui = self.state.ui
            normalized = self._normalize_key(key)
            if not normalized:
                return
            ui.last_key = normalized
            if normalized in {"tab"}:
                views: list[DashboardView] = ["overview", "agents", "processes", "detail"]
                ui.active_view = views[(views.index(ui.active_view) + 1) % len(views)]
                if ui.active_view == "agents":
                    ui.detail_kind = "agent"
                elif ui.active_view == "processes":
                    ui.detail_kind = "process"
            elif normalized in {"1", "o"}:
                ui.active_view = "overview"
            elif normalized in {"2", "a"}:
                ui.active_view = "agents"
                ui.detail_kind = "agent"
            elif normalized in {"3", "p"}:
                ui.active_view = "processes"
                ui.detail_kind = "process"
            elif normalized in {"4", "d"}:
                ui.active_view = "detail"
            elif normalized in {"[", "]"}:
                ui.detail_kind = "process" if normalized == "]" else "agent"
            elif normalized in {"j", "down"}:
                self._move_selection_locked(1)
            elif normalized in {"k", "up"}:
                self._move_selection_locked(-1)
            elif normalized == "g":
                self._jump_selection_locked(first=True)
            elif normalized == "G":
                self._jump_selection_locked(first=False)
            self._ensure_selections_locked()
        self.refresh_now()

    def _refresh_loop(self) -> None:
        while not self._stop_event.wait(self.refresh_interval_seconds):
            self.refresh_now()

    def _start_input_listener(self) -> None:
        if self.mode != "tui" or not self.is_tty or not self._real_stdin.isatty():
            return
        self._input_enabled = True
        if os.name != "nt" and termios is not None and tty is not None:
            try:
                self._stdin_fd = self._real_stdin.fileno()
                self._stdin_attrs = termios.tcgetattr(self._stdin_fd)
                tty.setcbreak(self._stdin_fd)
            except Exception:  # pragma: no cover - terminal-specific
                self._stdin_fd = None
                self._stdin_attrs = None
        self._input_thread = threading.Thread(target=self._input_loop, name="swarmllm-input", daemon=True)
        self._input_thread.start()

    def _restore_terminal(self) -> None:
        if self._stdin_fd is None or self._stdin_attrs is None or termios is None:
            return
        try:  # pragma: no cover - terminal-specific
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_attrs)
        except Exception:
            pass
        finally:
            self._stdin_fd = None
            self._stdin_attrs = None

    def _input_loop(self) -> None:
        while not self._stop_event.is_set() and self._input_enabled:
            key = self._read_key()
            if key:
                self.handle_input_key(key)

    def _read_key(self) -> str:
        if os.name == "nt":  # pragma: no cover - Windows-only branch
            try:
                import msvcrt
                import ctypes
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            except (ImportError, AttributeError):
                return ""
            # Use Win32 WaitForSingleObject on stdin handle for reliable
            # key detection — msvcrt.kbhit() can miss keys when Rich is
            # writing to the alternate screen buffer.
            STD_INPUT_HANDLE = -10
            WAIT_OBJECT_0 = 0
            handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
            result = kernel32.WaitForSingleObject(handle, 100)  # 100ms timeout
            if result != WAIT_OBJECT_0:
                return ""
            if not msvcrt.kbhit():
                # Signalled but no character (e.g. window resize event) — flush it
                try:
                    kernel32.FlushConsoleInputBuffer(handle)
                except Exception:
                    pass
                return ""
            ch = msvcrt.getwch()
            if ch in {"\x00", "\xe0"}:
                special = msvcrt.getwch()
                return {"H": "up", "P": "down"}.get(special, "")
            if ch == "\t":
                return "tab"
            return ch

        if self._stdin_fd is None or select is None:
            return ""
        try:  # pragma: no cover - terminal-specific
            ready, _, _ = select.select([self._stdin_fd], [], [], 0.1)
            if not ready:
                return ""
            ch = os.read(self._stdin_fd, 1).decode("utf-8", errors="ignore")
            if ch == "\x1b":
                ready, _, _ = select.select([self._stdin_fd], [], [], 0.01)
                if ready:
                    tail = os.read(self._stdin_fd, 2).decode("utf-8", errors="ignore")
                    return {"[A": "up", "[B": "down"}.get(tail, "")
                return ""
            if ch == "\t":
                return "tab"
            return ch
        except Exception:
            return ""

    def _normalize_key(self, key: str) -> str:
        if key in {"tab", "up", "down", "[", "]"}:
            return key
        if key == "g":
            return "g"
        if key == "G":
            return "G"
        if len(key) == 1:
            return key
        return ""

    def _serialize_state_locked(self) -> dict[str, Any]:
        payload = asdict(self.state)
        payload["agents"] = {str(agent_id): asdict(item) for agent_id, item in self.state.agents.items()}
        payload["iterations"] = {str(iteration): asdict(item) for iteration, item in self.state.iterations.items()}
        payload["processes"] = {str(pid): asdict(item) for pid, item in self.state.processes.items()}
        payload["recent_events"] = [asdict(item) for item in self.state.recent_events]
        payload["recent_logs"] = list(self.state.recent_logs)
        return payload

    def _started_at_epoch(self) -> float:
        started = datetime.fromisoformat(self.state.started_at)
        return started.timestamp()

    def _recompute_gap_locked(self) -> None:
        if self.state.best_score is None or self.state.best_baseline is None or self.state.best_baseline == 0:
            self.state.gap_vs_baseline_percent = None
            return
        self.state.gap_vs_baseline_percent = (
            ((self.state.best_score - self.state.best_baseline) / self.state.best_baseline) * 100.0
        )

    def _recompute_agent_counts_locked(self) -> None:
        counts = {"queued": 0, "running": 0, "done": 0, "failed": 0}
        for item in self.state.agents.values():
            if item.status == "queued":
                counts["queued"] += 1
            elif item.status == "failed":
                counts["failed"] += 1
            elif item.status == "done":
                counts["done"] += 1
            else:
                counts["running"] += 1
        self.state.agent_counts = counts

    def _ensure_selections_locked(self) -> None:
        agent_ids = sorted(self.state.agents)
        process_pids = sorted(self.state.processes)
        if agent_ids and self.state.ui.selected_agent_id not in self.state.agents:
            self.state.ui.selected_agent_id = agent_ids[0]
        if not agent_ids:
            self.state.ui.selected_agent_id = None
        if process_pids and self.state.ui.selected_process_pid not in self.state.processes:
            self.state.ui.selected_process_pid = process_pids[0]
        if not process_pids:
            self.state.ui.selected_process_pid = None

    def _move_selection_locked(self, delta: int) -> None:
        if self._selection_kind_locked() == "process":
            pids = sorted(self.state.processes)
            if not pids:
                return
            current = self.state.ui.selected_process_pid
            index = pids.index(current) if current in pids else 0
            self.state.ui.selected_process_pid = pids[max(0, min(len(pids) - 1, index + delta))]
        else:
            agent_ids = sorted(self.state.agents)
            if not agent_ids:
                return
            current = self.state.ui.selected_agent_id
            index = agent_ids.index(current) if current in agent_ids else 0
            self.state.ui.selected_agent_id = agent_ids[max(0, min(len(agent_ids) - 1, index + delta))]

    def _jump_selection_locked(self, *, first: bool) -> None:
        if self._selection_kind_locked() == "process":
            pids = sorted(self.state.processes)
            if pids:
                self.state.ui.selected_process_pid = pids[0] if first else pids[-1]
        else:
            agent_ids = sorted(self.state.agents)
            if agent_ids:
                self.state.ui.selected_agent_id = agent_ids[0] if first else agent_ids[-1]

    def _selection_kind_locked(self) -> DashboardDetailKind:
        if self.state.ui.active_view == "processes":
            return "process"
        if self.state.ui.active_view == "agents":
            return "agent"
        return self.state.ui.detail_kind

    def _prime_cpu_counter(self, pid: int, process: psutil.Process | None = None) -> None:
        handle = process or psutil.Process(pid)
        try:
            handle.cpu_percent(interval=None)
        except psutil.Error:
            pass

    def _update_process_metrics_locked(self) -> None:
        dead_pids: list[int] = []
        now = time.time()
        for pid, proc_state in list(self.state.processes.items()):
            handle = self._process_handles.get(pid)
            if handle is None:
                try:
                    handle = psutil.Process(pid)
                    self._process_handles[pid] = handle
                    self._prime_cpu_counter(pid, process=handle)
                except psutil.Error:
                    if pid != os.getpid():
                        dead_pids.append(pid)
                    continue
            try:
                if not handle.is_running():
                    if pid != os.getpid():
                        dead_pids.append(pid)
                    continue
                memory_info = handle.memory_info()
                create_time = handle.create_time()
                proc_state.cpu_percent = handle.cpu_percent(interval=None)
                proc_state.rss_mb = memory_info.rss / (1024 * 1024)
                proc_state.age_seconds = max(0.0, now - create_time)
                proc_state.status = "running"
            except psutil.Error:
                if pid != os.getpid():
                    dead_pids.append(pid)
        for pid in dead_pids:
            self._process_handles.pop(pid, None)
            self.state.processes.pop(pid, None)
