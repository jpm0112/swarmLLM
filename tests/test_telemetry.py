import json
import time

from swarmllm.tracking.telemetry import RunTelemetry, ThroughputTracker
from swarmllm.tracking.token_tracker import TokenUsage


def test_throughput_tracker_reports_rolling_and_lifetime_tps():
    tracker = ThroughputTracker(window_seconds=30.0)
    tracker.record("agent", total_tokens=300, duration_seconds=10.0, completed_at=10.0)
    tracker.record("coordinator", total_tokens=80, duration_seconds=4.0, completed_at=20.0)
    tracker.record("agent", total_tokens=50, duration_seconds=5.0, completed_at=50.0)

    snapshot = tracker.snapshot(now=50.0)

    assert round(snapshot.rolling_tps, 2) == round((80 + 50) / (4.0 + 5.0), 2)
    assert round(snapshot.lifetime_tps, 2) == round((300 + 80 + 50) / (10.0 + 4.0 + 5.0), 2)
    assert round(snapshot.rolling_agent_tps, 2) == round(50 / 5.0, 2)
    assert round(snapshot.rolling_coordinator_tps, 2) == round(80 / 4.0, 2)


def test_run_telemetry_writes_events_and_live_state(tmp_path):
    telemetry = RunTelemetry(
        str(tmp_path),
        requested_mode="plain",
        is_tty=False,
        start_thread=False,
    )
    telemetry.set_run_metadata(total_iterations=2, total_agents=1, current_stage="testing")
    telemetry.set_baseline(100.0)
    telemetry.set_best_score(90.0)
    telemetry.queue_agent(0, iteration=1, mode="explore", direction="Try EDD", endpoint_label="worker-a")
    telemetry.set_agent_phase(0, iteration=1, phase="llm", status="running")
    telemetry.record_llm_call(
        role="agent",
        iteration=1,
        agent_id=0,
        model="worker-model",
        duration_seconds=2.0,
        usage=TokenUsage(prompt_tokens=40, completion_tokens=10, total_tokens=50),
        endpoint_label="worker-a",
    )
    telemetry.complete_agent(
        agent_id=0,
        iteration=1,
        success=True,
        score=90.0,
        llm_time_seconds=2.0,
        exec_time_seconds=1.0,
        runtime_seconds=3.0,
    )
    telemetry.emit_event("custom_event", message="hello")
    telemetry.refresh_now()
    telemetry.close(status="completed")

    state = json.loads((tmp_path / "live_state.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert state["total_tokens"] == 50
    assert state["best_score"] == 90.0
    assert state["best_baseline"] == 100.0
    assert state["agent_counts"]["done"] == 1
    assert state["agents"]["0"]["phase"] == "done"
    assert state["agents"]["0"]["total_tokens"] == 50
    assert state["agents"]["0"]["llm_call_count"] == 1
    assert state["agents"]["0"]["last_model"] == "worker-model"
    assert any(event["event_type"] == "custom_event" for event in events)
    assert (tmp_path / "run.log").exists()


def test_run_telemetry_refreshes_process_metrics(monkeypatch, tmp_path):
    class FakeMemory:
        rss = 50 * 1024 * 1024

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def cpu_percent(self, interval=None):
            return 12.5

        def memory_info(self):
            return FakeMemory()

        def create_time(self):
            return time.time() - 5.0

        def is_running(self):
            return True

    monkeypatch.setattr("swarmllm.tracking.telemetry.psutil.Process", FakeProcess)

    telemetry = RunTelemetry(
        str(tmp_path),
        requested_mode="plain",
        is_tty=False,
        start_thread=False,
    )
    telemetry.register_process(4242, label="sandbox child", kind="python", role="sandbox")
    telemetry.refresh_now()
    telemetry.close(status="completed")

    state = json.loads((tmp_path / "live_state.json").read_text(encoding="utf-8"))
    proc_state = state["processes"]["4242"]
    assert proc_state["cpu_percent"] == 12.5
    assert round(proc_state["rss_mb"], 1) == 50.0
    assert proc_state["status"] == "running"


def test_run_telemetry_handles_view_navigation(tmp_path):
    class FakeMemory:
        rss = 30 * 1024 * 1024

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def cpu_percent(self, interval=None):
            return 3.0

        def memory_info(self):
            return FakeMemory()

        def create_time(self):
            return time.time() - 2.0

        def is_running(self):
            return True

    import swarmllm.tracking.telemetry as telemetry_mod

    original_process = telemetry_mod.psutil.Process
    telemetry_mod.psutil.Process = FakeProcess
    try:
        telemetry = RunTelemetry(
            str(tmp_path),
            requested_mode="plain",
            is_tty=False,
            start_thread=False,
        )
        telemetry.queue_agent(0, iteration=1, mode="explore", direction="Try EDD", endpoint_label="worker-a")
        telemetry.queue_agent(1, iteration=1, mode="explore", direction="Try SPT", endpoint_label="worker-b")
        telemetry.register_process(
            999999,
            label="sandbox child",
            kind="python",
            role="sandbox",
            metadata={"agent_id": 1, "instance": "tiny"},
            command="python agent_run.py",
            cwd="/tmp/swarmllm",
        )

        telemetry.handle_input_key("a")
        telemetry.handle_input_key("j")
        telemetry.handle_input_key("d")
        telemetry.handle_input_key("]")
        telemetry.handle_input_key("G")
        telemetry.refresh_now()
        telemetry.close(status="completed")
    finally:
        telemetry_mod.psutil.Process = original_process

    state = json.loads((tmp_path / "live_state.json").read_text(encoding="utf-8"))
    assert state["ui"]["active_view"] == "detail"
    assert state["ui"]["detail_kind"] == "process"
    assert state["ui"]["selected_agent_id"] == 1
    assert state["ui"]["selected_process_pid"] == 999999
    assert state["processes"]["999999"]["command"] == "python agent_run.py"
    assert state["processes"]["999999"]["cwd"] == "/tmp/swarmllm"
