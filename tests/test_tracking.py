from swarmllm.config import LogConfig
from swarmllm.tracking.shared_log import SharedLog
from swarmllm.tracking.token_tracker import TokenTracker, TokenUsage


def test_shared_log_tracks_best_score(tmp_path):
    log = SharedLog(LogConfig(log_file="results_log.md"), output_dir=str(tmp_path))

    log.append_result(
        iteration=1,
        agent_id=0,
        direction="Try EDD",
        approach="EDD baseline",
        code="def schedule(jobs): return []",
        score=25,
        success=True,
    )
    log.append_result(
        iteration=1,
        agent_id=1,
        direction="Try SPT",
        approach="SPT baseline",
        code="def schedule(jobs): return []",
        score=10,
        success=True,
    )

    assert log.get_best_score() == 10.0


def test_token_tracker_accumulates_iteration_and_role_totals():
    tracker = TokenTracker()

    tracker.record(
        role="agent",
        iteration=1,
        agent_id=0,
        model="worker-model",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    tracker.record(
        role="coordinator",
        iteration=1,
        agent_id=None,
        model="coord-model",
        usage=TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
    )

    summary = tracker.get_iteration_summary(1)

    assert tracker.total_tokens == 25
    assert tracker.agent_total == 15
    assert tracker.coordinator_total == 10
    assert summary["total_tokens"] == 25
    assert summary["agent_calls"] == 1
    assert summary["coordinator_calls"] == 1
