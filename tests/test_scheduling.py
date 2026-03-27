from swarmllm.problems.scheduling import (
    baseline_edd,
    baseline_fifo,
    baseline_spt,
    evaluate_schedule,
    generate_instance,
)


def test_generate_instance_is_reproducible():
    instance_a = generate_instance(num_jobs=6, seed=123, min_pt=1, max_pt=5, due_date_tightness=0.5)
    instance_b = generate_instance(num_jobs=6, seed=123, min_pt=1, max_pt=5, due_date_tightness=0.5)

    jobs_a = [(job.id, job.processing_time, job.due_date) for job in instance_a.jobs]
    jobs_b = [(job.id, job.processing_time, job.due_date) for job in instance_b.jobs]

    assert jobs_a == jobs_b
    assert instance_a.total_processing_time == instance_b.total_processing_time


def test_baselines_return_valid_permutations():
    instance = generate_instance(num_jobs=8, seed=7)

    expected_ids = {job.id for job in instance.jobs}

    for schedule in (baseline_fifo(instance), baseline_edd(instance), baseline_spt(instance)):
        assert set(schedule) == expected_ids
        assert len(schedule) == len(expected_ids)


def test_evaluate_schedule_rejects_invalid_permutation():
    instance = generate_instance(num_jobs=5, seed=99)
    invalid_schedule = [job.id for job in instance.jobs[:-1]]

    result = evaluate_schedule(instance, invalid_schedule)

    assert result["valid"] is False
    assert "Invalid permutation" in result["error"]


def test_evaluate_schedule_returns_metrics_for_valid_schedule():
    instance = generate_instance(num_jobs=5, seed=42)
    schedule = baseline_fifo(instance)

    result = evaluate_schedule(instance, schedule)

    assert result["valid"] is True
    assert result["num_jobs"] == 5
    assert len(result["details"]) == 5
    assert result["total_tardiness"] >= 0
