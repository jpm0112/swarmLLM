from swarmllm.problems import InstanceProfile
from swarmllm.problems.job_scheduling import JobSchedulingProblem


def _make_problem():
    return JobSchedulingProblem()


def _make_instance(problem, num_jobs=6, seed=123):
    profile = InstanceProfile(
        name="test",
        params={
            "num_jobs": num_jobs,
            "min_processing_time": 1,
            "max_processing_time": 5,
            "due_date_tightness": 0.5,
        },
    )
    return problem.generate_instance(profile, seed)


def test_generate_instance_is_reproducible():
    problem = _make_problem()
    instance_a = _make_instance(problem, num_jobs=6, seed=123)
    instance_b = _make_instance(problem, num_jobs=6, seed=123)

    jobs_a = [(job.id, job.processing_time, job.due_date) for job in instance_a.data]
    jobs_b = [(job.id, job.processing_time, job.due_date) for job in instance_b.data]

    assert jobs_a == jobs_b
    assert instance_a.metadata["total_processing_time"] == instance_b.metadata["total_processing_time"]


def test_baselines_return_valid_permutations():
    problem = _make_problem()
    profile = InstanceProfile(
        name="test", params={"num_jobs": 8, "min_processing_time": 1, "max_processing_time": 20, "due_date_tightness": 0.6}
    )
    instance = problem.generate_instance(profile, seed=7)

    expected_ids = {job.id for job in instance.data}
    baselines = problem.get_baselines(instance)

    assert len(baselines) > 0
    for name, score in baselines.items():
        assert isinstance(score, (int, float))


def test_evaluate_rejects_invalid_permutation():
    problem = _make_problem()
    instance = _make_instance(problem, num_jobs=5, seed=99)
    invalid_schedule = [job.id for job in instance.data[:-1]]

    result = problem.evaluate(instance, invalid_schedule)

    assert result["valid"] is False
    assert "Invalid permutation" in result["error"]


def test_evaluate_returns_metrics_for_valid_schedule():
    problem = _make_problem()
    instance = _make_instance(problem, num_jobs=5, seed=42)
    schedule = [job.id for job in instance.data]  # FIFO order

    result = problem.evaluate(instance, schedule)

    assert result["valid"] is True
    assert result["num_jobs"] == 5
    assert len(result["details"]) == 5
    assert result["score"] >= 0
