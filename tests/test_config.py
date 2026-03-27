from swarmllm.config import Config


def test_default_sandbox_packages_include_gurobipy():
    config = Config()

    assert "gurobipy" in config.sandbox.pip_packages


def test_default_model_temperatures_balance_stability_and_exploration():
    config = Config()

    assert config.llm.temperature_coordinator == 0.4
    assert config.llm.temperature_worker == 1.0
