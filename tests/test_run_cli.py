from scripts.run import build_parser
from swarmllm.tracking.telemetry import resolve_dashboard_mode


def test_dashboard_parser_defaults_to_auto():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.dashboard == "auto"


def test_dashboard_parser_accepts_explicit_mode():
    parser = build_parser()
    args = parser.parse_args(["--dashboard", "plain"])
    assert args.dashboard == "plain"


def test_resolve_dashboard_mode_falls_back_without_tty():
    assert resolve_dashboard_mode("auto", is_tty=False) == "plain"
    assert resolve_dashboard_mode("tui", is_tty=False) == "plain"
    assert resolve_dashboard_mode("plain", is_tty=True) == "plain"
