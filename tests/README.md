# tests

This folder is the home for the automated test suite.

## What Belongs Here

- Fast, deterministic unit tests for pure logic
- Regression tests for bugs that were fixed
- Parser and formatting tests for logs and coordinator or agent responses
- Sandbox behavior tests that do not require external services

## Testing Workflow

- Run the full suite with `uv run pytest`.
- Add tests with every feature or bug fix when practical.
- Keep default tests independent of a live Ollama server or network access.

## Suggested Growth

- Mirror package areas with files such as `test_core_*.py`, `test_problems_*.py`, and `test_tracking_*.py`.
- Add integration tests separately once there is a stable harness for end-to-end swarm runs.
