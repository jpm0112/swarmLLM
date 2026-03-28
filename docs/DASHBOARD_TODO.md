# Dashboard Follow-Ups

This file tracks planned dashboard enhancements after the first interactive TUI pass.

## Current Scope

Implemented today:

- Overview, agents, processes, and detail views in the Rich TUI
- Keyboard navigation for switching views and moving through agents/processes
- Agent detail with cumulative token usage, last completed call stats, timings, and instance results
- Process detail with PID, command, cwd, and sandbox/backend metadata

## Future Work

### Live Token / Text Streaming

Goal:

- Show in-flight token growth and partial model output while a coordinator or worker call is still running.

What this requires:

- Switch the LLM request path from completed-response accounting to a streaming-capable transport
- Emit partial telemetry events for:
  - request start
  - token delta
  - partial text delta
  - request end
  - cancellation or transport failure
- Keep final `TokenTracker` totals authoritative and derived from the completed call metadata
- Make streaming optional and fail-open so normal swarm execution continues if the dashboard cannot consume deltas

Suggested telemetry shape:

- `llm_stream_started`
- `llm_stream_delta`
  - role
  - iteration
  - agent_id
  - model
  - text_delta
  - prompt_tokens_so_far if available
  - completion_tokens_so_far if available
  - timestamp
- `llm_stream_completed`
  - final prompt/completion/total tokens
  - wall duration

Suggested TUI additions once streaming exists:

- Selected-agent live token counter that updates before the call completes
- Partial output pane for the selected worker/coordinator
- Request-state indicator such as `connecting`, `streaming`, `tool-call`, `completed`

Constraints:

- Do not block swarm execution on streamed UI updates
- Keep plain mode and saved artifacts working exactly as they do today
- Preserve deterministic tests by keeping streaming optional and mockable
