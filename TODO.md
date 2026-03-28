# SwarmLLM — TODO

## Features & Improvements

- [ ] **Agents planning (project lead)** — Add a planning/leadership layer where agents can coordinate or propose strategies before execution
- [ ] **Add Job Shop Scheduling problem** — Implement using JSPLIB library instances as benchmark
- [ ] **Complete CA problem files** — Finish the Combinatorial Auctions problem module (evaluation, baselines, instance loading)
- [ ] **Improve prompts for all** — Refine system/user prompts for agents and coordinator to increase success rates
- [ ] **Reduce failures** — Address common failure modes (invalid permutations, undefined variables, timeouts). Consider better error feedback, smarter retries, or prompt changes
- [ ] **Design coordinator memory** — Replace full results log with a summarized/compressed memory to reduce coordinator context size and avoid timeouts
- [ ] **Replace Ollama** — Evaluate alternatives (vLLM, llama.cpp server, TGI) that support higher parallelism for running more concurrent agents
