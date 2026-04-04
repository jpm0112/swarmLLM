**Algorithmic Performance Patterns**  
- **CP‑SAT (baseline)** – Lowest aggregate makespan (≈ 3579). Robust across all sizes.  
- **MILP + CP‑SAT warm‑start** – Tight big‑M & precedence cuts cut the gap (score 3575) but still ≈ 40 pts above baseline; warm‑start reduces solve time, not makespan.  
- **Hybrid SA (CP‑SAT start)** – Best hybrid (score 3565), critical‑path swaps + incremental feasibility give a modest improvement over pure CP‑SAT.  
- **LNS on CP‑SAT** – Improves feasibility exploration but current relaxation size yields score 3615 (worse than baseline). Needs larger/neighbour‑rich relaxations.  
- **Memetic GA, Tabu, RL, Beam, Learning‑guided priority** – All failed due to feasibility bugs; no performance data to assess merit.  

**Implementation Pitfalls (New & Recurring)**  
- Swapping operations across jobs must verify **identical machine** and **preserve job order**; otherwise precedence violations appear.  
- **State copying** in beam search: shallow copies corrupt schedule dict ordering → `TypeError`. Use deep copy or immutable snapshots.  
- **Precedence enforcement** in RL/SA/Tabu: compute start = max(prev‑finish, machine‑available) after every move.  
- Warm‑start MILP variables must match CP‑SAT naming; mismatched indices cause infeasible starts.  
- Dynamic LNS relaxation size must respect remaining time; static sizes cause early timeout or negligible change.  

**Design Guidelines for Future Iterations**  
- **Warm‑start hybrid pipeline**:  
  1. Solve fast CP‑SAT → extract start times.  
  2. Feed as initial solution to MILP or SA.  
  3. Limit secondary method to ≤ 80 % of overall budget.  
- **Hybrid SA focus** – Restrict moves to critical‑path operation swaps; maintain feasibility via incremental recomputation.  
- **Memetic GA blueprint** – Seed population with CP‑SAT schedule; enforce machine‑consistent crossover/mutation; pair with 2‑opt local search.  
- **LNS tuning** – Randomly select a subset of jobs (≥ 30 % of total) each iteration; increase subset size as time dwindles.  
- **Beam search checklist** –  
  - Deep‑copy schedule state before each expansion.  
  - Rank partial schedules by lower‑bound estimate (e.g., sum of remaining processing times).  
  - Add greedy fallback after beam collapse.  
- **RL policy** – Use shallow network, restrict horizon to 30 s; fall back to greedy dispatcher if timeout.  
- **Testing protocol** – After any mutation, run automated feasibility validator (precedence + machine overlap) before acceptance.  

*Apply these patterns to guide the next cycle of exploration and exploitation.*