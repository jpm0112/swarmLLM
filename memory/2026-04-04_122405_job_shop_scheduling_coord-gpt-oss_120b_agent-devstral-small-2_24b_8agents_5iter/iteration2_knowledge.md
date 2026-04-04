**Algorithmic Performance Patterns**

- **Constraint Programming (CP‑SAT)**  
  - Consistently lowest aggregate makespan (≈ 3 500) across all instance sizes.  
  - Default solver settings outperform custom‑tuned, version‑specific flags.  

- **Hybrid Metaheuristics**  
  - **ACO + 2‑opt** beats pure MILP/SA (≈ 7 800 vs 10 700) by combining constructive search with fast local refinement.  
  - Critical‑path‑only perturbations (SA) are viable but limited without a secondary improvement phase.  

- **Exact MILP (Gurobi) + Greedy Fallback**  
  - Guarantees feasible schedule; performance similar to SA, indicating model size or time limit hampers optimality on larger instances.  

- **Pure Evolutionary / Tabu / RL / ML Dispatch**  
  - All failed due to implementation bugs; no performance data to assess algorithmic merit.  

**Implementation Pitfalls (Recurring)**  

- Missing initialization of schedule fields (`'start'`, job lists) → `KeyError`, `IndexError`.  
- Shallow copying of mutable operation dicts → corrupted durations or machine assignments.  
- Solver‑specific parameters that no longer exist (`use_impact_based_search`) → immediate crashes.  
- Time‑limit handling must check elapsed time *before* expensive moves; otherwise timeout (`120 s`).  

**Design Guidelines for Future Iterations**  

- **Start‑up Robustness**  
  - Validate all data structures (deep copy, explicit defaults) before the main loop.  
  - Wrap external solvers in try/except; provide a deterministic greedy fallback.  

- **Hybrid Strategy Blueprint**  
  1. Construct initial sequence with a fast constructive heuristic (e.g., ACO, greedy dispatch).  
  2. Apply a lightweight local search (2‑opt, critical‑path swap) limited by a time budget (e.g., 80 % of timeout).  
  3. If time remains, invoke a secondary exact method (CP‑SAT on reduced sub‑problem).  

- **Parameter Management**  
  - Prefer default solver settings; tune only via documented APIs.  
  - Keep a version‑agnostic configuration layer to auto‑disable unsupported flags.  

- **Problem‑size Awareness**  
  - Use CP‑SAT for medium–hard instances (ft10, ft20).  
  - Reserve MILP/heuristic hybrids for very large or highly constrained instances where CP struggles.  

- **Testing Protocol**  
  - Unit‑test each schedule mutation for feasibility (duration, machine, precedence).  
  - Run a short sanity‑check (e.g., on ft06) after any code change before full evaluation.  