**Algorithmic Performance Patterns**  
- **CP‑SAT (baseline)** – Still the most reliable single method (aggregate ≈ 3579).  
- **MILP + CP‑SAT warm‑start + feasibility cuts** – New best (aggregate ≈ 3567). Nogood cuts from CP infeasible start‑time combos tighten the MILP substantially.  
- **Adaptive LNS‑SA hybrid** – Matches CP‑SAT baseline (aggregate ≈ 3579). Critical‑path swaps + annealing give robust exploration without degrading makespan.  
- **LNS alone** – Previously worse than baseline; needs adaptive relaxation size.  
- **Hybrid Beam (CP‑SAT init)** – Still inferior (≈ 3670); lower‑bound heuristic too weak.  
- **Ant‑Colony, ALNS, Benders, GPU‑local‑search** – All failed this round (bugs, missing ops, timeouts). Their concepts remain promising but require solid feasibility scaffolding.  

**Implementation Pitfalls (Confirmed / New)**  
- **Machine‑assignment integrity** – Any crossover, removal or construction must keep each operation on its original machine; violations cause “wrong machine” errors.  
- **Complete operation set** – ALNS/ACO removal must re‑insert *all* removed ops; dropping operations leads to “expected X ops, got Y” failures.  
- **IntervalVar creation** – CP‑SAT requires `model.NewIntervalVar(start, duration, end)`; passing tuples to `AddNoOverlap` yields infeasible schedules.  
- **Lazy‑constraint naming** – MILP cuts must use the same index mapping as CP‑SAT variables; mismatches produce silent infeasibility.  
- **Deep copy vs. reference** – Beam and GPU parallel searches need true deep copies of schedule state; shallow copies corrupt ordering and cause `TypeError`.  

**Design Guidelines for Next Cycle**  
- **MILP tightening** – After CP warm‑start, extract all start‑time pairs that violate precedence or capacity, encode as `x_i ≤ start_j - 1 ∨ x_i ≥ start_j + dur_j` lazy constraints.  
- **Adaptive LNS‑SA** –  
  1. Start from CP solution.  
  2. At each iteration, select a **critical‑path neighbourhood** of size ≈ 30 % of ops; increase size as temperature drops.  
  3. Apply SA acceptance with geometric cooling; reset temperature if no improvement for k iterations.  
- **ALNS removal heuristic** – Rank machines by load; remove a proportion p of their operations; re‑optimize the removed set with a *small* CP sub‑model (time‑bounded). Adapt p based on recent improvement ratio.  
- **Benders prototype** – Implement a two‑phase loop: (i) master solves sequencing binary variables for a *subset* of machines; (ii) CP sub‑problem solves full schedule, returns makespan cut and infeasibility proof; limit each master iteration to ≤ 5 s.  
- **GPU parallel local search** – Run independent critical‑path swaps on separate threads; after every t ms, perform a reduction to keep the best schedule and broadcast it as a warm‑start for the next batch.  

*Prioritize correctness of machine/operation handling before scaling any meta‑heuristic.*