# STL Specifications & Verification

The optimization algorithms (uncoordinated, centralized, Shared ADMM, Consensus ADMM)
produce charging schedules. This module checks whether a schedule **obeys the
temporal requirements**, using Signal Temporal Logic (STL).

> ADMM finds a good charging schedule. STL checks whether that schedule obeys the rules over time.

---

## 1. Requirements and their STL formulas

| # | Requirement | STL formula | Robustness |
|---|-------------|-------------|------------|
| 1 | Every EV reaches its target SoC before departure | `F_[a_i, d_i] (SoC_i >= SoC_i^target)` | `max_{t in [a_i, d_i]} (SoC_i(t) - SoC_i^target)` (SoC units) |
| 2 | Total charging power never exceeds grid capacity | `G_[0, T-1] (P_total <= P_grid)` | `min_t (P_grid(t) - P_total(t))` (kW) |
| — | Overall | `(AND_i Req1_i) AND Req2` | `min` of all terms (grid term normalised by capacity) |

- `F` = **Eventually**: the condition must hold at *some* step in the window.
- `G` = **Always**: the condition must hold at *every* step in the window.
- `a_i`, `d_i` = arrival and departure step of EV `i`. The EV is plugged in during
  `[a_i, d_i)`, so `SoC_i(d_i)` is the charge it leaves with. Charging after departure never counts.
- **Robustness > 0** → satisfied with margin, **= 0** → exactly on the boundary,
  **< 0** → violated, and `|rho|` says by how much.

Example (grid capacity 100 kW):

| P_total | robustness r(t) | verdict |
|---------|-----------------|---------|
| 80 kW   | +20 kW | satisfied |
| 95 kW   | +5 kW  | satisfied |
| 100 kW  | 0 kW   | on the boundary |
| 110 kW  | −10 kW | violated |

`G` takes the **minimum** of r(t) over time (the worst moment); `F` takes the **maximum**.

---

## 2. Files

```
verification/
├── stl.py                    # STL engine: Predicate, Not, And, Or, Always (G), Eventually (F)
│                             # with quantitative (robustness) semantics
├── stl_specifications.py     # builds the signals (SoC_i, P_total) and the two requirement formulas
├── stl_monitor.py            # runs the checks -> STLReport (deadline monitoring, grid episodes)
├── verification_report.py    # text report + SoC / grid / comparison plots
├── tests/test_stl.py         # unit tests (operators, worked examples, algorithms)
└── README.md

experiments/stl_verification.py   # benchmark: all algorithms x all scenarios
```

---

## 3. Quick start

Run from the project root (`ev_project/`).

**Verify one algorithm on one scenario**

```bash
python -m verification.stl_monitor --algorithm centralized --scenario morning_peak
python -m verification.stl_monitor --algorithm uncoordinated_asap --scenario evening_peak --grid-factor 0.5
python -m verification.stl_monitor --algorithm shared_admm_adaptive --topology star --all-evs
```

Algorithms: `uncoordinated_asap`, `uncoordinated_fcfs`, `centralized`, `shared_admm`,
`shared_admm_adaptive`, `consensus_admm`, `consensus_admm_adaptive`.
Scenarios: `morning_peak`, `evening_peak`, `random`, `high_demand`, `low_demand`.

**Benchmark all algorithms (same instances for every algorithm)**

```bash
python -m experiments.stl_verification                          # 5 scenarios, ~4 min
python -m experiments.stl_verification --grid-factor 0.5        # tighter grid
python -m experiments.stl_verification --no-admm                # fast: uncoordinated + centralized
python -m experiments.stl_verification --seeds 0 1 2 --tag _3seeds
```

**Use it from Python**

```python
from ev.fleet import scenario_instance
from common.objective import WEIGHT_CASES
from network.integration import run_algorithm
from verification import verify_schedule, verify_result

data, _ = scenario_instance("morning_peak", n_evs=30, n_stations=3, seed=0)
res = run_algorithm("centralized", data, WEIGHT_CASES["baseline"])

rep = verify_result(res, data)    # also adds stl_* keys to res.metrics
print(rep.text())                 # full report
rep.satisfied                     # overall True/False
rep.to_row()                      # flat dict of stl_* metrics
rep.ev_table()                    # per-EV pandas DataFrame

# any (N, T) schedule works, e.g. an ADMM iterate:
rep = verify_schedule(data, P, algorithm="my_schedule")
```

**Write your own STL formula**

```python
import numpy as np
from verification.stl import Predicate, Always, Eventually

trace = {"P_total": np.array([80, 95, 100, 110.0])}
phi = Always(Predicate("P_total", "<=", 100.0), 0, 3)
phi.rho(trace)            # -10.0  (violated by 10 kW at the worst step)
phi.critical_time(trace)  # 3
```

**Tests**

```bash
python -m pytest -q verification/tests
```

---

## 4. What the report contains

```
REQUIREMENT 1 - target SoC before departure: VIOLATED
  EVs satisfied: 29/30   min robustness: -1.80 pp
  deadline margin (target reached before departure): min 0.0 h, mean 2.3 h
  violated: 1  (of which 1 cannot reach target even charging alone at P_max for the whole stay)

    EV stn arrive depart   SoC0 target SoC@dep  rho[pp] reached  margin  status
    23   2  19:00  06:00  24.4%  81.8%   80.0%    -1.80     -         -  VIOLATED (unreachable)
    ...

REQUIREMENT 2 - grid capacity: VIOLATED
  Maximum capacity: 75.0 kW
  Observed peak:    168.2 kW at 18:30
  Robustness:       -93.19 kW  (-124.3 % of capacity), critical time 18:30
  Violations: 1 episode(s), 4.5 h over the limit in total
    17:30 for 4.5 h: worst excess 93.2 kW, overload 272.5 kWh; largest loads: EV28 (11.0 kW), ...

Overall STL status: VIOLATED
```

**Per EV (deadline monitoring):** SoC at departure, robustness, the time the target was
reached, the **deadline margin** (hours between reaching the target and departing), and
whether the target was **physically reachable** (`eta * P_max * stay >= required energy`).

**Grid:** peak load, robustness in kW and as a % of capacity, the critical time, and every
violation episode (start, duration, worst excess, energy above the limit, largest contributing EVs).

### Metrics (`rep.to_row()`, also the CSV columns)

| Column | Meaning |
|--------|---------|
| `stl_satisfied` | overall verdict (all EVs + grid) |
| `stl_satisfied_reachable` | overall verdict counting only EVs whose target is reachable |
| `stl_rho_overall` | overall robustness (normalised) |
| `stl_req1_sat_pct` | % of EVs meeting Req 1 |
| `stl_req1_reachable_sat_pct` | % of *reachable* EVs meeting Req 1 |
| `stl_req1_n_violated`, `stl_req1_n_unreachable` | violated EVs, and how many of them were impossible |
| `stl_req1_rho_min`, `stl_req1_rho_mean` | SoC robustness (fraction; ×100 = percentage points) |
| `stl_req1_min_margin_h`, `stl_req1_mean_margin_h` | deadline margin of satisfied EVs (temporal robustness) |
| `stl_req2_satisfied` | grid verdict |
| `stl_req2_rho_kw`, `stl_req2_rho_norm` | grid robustness in kW / fraction of capacity |
| `stl_req2_viol_steps`, `stl_req2_viol_hours` | time spent over capacity |
| `stl_req2_overload_kwh` | energy drawn above the limit |

### Outputs of the benchmark

```
results/verification/stl_runs.csv        one row per (scenario, seed, algorithm)
results/verification/stl_summary.csv     mean over seeds
results/verification/stl_evs.csv         per-EV verdicts
results/verification/reports/*.txt       full text report per run
plots/verification/stl_comparison.png    algorithms side by side
plots/verification/<scenario>_<algorithm>_grid.png   load vs capacity + robustness signal
plots/verification/<scenario>_<algorithm>_soc.png    SoC trajectories of the worst EVs
```

---

## 5. Results (30 EVs, 3 stations, seed 0, default grid = 0.7 × station capacity)

| Scenario | Algorithm | Reachable EVs meeting Req 1 | Grid robustness | Hours over capacity |
|----------|-----------|----------------------------:|----------------:|--------------------:|
| morning_peak | uncoordinated ASAP | 100 % | −87.2 kW | 3.5 |
| morning_peak | uncoordinated FCFS | 96.2 % | 0.0 kW | 0 |
| morning_peak | centralized | 100 % | 0.0 kW | 0 |
| morning_peak | Shared ADMM (star) | 100 % | −0.09 kW* | 0 |
| morning_peak | Consensus ADMM (random) | 100 % | 0.0 kW | 0 |
| evening_peak | uncoordinated ASAP | 100 % | −63.2 kW | 3.5 |
| evening_peak | uncoordinated FCFS | 89.7 % | 0.0 kW | 0 |
| evening_peak | centralized / both ADMM | 100 % | +4.1 kW | 0 |
| random, high_demand, low_demand | all | 100 % | positive | 0 |

\* within the numerical tolerance (0.1 % of capacity), so counted as satisfied.

**Takeaways**
- **Uncoordinated ASAP** violates Req 2 in the peak scenarios: everyone charges at once.
- **FCFS** respects the grid, but late arrivals miss their targets (it violates Req 1).
- **Centralized and both ADMM variants** satisfy both requirements for every EV that can
  physically be served, and ADMM matches the centralized verdicts.
- Some EVs (1–13 per scenario) **cannot** reach their target in any schedule (stay too short or
  charger too slow). They are flagged `VIOLATED (unreachable)`, so the strict overall verdict is
  VIOLATED for every algorithm. Use `stl_satisfied_reachable` to compare the algorithms.

---

## 6. Notes on interpretation

- **Req 1 robustness is at most 0 for our schedules.** The problem forbids charging beyond the
  target (headroom = `min(B(SoC_max − SoC_init), R_i)`), so a served EV ends exactly at its target
  (rho = 0) and an unserved one ends below it (rho < 0). How safely an EV meets its deadline is
  therefore measured by the **deadline margin**: the optimizers often finish right at departure
  to use cheap night tariffs, while ASAP finishes hours early.
- **Tolerance.** ADMM iterates are only feasible in the limit, so the Boolean verdict uses
  `rho >= −tol` with `TOL_SOC = 1e-3` (0.1 pp) and `TOL_GRID = 1e-3` (0.1 % of capacity), set in
  `stl_monitor.py`. Raw robustness is always reported unchanged.
- **Units in the overall formula.** STL `AND` takes a minimum, which only makes sense in common
  units, so the grid term is divided by capacity before combining it with the SoC terms.
- The monitor works on any `(N, T)` schedule, so it can also be applied to ADMM iterates
  during convergence.

## 7. References

- O. Maler, D. Nickovic, *Monitoring Temporal Properties of Continuous Signals*, FORMATS 2004.
- A. Donzé, O. Maler, *Robust Satisfaction of Temporal Logic over Real-Valued Signals*, FORMATS 2010.
