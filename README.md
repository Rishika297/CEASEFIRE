
# CEASEFIRE
EV charging optimization scheeduling

# Distributed Multi-Agent Optimization for Adaptive EV Charging

**Team Ceasefire** · Multi-Agent Systems course project · IIT Hyderabad

Many electric vehicles (EVs) plug into a few charging stations that share one grid
connection. Each EV wants enough charge before it leaves; the grid operator wants low
cost, low peaks and no overload. This project schedules charging as a **multi-agent
optimization problem**. It compares:

- **Uncoordinated** charging (ASAP and first-come-first-served),
- a **centralized** convex optimum (CVXPY), and
- two **distributed ADMM** algorithms (Shared ADMM and Consensus ADMM), in which every EV is an
  agent that only talks to a coordinator or to its neighbours on a communication graph.

**Signal Temporal Logic (STL)** monitoring then checks that each resulting schedule meets the
deadline and grid requirements.

```
 EV fleet & scenarios ──► optimization (uncoordinated / centralized / ADMM) ──► charging schedule P (N x T)
        ev/                    centralized/  algorithms/  network/                     │
                                                                                       ▼
                         evaluation/ (metrics, gaps, convergence)  +  verification/ (STL)
                                                                                       │
                                                                                       ▼
                                      experiments/ (CSV tables + plots)   simulations/ (GIF animations)
```

---

## 1. Problem formulation

Time is split into `T = 48` intervals of `dt = 0.5 h` (06:00 → 06:00 next day).
EV `i` is plugged in for steps `[a_i, d_i)` at station `m(i)`, and `p_{i,t}` is its charging power [kW].

**Objective**

```
min  J = J_cost + α J_peak + β J_delay + γ J_fair

J_cost  = (1/C_ref) Σ_t π_t dt L_t                       electricity cost (time-of-use price π_t)
J_peak  = (1/T) Σ_t (L_t / P_ref)²                       peak / load flattening,  L_t = Σ_i p_{i,t}
J_delay = Σ_i Σ_t (dt/H_ref) (R_i − e_{i,t}) / R_i        energy still missing over time (charge early)
J_fair  = Σ_i [ s_i/R_i + (s_i/R_i)² ]                   unmet energy, penalised fairly across EVs
```

**Constraints** (`common/constraints.py`)

| | Constraint | Meaning |
|---|---|---|
| C1 | `0 ≤ p_{i,t} ≤ P_i^max A_{i,t}` | charger limit, only while plugged in |
| C2 | `SoC_{i,t+1} = SoC_{i,t} + η_i p_{i,t} dt / B_i` | battery dynamics (eliminated analytically) |
| C3 | `SoC_min ≤ SoC_{i,t} ≤ SoC_max` | battery limits |
| C4 | `Σ_{i∈E_m} p_{i,t} ≤ P_m^max` | station capacity |
| C5 | `Σ_i p_{i,t} ≤ P_grid^max` | shared grid capacity (couples all agents) |
| C6–C8 | `η_i dt Σ_t p_{i,t} + s_i ≥ R_i`, `0 ≤ s_i ≤ R_i` | energy requirement with slack `s_i` (soft) |

`R_i = B_i (SoC_i^target − SoC_i^init)` is the energy EV `i` needs. The problem is a convex QP
and is always feasible (`P = 0, s = R`).

**Weight cases** (Experiment 2): `baseline (1,1,1)`, `peak_priority (10,1,1)`,
`delay_priority (1,10,1)`, `fairness_priority (1,1,10)` for `(α, β, γ)`.

---

## 2. Algorithms

| Name | Module | Communication | Notes |
|------|--------|---------------|-------|
| `uncoordinated_asap` | `centralized/uncoordinated.py` | none | full power on arrival; **can overload** station/grid |
| `uncoordinated_fcfs` | `centralized/uncoordinated.py` | none | greedy in arrival order, respects capacity |
| `centralized` | `centralized/solver.py` | all data at one operator | CVXPY QP; reference optimum `J*` |
| `shared_admm` / `_adaptive` | `algorithms/shared_admm/` | coordinator (hub) or gossip averaging | sharing ADMM (Boyd et al. 2011, §7.3) |
| `consensus_admm` / `_adaptive` | `algorithms/consensus_admm/` | neighbours only (fully decentralised) | proximal dual consensus ADMM |

`_adaptive` = residual-balancing penalty update (`algorithms/rho_update.py`). Each EV agent solves
its local proximal problem in `ev/local_opt.py`. The penalty `ρ` is set automatically from the
data (`algorithms/base.py: auto_rho`).

**Communication topologies** (`network/topologies.py`): `ring`, `star`, `mesh`, `random`
(Erdős–Rényi, resampled until connected). `network/averaging.py` gives exact hub averaging
(star) or gossip averaging (other graphs); `network/messaging.py` counts every message and
every scalar sent.

---

## 3. Project structure

```
ev_project/
├── common/              # shared foundation (everyone imports from here)
│   ├── constants.py     #   units, horizon, tariffs, scenario definitions, defaults
│   ├── utils.py         #   ProblemData container, TOU prices, paths, Timer
│   ├── constraints.py   #   C1–C8 in NumPy (violation checks) and CVXPY
│   ├── objective.py     #   J_cost, J_peak, J_delay, J_fair + weight cases
│   └── result.py        #   Result: common output of every algorithm (save/load .npz)
│
├── ev/                  # EV modelling & fleet generation
│   ├── ev_model.py      #   EV dataclass, SoC trajectory, ASAP schedule, EVState
│   ├── schedule.py      #   arrival/departure sampling, availability, events
│   ├── fleet.py         #   generate fleets -> ProblemData (make_instance, scenario_instance)
│   ├── constraints.py   #   per-EV constraints and projections
│   ├── local_opt.py     #   per-agent proximal subproblem used by ADMM
│   ├── experiments/     #   arrival scenarios, EV trajectories, feasibility, sensitivity
│   └── tests/
│
├── centralized/         # centralized benchmark & grid model
│   ├── solver.py        #   CVXPY QP -> J*
│   ├── uncoordinated.py #   ASAP and FCFS baselines
│   ├── grid_model.py    #   capacity variants, station loads, utilisation
│   ├── prices.py        #   tou / flat / two_level / real_time price profiles
│   ├── experiments/     #   (placeholders, see §8)
│   └── tests/
│
├── algorithms/          # distributed ADMM
│   ├── base.py          #   ADMMSettings, auto rho, load distance, message counter
│   ├── rho_update.py    #   adaptive penalty (residual balancing)
│   ├── stopping.py      #   stopping criteria
│   ├── shared_admm/     #   Shared ADMM + building blocks
│   ├── consensus_admm/  #   Consensus ADMM + building blocks
│   └── tests/
│
├── network/             # communication layer
│   ├── topologies.py    #   ring / star / mesh / random graphs
│   ├── averaging.py     #   HubAverager (exact), GossipAverager (Metropolis weights)
│   ├── messaging.py     #   neighbour message bus (+ optional link drops)
│   ├── comm_metrics.py  #   graph properties, communication cost
│   ├── graph_utils.py
│   ├── integration.py   #   run_algorithm(): ONE entry point for every algorithm
│   └── tests/
│
├── verification/        # STL specifications & verification  (see verification/README.md)
│   ├── stl.py           #   STL engine with robustness
│   ├── stl_specifications.py, stl_monitor.py, verification_report.py
│   └── tests/
│
├── evaluation/          # judging schedules identically
│   ├── metrics.py       #   peak, cost, delay, target-SoC %, Jain fairness, violations
│   ├── objective_evaluation.py  # run_suite(): run all algorithms on one instance, gaps to J*
│   ├── convergence.py   #   iterations / messages to a given gap or load distance
│   └── visualization.py #   shared plotting functions
│
├── experiments/         # numbered experiments -> results/ and plots/
├── simulations/         # presentation animations (GIF + PNG)
├── configs/             # YAML settings (fleet, simulation, ADMM, network)
├── results/             # CSV / NPZ outputs
├── plots/               # PNG / GIF outputs
├── tests/               # integration & end-to-end tests
└── theory/              # (placeholder)
```

---

## 4. Installation

Python 3.10+ is needed (tested on 3.13).

```bash
pip install -r requirements.txt
# numpy scipy pandas cvxpy osqp networkx matplotlib pillow pyyaml pytest
```

**Run every command from the project root (`ev_project/`)** with `python -m ...`, so the package
imports resolve.

---

## 5. Quick start

```bash
python -m ev.fleet                     # generate the 5 scenario fleets, print summaries
python -m centralized.solver           # solve one centralized instance
python -m network.integration          # demo: run algorithms on topologies
python -m network.topologies           # topology summaries
```

From Python, every algorithm goes through one function and returns the same `Result` object:

```python
from ev.fleet import scenario_instance
from common.objective import WEIGHT_CASES
from network.integration import run_algorithm
from verification import verify_result

data, evs = scenario_instance("morning_peak", n_evs=30, n_stations=3, seed=0)
w = WEIGHT_CASES["baseline"]

cen = run_algorithm("centralized", data, w)
adm = run_algorithm("consensus_admm_adaptive", data, w, topology="random",
                    J_star=cen.objective, L_star=cen.P.sum(axis=0))

adm.P                         # (N, T) charging powers [kW]
adm.metrics["peak_load_kw"]   # evaluation metrics
adm.iterations, adm.messages  # convergence & communication
print(verify_result(adm, data).text())   # STL verification report
```

---

## 6. Experiments

| # | Experiment | Command | Outputs |
|---|-----------|---------|---------|
| 1 | Algorithms × topologies (iterations, residuals, messages, runtime) | `python -m experiments.algorithm_topology` (~10 min) | `results/topologies/exp1_*.csv`, `plots/topology_comparison/` |
| 2 | Objective-weight sensitivity (α, β, γ) | `python -m experiments.objective_weight_sensitivity [--grid-factor 0.35]` | `results/sensitivity/exp2*_*.csv`, `plots/objective_sensitivity/` |
| 3 | Scalability, N = 10 … 200 EVs | `python -m experiments.scalability` (~5 min) | `results/scalability/exp3_*.csv`, `plots/scalability/` |
| 5 | Charging scenarios (morning/evening peak, random, high/low demand) | `python -m experiments.charging_scenarios` (~4 min) | `results/scenarios/exp5_*.csv`, `plots/scenarios/` |
| STL | Temporal-requirement verification of every algorithm | `python -m experiments.stl_verification` (~4 min) | `results/verification/`, `plots/verification/` |
| — | Penalty sweep, fixed vs adaptive ρ | `python -m experiments.compare_fixed_adaptive ring consensus` | `results/comparisons/rho_sweep_*.csv` |
| — | Convergence traces on one instance | `python -m experiments.compare_algorithms --case fairness_priority --grid-factor 0.35` | `plots/convergence/` |

Useful flags (most experiments): `--seeds 0 1 2` (report version), `--n-evs`, `--stations`,
`--max-iter`, `--grid-factor` (grid capacity as a fraction of total station capacity; default 0.7).

**EV-level analysis** (`ev/experiments/`):

```bash
python -m ev.experiments.arrival_scenarios   # arrival histograms & occupancy
python -m ev.experiments.ev_trajectories     # ASAP vs selfish SoC trajectories vs grid limit
python -m ev.experiments.feasibility_check   # % EVs able to reach target, demand/supply ratio
python -m ev.experiments.sensitivity         # battery / power / SoC sweeps
```

---

## 7. Simulations (presentation animations)

All write a GIF and a PNG to `plots/animations/`.

| Command | Shows |
|---------|-------|
| `python -m simulations.sim_story` | EVs as moving dots: requests → negotiation → the day, for all 4 approaches |
| `python -m simulations.sim_grid_load` | total load through the day, uncoordinated vs centralized vs ADMM |
| `python -m simulations.sim_ev_soc` | every EV's battery bar, uncoordinated vs coordinated |
| `python -m simulations.sim_admm_convergence` | Shared and Consensus ADMM converging to the optimum |
| `python -m simulations.sim_topology_race` | Consensus ADMM on ring / star / mesh / random side by side |
| `python -m simulations.sim_network_messages` | messages flowing on the communication graph |
| `python -m simulations.sim_weight_cases` | how (α, β, γ) change the optimal schedule |
| `python -m simulations.sim_dynamic_arrivals` | online rolling-horizon control with unknown future arrivals |

---

## 8. Scenarios and default parameters

Defined in `common/constants.py` and `configs/fleet_config.yaml`.

| Scenario | Arrivals | Stay | Initial SoC | Target SoC |
|----------|----------|------|-------------|------------|
| `morning_peak` | ~N(09:00, 1 h) | ~N(8 h, 1.5 h) | 20–50 % | 80–90 % |
| `evening_peak` | ~N(18:30, 1.5 h) | ~N(10 h, 2 h) | 20–50 % | 80–90 % |
| `random` | uniform over the day | ~N(6 h, 2.5 h) | 20–50 % | 80–90 % |
| `high_demand` | uniform | ~N(6 h, 2.5 h) | 5–25 % | 90–95 % |
| `low_demand` | uniform | ~N(6 h, 2.5 h) | 50–70 % | 80–85 % |

- Batteries 30/40/60/75 kWh, chargers 3.3/7.4/11 kW, efficiency η = 0.92.
- Station capacity = 5 kW × (EVs at the station); grid capacity = 0.7 × total station capacity.
- Time-of-use tariff (Rs/kWh): 4.5 (00–06), 6.0 (06–09), 7.0 (09–17), 9.5 (17–22), 5.0 (22–24).
- ADMM settings live in `configs/admm_config.yaml` (`rho: auto`, `max_iter: 3000`,
  `eps_abs = eps_rel = 1e-4`, adaptive ρ frozen after 300 iterations).
- Network settings live in `configs/network_config.yaml`.

---

## 9. Selected results

These come from the saved CSVs in `results/` (30 EVs, 3 stations, seed 0, baseline weights,
unless stated otherwise).

**Experiment 5, morning peak** (`results/scenarios/exp5_summary.csv`)

| Algorithm | Peak load [kW] | Cost [Rs] | EVs at target | Gap to J* | Max violation [kW] |
|-----------|---------------:|----------:|--------------:|----------:|-------------------:|
| Uncoordinated ASAP | 192.2 | 5462 | 86.7 % | 5.5 % | **87.2** |
| Uncoordinated FCFS | 105.0 | 5274 | 83.3 % | 80.7 % | 0 |
| Centralized | 105.0 | 5482 | 86.7 % | 0 | 0 |
| Shared ADMM adaptive (star) | 105.1 | 5482 | 86.7 % | ≈ 0 | 0.09 |
| Consensus ADMM (random) | 105.0 | 5482 | 86.7 % | ≈ 0 | 0 |

FCFS is cheapest in rupees but has the largest objective gap: it leaves 44.4 kWh undelivered
(vs 15.9 kWh for centralized), so its fairness term `J_fair` is almost 4× higher.

**Experiment 3, scalability** (`results/scalability/exp3_summary.csv`)

| N EVs | Centralized [s] | Shared ADMM adaptive (star) [s / iters] | Consensus ADMM (random) [s / iters] |
|------:|----------------:|-----------------------------------------:|------------------------------------:|
| 10 | 0.92 | 0.36 / 108 | 2.7 / 780 |
| 50 | 0.08 | 0.97 / 200 | 16.2 / 2000 |
| 200 | 0.30 | 1.84 / 190 | 155.8 / 2000 |

Shared ADMM keeps ~2 messages per agent per iteration at every size. Consensus ADMM needs no
coordinator, but its communication grows with node degree (≈ 10.7 messages per agent per
iteration at N = 200).

**STL verification** (`results/verification/`, see `verification/README.md`)
- ASAP violates the grid requirement in both peak scenarios (−87 kW and −63 kW robustness).
- FCFS misses targets of EVs that could have been served.
- Centralized and both ADMM variants satisfy both requirements for every EV whose target is
  physically reachable.

---

## 10. Testing

```bash
python -m pytest -q                        # everything (~1 min)
python -m pytest -q verification/tests     # one package
```

Known issue: `tests/test_experiment_configs.py::test_exp2_smoke` expects
4 weight cases × 5 runs = 20 rows but the smoke run produces 12.

---

## 11. Conventions

- **Units:** power kW, energy kWh, time h (or integer step index), price Rs/kWh, SoC a fraction in [0, 1].
- **Time:** EV `i` is plugged in for `t ∈ [arrival, departure)` (departure exclusive).
  `SoC[:, t]` is the charge at the *start* of interval `t`, so the trajectory has `T+1` points.
- **Schedules** are always an `(N, T)` array `P`, so any metric or check applies to any algorithm.
- **Outputs** go to `results/<topic>/` (CSV, NPZ via `Result.save`) and `plots/<topic>/`.


## References

- S. Boyd et al., *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*, 2011.
- A. Donzé, O. Maler, *Robust Satisfaction of Temporal Logic over Real-Valued Signals*, FORMATS 2010.
