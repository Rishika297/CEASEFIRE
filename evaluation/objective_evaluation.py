"""
Objective / optimality-gap evaluation shared by all experiment runners.

    results, rows, J_star = run_suite(data, weights, DEFAULT_RUNS, scenario, seed)

`runs` is a list of (algorithm, topology) pairs; topology is None for
centralized / uncoordinated. Every row holds the objective components, the
physical metrics, the gap to the centralized optimum and (for ADMM) the
convergence summary.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from evaluation.convergence import convergence_summary
from evaluation.metrics import optimality_gap

DEFAULT_RUNS: Tuple[Tuple[str, Optional[str]], ...] = (
    ("uncoordinated_asap", None),
    ("uncoordinated_fcfs", None),
    ("centralized", None),
    ("shared_admm_adaptive", "star"),
    ("consensus_admm_adaptive", "random"),
)

METRIC_KEYS = ("peak_load_kw", "peak_to_average", "total_cost_inr", "avg_delay_h",
               "target_soc_pct", "mean_satisfaction", "jain_fairness",
               "total_unmet_kwh", "max_violation")


def label(algorithm: str, topology: Optional[str]) -> str:
    return algorithm if not topology else f"{algorithm} ({topology})"


def run_row(res, J_star: Optional[float] = None, **extra) -> Dict:
    row = {
        "algorithm": res.algorithm, "topology": res.topology or "-",
        "label": label(res.algorithm, res.topology),
        "scenario": res.scenario, "seed": res.seed,
        "n_evs": res.n_evs, "n_stations": res.n_stations,
        "objective": res.objective, "iterations": res.iterations,
        "converged": res.converged, "runtime_s": res.runtime_s,
        "messages": res.messages,
    }
    row.update(res.components)
    row.update({k: res.metrics.get(k) for k in METRIC_KEYS})
    if J_star is not None:
        row["gap"] = optimality_gap(res.objective, J_star)
    if res.history and "iteration" in res.history:
        row.update(convergence_summary(res))
    row.update(extra)
    return row


def run_suite(data, weights, runs: Sequence[Tuple[str, Optional[str]]] = DEFAULT_RUNS,
              scenario: Optional[str] = None, seed: Optional[int] = None,
              max_iter: int = 3000, verbose: bool = True):
    """Run every (algorithm, topology) on one instance. Centralized is always
    solved first because it provides J* for the gaps."""
    from network.integration import run_algorithm

    cen = run_algorithm("centralized", data, weights, scenario=scenario, seed=seed)
    J_star = cen.objective
    results: Dict[Tuple[str, Optional[str]], object] = {}
    rows: List[Dict] = []
    for algo, topo in runs:
        if algo == "centralized":
            r = cen
        else:
            r = run_algorithm(algo, data, weights, topology=topo or "random",
                              graph_seed=seed or 0, scenario=scenario, seed=seed,
                              max_iter=max_iter, J_star=J_star, L_star=cen.P.sum(axis=0))
        results[(algo, topo)] = r
        row = run_row(r, J_star)
        rows.append(row)
        if verbose:
            print(f"    {row['label']:34s} J={r.objective:8.4f} gap={row['gap']:.1e} "
                  f"peak={row['peak_load_kw']:7.1f}kW cost={row['total_cost_inr']:8.1f} "
                  f"delay={row['avg_delay_h']:5.2f}h target={row['target_soc_pct']:5.1f}% "
                  f"viol={row['max_violation']:.3f} t={r.runtime_s:6.1f}s", flush=True)
    return results, rows, J_star