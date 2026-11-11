"""
Common output structure returned by every algorithm (uncoordinated,
centralized, all ADMM variants), so experiments and plots treat them the same.

Usage inside an iterative algorithm
    res = Result(algorithm="shared_admm", topology="ring", ...)
    for k in range(max_iter):
        ...
        res.log(iteration=k, objective=J, r_primal=rp, r_dual=rd, rho=rho, messages=msgs)
    res.finalize(P, s, objective=J, components=comps, converged=True)
    res.save(results_dir("shared_admm") / "ring_seed0")
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Result:
    algorithm: str
    P: Optional[np.ndarray] = None                 # (N, T) charging powers [kW]
    s: Optional[np.ndarray] = None                 # (N,) unmet energy [kWh]
    objective: float = float("nan")
    components: Dict[str, float] = field(default_factory=dict)   # J_cost, J_peak, ...
    weights: Dict[str, float] = field(default_factory=dict)      # alpha, beta, gamma

    topology: Optional[str] = None
    scenario: Optional[str] = None
    n_evs: int = 0
    n_stations: int = 0
    seed: Optional[int] = None

    iterations: int = 0
    converged: bool = False
    runtime_s: float = 0.0
    messages: int = 0                              # total messages sent (communication overhead)
    floats_sent: int = 0                           # total scalars sent

    metrics: Dict[str, float] = field(default_factory=dict)      # evaluation.metrics output
    history: Dict[str, List[float]] = field(default_factory=dict)  # per-iteration logs
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def log(self, **values: float) -> None:
        """Append one value per key to the per-iteration history."""
        for k, v in values.items():
            self.history.setdefault(k, []).append(float(v))

    def finalize(self, P: np.ndarray, s: np.ndarray, objective: float,
                 components: Optional[Dict[str, float]] = None,
                 converged: bool = True, iterations: Optional[int] = None) -> "Result":
        self.P = np.asarray(P, dtype=float)
        self.s = np.asarray(s, dtype=float)
        self.objective = float(objective)
        if components is not None:
            self.components = {k: float(v) for k, v in components.items()}
        self.converged = bool(converged)
        if iterations is not None:
            self.iterations = int(iterations)
        elif self.history:
            self.iterations = len(next(iter(self.history.values())))
        if self.P.ndim == 2:
            self.n_evs = self.P.shape[0]
        return self

    # ------------------------------------------------------------------
    def summary_row(self) -> Dict[str, Any]:
        """Flat dict of scalars for a results table (pandas.DataFrame([...]))."""
        row = {
            "algorithm": self.algorithm, "topology": self.topology, "scenario": self.scenario,
            "n_evs": self.n_evs, "n_stations": self.n_stations, "seed": self.seed,
            "objective": self.objective, "iterations": self.iterations,
            "converged": self.converged, "runtime_s": self.runtime_s,
            "messages": self.messages, "floats_sent": self.floats_sent,
        }
        row.update({f"w_{k}": v for k, v in self.weights.items()})
        row.update(self.components)
        row.update(self.metrics)
        return row

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Writes <path>.npz holding P, s and all other fields as JSON."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        info = asdict(self)
        info.pop("P"), info.pop("s")
        np.savez_compressed(
            path,
            P=np.array([]) if self.P is None else self.P,
            s=np.array([]) if self.s is None else self.s,
            info=np.array(json.dumps(info, default=_json_default)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Result":
        with np.load(Path(path).with_suffix(".npz"), allow_pickle=False) as z:
            info = json.loads(str(z["info"]))
            P, s = z["P"], z["s"]
        return cls(P=P if P.size else None, s=s if s.size else None, **info)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))
