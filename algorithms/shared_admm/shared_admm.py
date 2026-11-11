"""
Shared ADMM (fixed or adaptive rho) for distributed EV charging.

Two communication modes
    averager=None   a coordinator (outside the EV graph) averages exactly:
                    2N messages of T numbers per iteration.        topology="hub"
    averager=...    the average is computed on the EV communication graph by
                    network.averaging (HubAverager for star = exact,
                    GossipAverager for ring/mesh/random = inexact). With an
                    inexact average every EV keeps its own copies of
                    xbar, zbar, u and runs the z-update locally.

Internal scaling: objective and rho multiplied by kappa = 1/auto_rho so both
residuals are in kW. Inputs (rho) and logs are in ORIGINAL units.

Usage
    res = run_shared_admm(data, w, adaptive=True, J_star=J_star)                  # coordinator
    res = run_shared_admm(data, w, averager=GossipAverager(g), topology="ring")   # on a graph

Demo:  python -m algorithms.shared_admm.shared_admm
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from algorithms.base import ADMMSettings, CommCounter, auto_rho, load_distance, resolve_rho
from algorithms.consensus_admm.consensus_utils import embed, prox_aggregate_batch
from algorithms.rho_update import make_rho_rule, rescale_scaled_dual
from algorithms.shared_admm.shared_utils import (prox_station_grid, sharing_residuals,
                                                 station_mean)
from algorithms.stopping import MAX_VIOLATION_KW,admm_tolerances, has_converged
from common.constraints import constraint_violations
from common.objective import (Weights, linear_coefficients, objective_components,
                              peak_quadratic_coeff, total_objective)
from common.result import Result
from common.utils import ProblemData, Timer
from ev.local_opt import prox_local
from evaluation.metrics import compute_metrics


def _own_rows(D: np.ndarray, station: np.ndarray) -> np.ndarray:
    """Row m(i) of agent i's copy. D is (1, M, T) (shared copy) or (N, M, T)."""
    if D.shape[0] == 1:
        return D[0][station]
    return D[np.arange(station.size), station]


class SharedADMM:
    def __init__(self, settings: Optional[ADMMSettings] = None, topology: str = "hub",
                 averager=None):
        self.settings = settings or ADMMSettings()
        self.topology = topology
        self.averager = averager

    @property
    def name(self) -> str:
        return "shared_admm_adaptive" if self.settings.adaptive_rho else "shared_admm"

    def run(self, data: ProblemData, weights: Weights, scenario: Optional[str] = None,
            seed: Optional[int] = None, P0: Optional[np.ndarray] = None) -> Result:
        st = self.settings
        N, M, T = data.N, data.M, data.T
        exact = self.averager is None or getattr(self.averager, "exact", False)
        B = 1 if exact else N                       # number of copies of xbar / zbar / u

        # ---- internal scaling ------------------------------------------
        kappa = 1.0 / auto_rho(data, weights)
        c = kappa * linear_coefficients(data, weights)
        a = kappa * peak_quadratic_coeff(data, weights)
        gamma = kappa * weights.gamma
        rho = kappa * resolve_rho(st, data, weights)
        rho0 = rho
        rule = make_rho_rule(st.adaptive_rho, st.mu, st.tau_incr, st.tau_decr, st.adapt_until)

        P = np.zeros((N, T)) if P0 is None else np.clip(P0, 0.0, data.p_upper)
        s = np.zeros(N)
        xbar = station_mean(data, P)                # exact average (for residuals / logs)
        Xb = np.repeat(xbar[None], B, axis=0)       # agents' copies
        Zb = Xb.copy()
        U = np.zeros((B, M, T))
        n_dim = N * M * T

        res = Result(algorithm=self.name, topology=self.topology, weights=weights.as_dict(),
                     scenario=scenario, n_evs=N, n_stations=M, seed=seed)
        comm = CommCounter()
        iterates = []
        converged = False
        k = -1

        with Timer() as tm:
            for k in range(st.max_iter):
                # ---- x-update: every EV in parallel, with its own copies ----
                V = P - _own_rows(Xb - Zb + U, data.station)
                P_old, xbar_old, zbar_old = P, xbar, Zb.mean(axis=0)
                P, s, _ = prox_local(data, c, V, rho, gamma)
                xbar = station_mean(data, P)

                # ---- averaging -------------------------------------------
                if self.averager is None:
                    Xb = xbar[None]
                    comm.add(2 * N, 2 * N * T)
                    rounds, avg_err = 1, 0.0
                else:
                    est, stats = self.averager.average(embed(data, P))
                    Xb = est[:1] if exact else est
                    comm.add(stats.messages, stats.floats)
                    rounds, avg_err = stats.rounds, stats.max_error

                # ---- z-update and dual update (locally on every copy) -------
                W = U + Xb
                if B == 1:
                    Zb = prox_station_grid(data, a, W[0], rho)[None]
                else:   # same problem, batched: tau = rho / N^2, centre = N W
                    Zb = prox_aggregate_batch(data, a, N * W, np.full(B, rho / N ** 2)) / N
                U = U + Xb - Zb

                # ---- residuals (network-wide, for monitoring / stopping) ----
                zbar, u = Zb.mean(axis=0), U.mean(axis=0)
                r, sd, xn, zn, yn = sharing_residuals(data, P, P_old, xbar, zbar,
                                                      xbar_old, zbar_old, u, rho)
                eps_pri, eps_dual = admm_tolerances(n_dim, xn, zn, yn, st.eps_abs, st.eps_rel)
                viol = max(constraint_violations(data, P, s).values())
                done = has_converged(r, sd, eps_pri, eps_dual) and viol <= MAX_VIOLATION_KW
                if k % st.log_every == 0 or done or k == st.max_iter - 1:
                    J = total_objective(data, P, s, weights)
                    log = dict(iteration=k, objective=J, r_primal=r, r_dual=sd,
                               eps_pri=eps_pri, eps_dual=eps_dual, rho=rho / kappa,
                               messages=comm.messages, avg_rounds=rounds, avg_error=avg_err,
                               max_violation=viol)
                    if st.J_star is not None:
                        log["gap"] = abs(J - st.J_star) / max(abs(st.J_star), 1e-12)
                    res.log(**log)
                    if st.L_star is not None:
                        log["load_dist"] = load_distance(P, st.L_star)
                    if st.verbose and k % 100 == 0:
                        print(f"[{self.name}/{self.topology}] k={k:5d} J={J:.6f} r={r:.3e}/{eps_pri:.1e} "
                              f"s={sd:.3e}/{eps_dual:.1e} rho={rho / kappa:.2e} rounds={rounds}")
                if st.store_iterates and k % st.store_every == 0:
                    iterates.append(P.copy())

                if done:
                    converged = True
                    break

                new_rho = rule.update(k, rho, r, sd)
                if new_rho != rho:
                    U = rescale_scaled_dual(U, rho, new_rho)
                    rho = new_rho

        J = total_objective(data, P, s, weights)
        res.finalize(P, s, J, objective_components(data, P, s),
                     converged=converged, iterations=k + 1)
        res.runtime_s = tm.elapsed
        res.messages, res.floats_sent = comm.messages, comm.floats
        res.metrics = compute_metrics(data, P, s, weights)
        res.meta.update(rho_initial=rho0 / kappa, rho_final=rho / kappa, kappa=kappa,
                        n_rho_changes=getattr(rule, "n_changes", 0),
                        averaging="coordinator" if self.averager is None else type(self.averager).__name__,
                        settings={k_: v for k_, v in vars(st).items()})
        if st.store_iterates:
            res.meta["iterates"] = [x.tolist() for x in iterates]
        return res


def run_shared_admm(data: ProblemData, weights: Weights, adaptive: bool = False,
                    settings: Optional[ADMMSettings] = None, topology: str = "hub",
                    averager=None, scenario: Optional[str] = None,
                    seed: Optional[int] = None, **overrides) -> Result:
    st = (settings or ADMMSettings()).with_(adaptive_rho=adaptive, **overrides)
    return SharedADMM(st, topology, averager).run(data, weights, scenario=scenario, seed=seed)


if __name__ == "__main__":
    from centralized.solver import solve_centralized
    from common.objective import WEIGHT_CASES
    from ev.fleet import scenario_instance

    data, _ = scenario_instance("morning_peak", n_evs=30, n_stations=3, seed=0)
    w = WEIGHT_CASES["baseline"]
    J_star = solve_centralized(data, w).objective
    rho_a = resolve_rho(ADMMSettings(), data, w)
    print(f"J* (centralized) = {J_star:.6f}   auto rho = {rho_a:.3e}\n")

    def show(tag, r):
        print(f"{tag:28s} iters={r.iterations:5d} conv={r.converged!s:5s} "
              f"J={r.objective:.6f} gap={abs(r.objective - J_star) / abs(J_star):.2e} "
              f"viol={r.metrics['max_violation']:.3f} kW  t={r.runtime_s:.2f}s "
              f"msgs={r.messages}  rho_end={r.meta['rho_final']:.2e}")

    show("fixed rho (auto)", run_shared_admm(data, w, adaptive=False, J_star=J_star))
    show("adaptive rho (auto start)", run_shared_admm(data, w, adaptive=True, J_star=J_star))