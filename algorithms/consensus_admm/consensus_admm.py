"""
Consensus ADMM (proximal dual consensus ADMM, fixed or adaptive penalty c).

Fully decentralised: each EV i only exchanges its price vector lambda_i (M x T
numbers) with its graph neighbours, once per iteration (2|E| messages).

Per iteration, for every EV i with degree n_i (all in parallel):
    b_i      = p_i - c (n_i lambda_i + sum_{j in N(i)} lambda_j)
    R_i      = E_i v_i^- - b_i
    x_i      = prox_local( x_i^- - R_i[m(i)] / (2 c n_i tau_x) , rho = tau_x )
    w_i      = prox_aggregate( w_i^- + R_i / (2 c n_i tau_w N) , tau_w )
    lambda_i = (n_i lambda_i + sum_j lambda_j) / (2 n_i) + (E_i v_i - p_i) / (2 c n_i)
    -- exchange lambda with neighbours --
    p_i      = p_i + c (n_i lambda_i - sum_j lambda_j)
with block proximal weights tau_x = delta / (c n_i), tau_w = tau_x / N^2, delta > 1.

Stopping (all in kW; c0 = initial penalty, so nothing shrinks when c changes)
    feasibility   ||sum_i E_i v_i||                          <= eps_feas
    consensus     c0 sqrt(sum_edges ||lambda_i - lambda_j||^2) <= eps_cons
    dual change   c0 ||lambda^k - lambda^{k-1}||              <= eps_cons
    primal change ||x^k - x^{k-1}||                           <= eps_x
    physical      max constraint violation of the real schedule <= MAX_VIOLATION_KW

Adaptive penalty (residual balancing on the dual problem):
    r = consensus residual   (price disagreement between neighbours)
    s = feasibility residual (overload of the coupling constraint)
    r > mu s -> c *= tau_incr   (agree faster)
    s > mu r -> c /= tau_decr   (price reacts faster to congestion)
    c is kept in [c0/100, 100 c0], adaptation starts after WARMUP iterations
    and stops after ADAPT_UNTIL iterations (then it is fixed-c ADMM again).
Why: a large c is best when the grid limit does not bind, but when it binds
the price must rise far and it rises at a rate proportional to 1/c.

Extra diagnostics in the history: r_w (change of the load estimates) and
price_max (largest network-average price, original units).
With store_iterates=True the result also keeps, every store_every iterations,
    meta["iterates"]           the schedules X                      (N, T)
    meta["estimate_iterates"]  each EV's estimate of the total load  (N, T)
                               ("opinions", used by simulations/sim_story.py)

Same internal scaling as Shared ADMM (objective x kappa, kappa = 1/auto_rho);
settings.rho is the penalty c in ORIGINAL units ("auto" -> chosen from the graph).

Usage
    res = run_consensus_admm(data, w, topology="ring", adaptive=True, J_star=J_star)

Demo:  python -m algorithms.consensus_admm.consensus_admm
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from algorithms.base import ADMMSettings, auto_rho, load_distance
from algorithms.consensus_admm.consensus_utils import (auto_c, coupling, edge_differences,
                                                       own_rows, prox_aggregate_batch,
                                                       proximal_taus)
from algorithms.rho_update import make_rho_rule
from algorithms.stopping import MAX_VIOLATION_KW
from common.constraints import constraint_violations
from common.objective import (Weights, linear_coefficients, objective_components,
                              peak_quadratic_coeff, total_objective)
from common.result import Result
from common.utils import ProblemData, Timer
from ev.local_opt import prox_local
from evaluation.metrics import compute_metrics
from network.messaging import MessageBus
from network.topologies import Graph

WARMUP = 10            # iterations before the adaptive rule may change c
C_RANGE = 100.0        # c stays within [c0 / C_RANGE, c0 * C_RANGE]
ADAPT_UNTIL = 2000     # consensus ADMM adapts longer: the price may need hundreds of iterations to rise


class ConsensusADMM:
    def __init__(self, graph: Graph, settings: Optional[ADMMSettings] = None,
                 delta: float = 1.05, drop_prob: float = 0.0, seed: Optional[int] = None):
        if delta <= 1.0:
            raise ValueError("delta must be > 1 (convergence condition)")
        self.graph = graph
        self.settings = settings or ADMMSettings()
        self.delta = delta
        self.drop_prob = drop_prob
        self.bus_seed = seed

    @property
    def name(self) -> str:
        return "consensus_admm_adaptive" if self.settings.adaptive_rho else "consensus_admm"

    def run(self, data: ProblemData, weights: Weights, scenario: Optional[str] = None,
            seed: Optional[int] = None) -> Result:
        g, st = self.graph, self.settings
        N, M, T = data.N, data.M, data.T
        if g.N != N:
            raise ValueError(f"graph has {g.N} nodes but problem has {N} EVs")
        if not g.connected:
            raise ValueError("communication graph must be connected")
        n = g.degrees.astype(float)
        if np.any(n == 0):
            raise ValueError("every agent needs at least one neighbour")
        nb = n[:, None, None]

        # ---- internal scaling ------------------------------------------
        kappa = 1.0 / auto_rho(data, weights)
        cc = kappa * linear_coefficients(data, weights)
        a = kappa * peak_quadratic_coeff(data, weights)
        gamma = kappa * weights.gamma
        if isinstance(st.rho, str):
            if st.rho != "auto":
                raise ValueError("rho must be a number or 'auto'")
            c = auto_c(n, N, self.delta)
        else:
            c = float(st.rho) / kappa
        c0 = c
        rule = make_rho_rule(st.adaptive_rho, st.mu_consensus, st.tau_incr, st.tau_decr,
                             st.adapt_until)
        if hasattr(rule, "rho_min"):
            rule.rho_min, rule.rho_max = c0 / C_RANGE, c0 * C_RANGE
            rule.adapt_until = max(st.adapt_until, ADAPT_UNTIL)

        bus = MessageBus(g, drop_prob=self.drop_prob, seed=self.bus_seed)
        edges = np.array(g.edges, dtype=int).reshape(-1, 2)

        X = np.zeros((N, T))
        s = np.zeros(N)
        Wg = np.zeros((N, M, T))
        Lam = np.zeros((N, M, T))
        Pd = np.zeros((N, M, T))
        S_lam = np.zeros((N, M, T))          # sum of neighbours' lambda (all zero at start)

        res = Result(algorithm=self.name, topology=g.name, weights=weights.as_dict(),
                     scenario=scenario, n_evs=N, n_stations=M, seed=seed)
        iterates = []
        estimates = []                       # NEW: each EV's estimate of total grid load (for animations)
        converged = False
        r_feas = float("nan")
        k = -1

        with Timer() as tm:
            for k in range(st.max_iter):
                tau_x, tau_w = proximal_taus(c, n, N, self.delta)          # (N,), (N,)

                # ---- local primal update (EV i uses only its own state + neighbour sum)
                b = Pd - c * (nb * Lam + S_lam)
                R = coupling(data, X, Wg) - b
                x_ctr = X - (1.0 / (2.0 * c * n * tau_x))[:, None] * own_rows(data, R)
                w_ctr = Wg + (1.0 / (2.0 * c * n * tau_w * N))[:, None, None] * R
                X_old, Wg_old = X, Wg
                X, s, _ = prox_local(data, cc, x_ctr, tau_x, gamma)
                Wg = prox_aggregate_batch(data, a, w_ctr, tau_w)

                # ---- local price update -------------------------------------
                Ev = coupling(data, X, Wg)
                Lam_old = Lam
                Lam = (nb * Lam_old + S_lam) / (2.0 * nb) + (Ev - Pd) / (2.0 * c * nb)

                # ---- one neighbour exchange of lambda -----------------------
                S_lam = bus.neighbor_sum(Lam)
                Pd = Pd + c * (nb * Lam - S_lam)

                # ---- residuals (kW, independent of the current c) ----------
                load = data.station_matrix @ X
                dlam = float(np.linalg.norm(Lam - Lam_old))
                r_feas = float(np.linalg.norm(Ev.sum(axis=0)))
                r_cons = float(c0 * np.linalg.norm(edge_differences(Lam, edges)))
                r_dual = c0 * dlam
                r_x = float(np.linalg.norm(X - X_old))
                r_w = float(np.linalg.norm(Wg - Wg_old))
                viol = max(constraint_violations(data, X, s).values())
                scale_kw = max(float(np.linalg.norm(load)), float(np.linalg.norm(Wg.mean(axis=0))))
                eps_feas = math.sqrt(M * T) * st.eps_abs + st.eps_rel * scale_kw
                eps_cons = math.sqrt(N * M * T) * st.eps_abs + st.eps_rel * c0 * float(np.linalg.norm(Lam))
                eps_x = math.sqrt(N * T) * st.eps_abs + st.eps_rel * float(np.linalg.norm(X))
                done = (r_feas <= eps_feas and r_cons <= eps_cons and r_dual <= eps_cons
                        and r_x <= eps_x and viol <= MAX_VIOLATION_KW)

                if k % st.log_every == 0 or done or k == st.max_iter - 1:
                    J = total_objective(data, X, s, weights)
                    log = dict(iteration=k, objective=J, r_primal=r_feas, r_dual=r_dual,
                               r_consensus=r_cons, r_x=r_x, r_w=r_w,
                               eps_pri=eps_feas, eps_dual=eps_cons,
                               rho=c * kappa, messages=bus.messages, max_violation=viol,
                               price_max=float(Lam.mean(axis=0).max()) / kappa)
                    if st.J_star is not None:
                        log["gap"] = abs(J - st.J_star) / max(abs(st.J_star), 1e-12)
                    if st.L_star is not None:
                        log["load_dist"] = load_distance(X, st.L_star)
                    res.log(**log)
                    if st.verbose and k % 100 == 0:
                        print(f"[{self.name}/{g.name}] k={k:5d} J={J:.6f} feas={r_feas:.2e}/{eps_feas:.1e} "
                              f"cons={r_cons:.2e} dual={r_dual:.2e}/{eps_cons:.1e} dx={r_x:.2e}/{eps_x:.1e} "
                              f"viol={viol:.2f} c={c * kappa:.2e}")
                if st.store_iterates and k % st.store_every == 0:
                    iterates.append(X.copy())
                    estimates.append(Wg.sum(axis=1).copy())    # NEW: (N, T) EV i's view of total load

                if done:
                    converged = True
                    break

                # grid overload (r_feas) >> price disagreement (r_cons) -> lower c (price rises faster)
                # price disagreement >> grid overload               -> raise c (agree faster)
                if k >= WARMUP:
                    c = rule.update(k, c, r_cons, r_feas)

        J = total_objective(data, X, s, weights)
        res.finalize(X, s, J, objective_components(data, X, s),
                     converged=converged, iterations=k + 1)
        res.runtime_s = tm.elapsed
        res.messages, res.floats_sent = bus.messages, bus.floats
        res.metrics = compute_metrics(data, X, s, weights)
        res.meta.update(
            rho_initial=c0 * kappa, rho_final=c * kappa, kappa=kappa, delta=self.delta,
            n_rho_changes=getattr(rule, "n_changes", 0),
            price_mean=(Lam.mean(axis=0) / kappa).tolist(),          # agreed price signal (M x T)
            price_disagreement=float(np.max(np.abs(Lam - Lam.mean(axis=0)))) / kappa,
            load_estimate_disagreement=float(np.max(np.abs(Wg - Wg.mean(axis=0)))),
            final_feasibility_kw=r_feas,
            dropped_messages=bus.dropped,
            settings={k_: v for k_, v in vars(st).items() if k_ != "L_star"},
        )
        if st.store_iterates:
            res.meta["iterates"] = [x.tolist() for x in iterates]
            res.meta["estimate_iterates"] = [e.tolist() for e in estimates]   # NEW
        return res


def run_consensus_admm(data: ProblemData, weights: Weights, topology: str = "random",
                       adaptive: bool = False, settings: Optional[ADMMSettings] = None,
                       graph: Optional[Graph] = None, graph_seed: int = 0,
                       scenario: Optional[str] = None, seed: Optional[int] = None,
                       delta: float = 1.05, drop_prob: float = 0.0, **overrides) -> Result:
    from network.integration import make_network
    if graph is None:
        graph = make_network(topology, data.N, seed=graph_seed)
    st = (settings or ADMMSettings()).with_(adaptive_rho=adaptive, **overrides)
    return ConsensusADMM(graph, st, delta=delta, drop_prob=drop_prob).run(
        data, weights, scenario=scenario, seed=seed)


if __name__ == "__main__":
    from algorithms.shared_admm.shared_admm import run_shared_admm
    from centralized.solver import solve_centralized
    from common.objective import WEIGHT_CASES
    from ev.fleet import scenario_instance

    data, _ = scenario_instance("morning_peak", n_evs=30, n_stations=3, seed=0)
    w = WEIGHT_CASES["baseline"]
    cen = solve_centralized(data, w)
    J_star, L_star = cen.objective, cen.P.sum(axis=0)
    print(f"J* = {J_star:.6f}\n")

    def show(tag, r):
        print(f"{tag:34s} iters={r.iterations:5d} conv={r.converged!s:5s} "
              f"gap={abs(r.objective - J_star) / abs(J_star):.2e} "
              f"viol={r.metrics['max_violation']:.3f}kW t={r.runtime_s:6.2f}s "
              f"msgs={r.messages:8d} MB={r.floats_sent * 8 / 1e6:8.2f} "
              f"c {r.meta['rho_initial']:.1e}->{r.meta['rho_final']:.1e}")

    show("shared ADMM adaptive (hub)", run_shared_admm(data, w, adaptive=True, J_star=J_star))
    for topo in ("ring", "star", "mesh", "random"):
        for adaptive in (False, True):
            tag = f"consensus {'adaptive' if adaptive else 'fixed   '} {topo}"
            show(tag, run_consensus_admm(data, w, topology=topo, adaptive=adaptive,
                                         max_iter=5000, J_star=J_star, L_star=L_star))