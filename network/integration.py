"""
Glue between the network module and the algorithms.

    g   = make_network("random", 50, seed=0)
    avg = make_averager(g)                                   # hub for star, gossip otherwise
    res = run_algorithm("consensus_admm", data, w, topology="ring", J_star=J_star)

Algorithms: centralized, uncoordinated_asap, uncoordinated_fcfs,
            shared_admm, shared_admm_adaptive, consensus_admm, consensus_admm_adaptive

Demo:  python -m network.integration
"""
from __future__ import annotations

from typing import Iterable, Optional

from common.utils import load_yaml
from network.averaging import GossipAverager, HubAverager
from network.comm_metrics import gossip_rounds_estimate
from network.topologies import Graph, make_topology

ADMM_ALGORITHMS = ("shared_admm", "shared_admm_adaptive", "consensus_admm_adaptive")
RUNNABLE_ADMM = ADMM_ALGORITHMS + ("consensus_admm",)

def load_network_config(path: str = "configs/network_config.yaml") -> dict:
    return load_yaml(path).get("network", {})


def make_network(topology: str, N: int, seed: int = 0, **kwargs) -> Graph:
    return make_topology(topology, N, seed=seed, **kwargs)


def make_averager(graph: Graph, mode: str = "auto", tol: float = 1e-6, max_rounds: int = 2000):
    """mode: 'auto' (hub for star, gossip otherwise) | 'hub' | 'gossip'."""
    if mode == "hub" or (mode == "auto" and graph.name == "star"):
        return HubAverager(graph)
    if mode in ("gossip", "auto"):
        return GossipAverager(graph, tol=tol, max_rounds=max_rounds)
    raise ValueError("mode must be 'auto', 'hub' or 'gossip'")


def run_algorithm(algorithm: str, data, weights, topology: str = "random",
                  graph: Optional[Graph] = None, graph_seed: int = 0,
                  scenario: Optional[str] = None, seed: Optional[int] = None,
                  settings=None, **overrides):
    """Run any algorithm on any topology and return a common.result.Result.
    ADMM settings come from configs/admm_config.yaml unless `settings` is given;
    keyword overrides (max_iter=..., J_star=..., rho=...) are applied on top."""
    if algorithm == "centralized":
        from centralized.solver import solve_centralized
        return solve_centralized(data, weights, scenario=scenario, seed=seed)
    if algorithm.startswith("uncoordinated"):
        from centralized.uncoordinated import run_uncoordinated
        mode = algorithm.split("_", 1)[1] if "_" in algorithm else "asap"
        return run_uncoordinated(data, weights, mode, scenario=scenario, seed=seed)
    if algorithm not in RUNNABLE_ADMM:
        raise ValueError(f"unknown algorithm '{algorithm}'")

    from algorithms.base import ADMMSettings
    cfg = load_network_config()
    if graph is None:
        kw = {}
        if topology == "random" and cfg.get("random_p") is not None:
            kw["p"] = cfg["random_p"]
        graph = make_network(topology, data.N, seed=graph_seed, **kw)
    settings = settings or ADMMSettings.load()
    adaptive = algorithm.endswith("_adaptive")

    if algorithm.startswith("shared_admm"):
        from algorithms.shared_admm.shared_admm import run_shared_admm
        a = cfg.get("averaging", {})
        avg = make_averager(graph, a.get("mode", "auto"), float(a.get("tol", 1e-6)),
                            int(a.get("max_rounds", 2000)))
        return run_shared_admm(data, weights, adaptive=adaptive, settings=settings,
                               topology=graph.name, averager=avg,
                               scenario=scenario, seed=seed, **overrides)

    from algorithms.consensus_admm.consensus_admm import run_consensus_admm
    return run_consensus_admm(data, weights, adaptive=adaptive, settings=settings,
                              graph=graph, scenario=scenario, seed=seed,
                              drop_prob=float(cfg.get("link_drop_prob", 0.0)), **overrides)


def network_table(sizes: Iterable[int] = (10, 25, 50, 100, 200),
                  topologies: Iterable[str] = ("ring", "star", "mesh", "random"),
                  seed: int = 0, tol: float = 1e-6):
    import pandas as pd
    rows = []
    for N in sizes:
        for name in topologies:
            g = make_network(name, N, seed)
            row = g.summary()
            row["gossip_rounds_est"] = gossip_rounds_estimate(g, tol)
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 200)
    print(network_table().round(4).to_string(index=False))