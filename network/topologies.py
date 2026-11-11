"""
Communication topologies between EV agents (node i = EV i).

    ring    i <-> i+1 (mod N)                    degree 2
    star    node `hub` (default 0) <-> everyone  hub degree N-1, others 1
    mesh    complete graph                       degree N-1
    random  Erdos-Renyi G(N, p), resampled until connected
            default p = min(1, 2 ln N / N)  (about twice the connectivity threshold)

Usage
    g = make_topology("ring", 50)
    g.neighbors[3], g.degrees, g.metropolis, g.summary()

Demo:  python -m network.topologies
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import numpy as np

from network import graph_utils as gu


@dataclass
class Graph:
    name: str
    A: np.ndarray                      # (N, N) bool adjacency
    hub: Optional[int] = None          # only for star
    params: Dict = field(default_factory=dict)

    def __post_init__(self):
        A = np.asarray(self.A, dtype=bool)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        if not np.array_equal(A, A.T):
            raise ValueError("A must be symmetric")
        if np.any(np.diag(A)):
            raise ValueError("A must have a zero diagonal")
        self.A = A

    # ---- basic structure ----------------------------------------------
    @property
    def N(self) -> int:
        return self.A.shape[0]

    @cached_property
    def edges(self) -> List[Tuple[int, int]]:
        return gu.edges_from_adjacency(self.A)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @cached_property
    def neighbors(self) -> List[np.ndarray]:
        return gu.neighbors(self.A)

    @cached_property
    def degrees(self) -> np.ndarray:
        return gu.degrees(self.A)

    # ---- spectral / distance properties --------------------------------
    @cached_property
    def laplacian(self) -> np.ndarray:
        return gu.laplacian(self.A)

    @cached_property
    def metropolis(self) -> np.ndarray:
        return gu.metropolis_weights(self.A)

    @cached_property
    def lambda2(self) -> float:
        return gu.algebraic_connectivity(self.A)

    @cached_property
    def slem(self) -> float:
        return gu.slem(self.metropolis)

    @cached_property
    def diameter(self) -> float:
        return gu.diameter(self.A)

    @cached_property
    def connected(self) -> bool:
        return gu.is_connected(self.A)

    @cached_property
    def incidence(self) -> np.ndarray:
        return gu.incidence(self.A)

    def summary(self) -> Dict[str, float]:
        return {
            "topology": self.name, "N": self.N, "n_edges": self.n_edges,
            "avg_degree": float(self.degrees.mean()) if self.N else 0.0,
            "max_degree": int(self.degrees.max()) if self.N else 0,
            "diameter": self.diameter, "lambda2": self.lambda2,
            "slem": self.slem, "connected": self.connected,
        }

    def to_networkx(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(self.N))
        G.add_edges_from(self.edges)
        return G


# ----------------------------------------------------------------------
def ring(N: int) -> Graph:
    edges = [(i, (i + 1) % N) for i in range(N)] if N > 2 else ([(0, 1)] if N == 2 else [])
    return Graph("ring", gu.adjacency_from_edges(N, edges))


def star(N: int, hub: int = 0) -> Graph:
    edges = [(hub, j) for j in range(N) if j != hub]
    return Graph("star", gu.adjacency_from_edges(N, edges), hub=hub)


def mesh(N: int) -> Graph:
    A = ~np.eye(N, dtype=bool)
    return Graph("mesh", A)


def random_graph(N: int, p: Optional[float] = None, seed: int = 0,
                 max_tries: int = 1000) -> Graph:
    if N <= 1:
        return Graph("random", np.zeros((N, N), dtype=bool), params={"p": p})
    if p is None:
        p = min(1.0, 2.0 * math.log(N) / N)
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        U = np.triu(rng.random((N, N)) < p, k=1)
        A = U | U.T
        if gu.is_connected(A):
            return Graph("random", A, params={"p": p, "seed": seed})
    raise RuntimeError(f"no connected G({N}, {p}) found in {max_tries} tries; increase p")


TOPOLOGY_BUILDERS = {"ring": ring, "star": star, "mesh": mesh, "random": random_graph}


def make_topology(name: str, N: int, seed: int = 0, **kwargs) -> Graph:
    if name not in TOPOLOGY_BUILDERS:
        raise ValueError(f"unknown topology '{name}', choose from {list(TOPOLOGY_BUILDERS)}")
    if name == "random":
        return random_graph(N, seed=seed, **kwargs)
    return TOPOLOGY_BUILDERS[name](N, **kwargs)


if __name__ == "__main__":
    for N in (10, 50, 200):
        for name in TOPOLOGY_BUILDERS:
            s = make_topology(name, N).summary()
            print(f"N={N:4d} {name:7s} edges={s['n_edges']:6d} avg_deg={s['avg_degree']:6.2f} "
                  f"diam={s['diameter']:5.0f} lambda2={s['lambda2']:8.4f} slem={s['slem']:.4f}")