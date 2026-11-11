"""
Network-wide averaging, needed by Shared ADMM on non-star graphs
(every EV needs xbar = average of all EVs' contributions).

HubAverager      star topology: every EV sends its value to the hub, hub sends
                 the exact mean back.  2(N-1) messages, exact.
GossipAverager   any connected graph: repeat x <- W x with Metropolis weights
                 until neighbours (almost) agree.  2|E| messages per round,
                 inexact (error ~ slem^rounds).

Both return (estimates, stats): estimates[i] is agent i's estimate of the mean.

Stopping rule of GossipAverager (uses only local information):
    max over edges (i, j) of |x_i - x_j|  <=  tol * (max |x| + 1e-12)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from network.topologies import Graph


@dataclass
class AveragingStats:
    rounds: int
    messages: int
    floats: int
    max_error: float          # vs the exact mean (for analysis only; agents do not know it)
    exact: bool


class HubAverager:
    exact = True

    def __init__(self, graph: Graph):
        if graph.hub is None:
            raise ValueError("HubAverager needs a star graph (graph.hub set)")
        self.graph = graph

    def average(self, X: np.ndarray) -> Tuple[np.ndarray, AveragingStats]:
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        D = int(np.prod(X.shape[1:])) if X.ndim > 1 else 1
        est = np.broadcast_to(X.mean(axis=0), X.shape).copy()
        msgs = 2 * (N - 1)                       # leaves -> hub, hub -> leaves
        return est, AveragingStats(rounds=2, messages=msgs, floats=msgs * D,
                                   max_error=0.0, exact=True)


class GossipAverager:
    exact = False

    def __init__(self, graph: Graph, tol: float = 1e-6, max_rounds: int = 2000,
                 min_rounds: int = 1):
        if not graph.connected:
            raise ValueError("graph must be connected for averaging")
        self.graph = graph
        self.W = graph.metropolis
        self.tol = tol
        self.max_rounds = max_rounds
        self.min_rounds = min_rounds
        E = np.array(graph.edges, dtype=int).reshape(-1, 2)
        self._ei, self._ej = E[:, 0], E[:, 1]

    def _disagreement(self, Y: np.ndarray) -> float:
        if self._ei.size == 0:
            return 0.0
        return float(np.max(np.abs(Y[self._ei] - Y[self._ej])))

    def average(self, X: np.ndarray) -> Tuple[np.ndarray, AveragingStats]:
        X = np.asarray(X, dtype=float)
        shape = X.shape
        N = shape[0]
        Y = X.reshape(N, -1).copy()
        D = Y.shape[1]
        scale = float(np.max(np.abs(Y))) + 1e-12
        rounds = 0
        while rounds < self.max_rounds:
            if rounds >= self.min_rounds and self._disagreement(Y) <= self.tol * scale:
                break
            Y = self.W @ Y
            rounds += 1
        msgs = 2 * self.graph.n_edges * rounds
        exact_mean = X.reshape(N, -1).mean(axis=0)
        err = float(np.max(np.abs(Y - exact_mean[None, :]))) if N else 0.0
        return Y.reshape(shape), AveragingStats(rounds=rounds, messages=msgs,
                                                floats=msgs * D, max_error=err, exact=False)