"""
Neighbour-to-neighbour message passing with message counting.
Used by Consensus ADMM (Step 5): every iteration each agent sends its local
vector to all its neighbours.

    bus = MessageBus(graph)
    S = bus.neighbor_sum(X)         # S[i] = sum_{j in N(i)} X[j]   (counts 2|E| messages)
    bus.messages, bus.floats

Optional link failures (robustness experiments): each directed message is
dropped independently with probability drop_prob.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from network.topologies import Graph


class MessageBus:
    def __init__(self, graph: Graph, drop_prob: float = 0.0, seed: Optional[int] = None):
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.graph = graph
        self.A = graph.A.astype(float)
        self.drop_prob = drop_prob
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.messages = 0
        self.floats = 0
        self.dropped = 0

    def _links(self) -> np.ndarray:
        if self.drop_prob == 0.0:
            return self.A
        keep = self.rng.random(self.A.shape) >= self.drop_prob
        L = self.A * keep
        self.dropped += int(self.A.sum() - L.sum())
        return L

    def neighbor_sum(self, X: np.ndarray) -> np.ndarray:
        """Every agent sends X[i] to each neighbour; returns what each agent receives, summed."""
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        flat = X.reshape(N, -1)
        L = self._links()
        sent = int(self.A.sum())                      # = 2 |E|
        self.messages += sent
        self.floats += sent * flat.shape[1]
        return (L @ flat).reshape(X.shape)

    def received_counts(self) -> np.ndarray:
        """Degree of each node (number of messages it receives per exchange, no drops)."""
        return self.A.sum(axis=1)