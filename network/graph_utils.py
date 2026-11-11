"""
Graph helpers on a boolean adjacency matrix A (N x N, symmetric, zero diagonal).
NumPy only.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, List, Tuple

import numpy as np


def adjacency_from_edges(N: int, edges: Iterable[Tuple[int, int]]) -> np.ndarray:
    A = np.zeros((N, N), dtype=bool)
    for i, j in edges:
        if i != j:
            A[i, j] = A[j, i] = True
    return A


def edges_from_adjacency(A: np.ndarray) -> List[Tuple[int, int]]:
    I, J = np.nonzero(np.triu(A, k=1))
    return list(zip(I.tolist(), J.tolist()))


def neighbors(A: np.ndarray) -> List[np.ndarray]:
    return [np.flatnonzero(A[i]) for i in range(A.shape[0])]


def degrees(A: np.ndarray) -> np.ndarray:
    return A.sum(axis=1).astype(int)


def bfs_distances(A: np.ndarray, source: int) -> np.ndarray:
    N = A.shape[0]
    dist = np.full(N, -1)
    dist[source] = 0
    q = deque([source])
    nbrs = neighbors(A)
    while q:
        i = q.popleft()
        for j in nbrs[i]:
            if dist[j] < 0:
                dist[j] = dist[i] + 1
                q.append(j)
    return dist


def is_connected(A: np.ndarray) -> bool:
    N = A.shape[0]
    return True if N <= 1 else bool(np.all(bfs_distances(A, 0) >= 0))


def diameter(A: np.ndarray) -> float:
    N = A.shape[0]
    if N <= 1:
        return 0.0
    best = 0
    for s in range(N):
        d = bfs_distances(A, s)
        if np.any(d < 0):
            return float("inf")
        best = max(best, int(d.max()))
    return float(best)


def laplacian(A: np.ndarray) -> np.ndarray:
    Af = A.astype(float)
    return np.diag(Af.sum(axis=1)) - Af


def algebraic_connectivity(A: np.ndarray) -> float:
    """lambda_2 of the Laplacian (0 if disconnected). Larger = better connected."""
    if A.shape[0] <= 1:
        return 0.0
    return float(np.sort(np.linalg.eigvalsh(laplacian(A)))[1])


def metropolis_weights(A: np.ndarray) -> np.ndarray:
    """Symmetric, doubly-stochastic averaging matrix:
         W_ij = 1 / (1 + max(d_i, d_j))  for neighbours,  W_ii = 1 - sum_j W_ij.
    Repeated x <- W x converges to the average on any connected graph."""
    d = degrees(A)
    W = np.where(A, 1.0 / (1.0 + np.maximum(d[:, None], d[None, :])), 0.0)
    W[np.diag_indices_from(W)] = 1.0 - W.sum(axis=1)
    return W


def slem(W: np.ndarray) -> float:
    """Second largest eigenvalue modulus of W: averaging error shrinks ~ slem^k."""
    if W.shape[0] <= 1:
        return 0.0
    vals = np.sort(np.linalg.eigvalsh(W))
    return float(max(abs(vals[0]), abs(vals[-2])))


def incidence(A: np.ndarray) -> np.ndarray:
    """Oriented edge-node incidence matrix B (|E| x N): row e = +1 at i, -1 at j."""
    E = edges_from_adjacency(A)
    B = np.zeros((len(E), A.shape[0]))
    for e, (i, j) in enumerate(E):
        B[e, i], B[e, j] = 1.0, -1.0
    return B