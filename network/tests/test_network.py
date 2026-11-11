import numpy as np
import pytest

from network.graph_utils import incidence
from network.messaging import MessageBus
from network.topologies import make_topology, mesh, random_graph, ring, star

N = 20


def test_ring():
    g = ring(N)
    assert np.all(g.degrees == 2) and g.n_edges == N and g.connected
    assert g.diameter == N // 2


def test_star():
    g = star(N)
    assert g.degrees[0] == N - 1 and np.all(g.degrees[1:] == 1)
    assert g.n_edges == N - 1 and g.diameter == 2


def test_mesh():
    g = mesh(N)
    assert np.all(g.degrees == N - 1) and g.n_edges == N * (N - 1) // 2 and g.diameter == 1


@pytest.mark.parametrize("seed", range(5))
def test_random_connected_and_reproducible(seed):
    g1, g2 = random_graph(N, seed=seed), random_graph(N, seed=seed)
    assert g1.connected and np.array_equal(g1.A, g2.A)


def test_connectivity_order():
    l = {name: make_topology(name, N).lambda2 for name in ("ring", "star", "mesh", "random")}
    assert l["mesh"] > l["random"] > l["ring"]


@pytest.mark.parametrize("name", ["ring", "star", "mesh", "random"])
def test_metropolis_doubly_stochastic(name):
    W = make_topology(name, N).metropolis
    assert np.allclose(W, W.T) and np.allclose(W.sum(1), 1) and np.all(W >= -1e-12)


def test_small_graphs():
    assert ring(2).n_edges == 1 and ring(1).n_edges == 0
    assert make_topology("random", 1).N == 1


def test_incidence_rows():
    B = incidence(ring(5).A)
    assert B.shape == (5, 5) and np.allclose(B.sum(1), 0)


def test_message_bus_counts_and_sums():
    g = ring(6)
    bus = MessageBus(g)
    X = np.arange(6, dtype=float)[:, None] * np.ones((6, 4))
    S = bus.neighbor_sum(X)
    assert S[0, 0] == X[1, 0] + X[5, 0]
    assert bus.messages == 2 * g.n_edges and bus.floats == 2 * g.n_edges * 4


def test_message_bus_drops():
    bus = MessageBus(mesh(10), drop_prob=0.5, seed=0)
    bus.neighbor_sum(np.ones((10, 3)))
    assert bus.dropped > 0