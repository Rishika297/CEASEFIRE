import numpy as np
import pytest

from network.averaging import GossipAverager, HubAverager
from network.integration import make_averager, make_network


@pytest.fixture
def X():
    return np.random.default_rng(0).normal(5, 3, (30, 2, 48))     # (N, M, T) like xbar inputs


def test_hub_exact(X):
    est, st = HubAverager(make_network("star", 30)).average(X)
    assert np.allclose(est, X.mean(0)[None]) and st.messages == 2 * 29 and st.exact


@pytest.mark.parametrize("name", ["ring", "mesh", "random", "star"])
def test_gossip_converges(name, X):
    est, st = GossipAverager(make_network(name, 30), tol=1e-8, max_rounds=20000).average(X)
    assert st.max_error < 1e-5
    assert est.shape == X.shape


def test_gossip_preserves_mean(X):
    g = make_network("ring", 30)
    avg = GossipAverager(g, tol=0.0, max_rounds=5)       # exactly 5 rounds
    est, st = avg.average(X)
    assert st.rounds == 5 and st.messages == 5 * 2 * g.n_edges
    assert np.allclose(est.mean(0), X.mean(0))            # doubly stochastic -> mean kept


def test_ring_needs_more_rounds_than_mesh(X):
    r = GossipAverager(make_network("ring", 30), tol=1e-6, max_rounds=50000).average(X)[1]
    m = GossipAverager(make_network("mesh", 30), tol=1e-6, max_rounds=50000).average(X)[1]
    assert r.rounds > m.rounds


def test_make_averager_auto():
    assert isinstance(make_averager(make_network("star", 10)), HubAverager)
    assert isinstance(make_averager(make_network("ring", 10)), GossipAverager)