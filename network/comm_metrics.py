"""
Communication metrics: graph properties and communication cost of a run.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from network.topologies import Graph

BYTES_PER_FLOAT = 8


def graph_metrics(graph: Graph) -> Dict[str, float]:
    return graph.summary()


def comm_summary(messages: int, floats: int, iterations: int, n_agents: int) -> Dict[str, float]:
    it = max(iterations, 1)
    return {
        "messages_total": int(messages),
        "messages_per_iter": messages / it,
        "messages_per_agent_per_iter": messages / it / max(n_agents, 1),
        "floats_total": int(floats),
        "MB_total": floats * BYTES_PER_FLOAT / 1e6,
    }


def result_comm_summary(result, graph: Optional[Graph] = None) -> Dict[str, float]:
    """Same as comm_summary for a common.result.Result."""
    out = comm_summary(result.messages, result.floats_sent, result.iterations, result.n_evs)
    if graph is not None:
        out["max_node_degree"] = int(graph.degrees.max())
    return out


def max_node_load(graph: Graph, floats_per_message: int) -> int:
    """Floats sent + received by the busiest node in one neighbour exchange
    (the star hub is the bottleneck)."""
    return int(2 * graph.degrees.max() * floats_per_message)


def gossip_rounds_estimate(graph: Graph, tol: float) -> float:
    """Rounds needed to shrink the averaging error by `tol`: log(tol)/log(slem)."""
    s = graph.slem
    if s <= 0:
        return 1.0
    if s >= 1:
        return float("inf")
    return max(1.0, math.log(tol) / math.log(s))