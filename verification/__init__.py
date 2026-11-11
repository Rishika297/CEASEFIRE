"""
STL specifications & verification layer.

ADMM / centralized / uncoordinated produce charging schedules; this package
checks whether a schedule satisfies the temporal requirements:

    Req 1  F_[a_i, d_i] (SoC_i >= SoC_i^target)   every EV reaches its target before departure
    Req 2  G_[0, T-1]  (P_total <= P_grid)         total power never exceeds grid capacity

    from verification import verify_schedule
    rep = verify_schedule(data, P, algorithm="centralized")
    print(rep.text())
"""
from importlib import import_module

_EXPORTS = {
    "Formula": "stl", "Predicate": "stl", "Not": "stl", "And": "stl", "Or": "stl",
    "Always": "stl", "Eventually": "stl",
    "build_trace": "stl_specifications", "build_specifications": "stl_specifications",
    "soc_target_spec": "stl_specifications", "grid_capacity_spec": "stl_specifications",
    "verify_schedule": "stl_monitor", "verify_result": "stl_monitor",
    "stl_metrics": "stl_monitor", "STLReport": "stl_monitor",
}
__all__ = list(_EXPORTS)


def __getattr__(name):          # lazy, so `python -m verification.stl_monitor` runs cleanly
    if name in _EXPORTS:
        return getattr(import_module(f"verification.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module 'verification' has no attribute '{name}'")
