"""
Minimal discrete-time Signal Temporal Logic (STL) with quantitative semantics.

A *signal* is a 1-D NumPy array sampled at the integer steps of the
simulation; a *trace* is a dict  {signal name -> array}.

Grammar
    phi ::= mu | not phi | phi and phi | phi or phi
          | G_[a,b] phi | F_[a,b] phi
    mu  ::= x >= c   |   x <= c        (c a scalar or a per-step array)

Robustness rho(phi, s, t)  (Donze & Maler 2010, discrete time)
    rho(x >= c, t)         = (x[t] - c[t]) / scale
    rho(x <= c, t)         = (c[t] - x[t]) / scale
    rho(not phi, t)        = -rho(phi, t)
    rho(phi1 and phi2, t)  = min(rho(phi1, t), rho(phi2, t))
    rho(phi1 or  phi2, t)  = max(rho(phi1, t), rho(phi2, t))
    rho(G_[a,b] phi, t)    = min_{tau in [t+a, t+b]} rho(phi, tau)
    rho(F_[a,b] phi, t)    = max_{tau in [t+a, t+b]} rho(phi, tau)

rho > 0  => the trace satisfies phi, rho < 0 => it violates phi, and |rho| is
how far the signal is from the boundary (the satisfaction margin).
Windows are clipped to the signal length; an empty window gives +inf for G
(vacuously true) and -inf for F (never happens).

Example
    phi = Always(Predicate("P_total", "<=", 100.0), 0, 47)
    phi.rho({"P_total": L})        # = min_t (100 - L[t])
"""
from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np

Trace = Dict[str, np.ndarray]
Threshold = Union[float, np.ndarray]


class Formula:
    """Base class. Subclasses implement rho(trace, t)."""

    def rho(self, trace: Trace, t: int = 0) -> float:
        raise NotImplementedError

    def sat(self, trace: Trace, t: int = 0, tol: float = 0.0) -> bool:
        """Boolean semantics with a numerical tolerance: rho >= -tol."""
        return self.rho(trace, t) >= -tol

    def rho_trace(self, trace: Trace, ts: Sequence[int]) -> np.ndarray:
        """rho(phi, t) for every t in ts (robustness signal of the formula)."""
        return np.array([self.rho(trace, int(t)) for t in ts], dtype=float)

    def horizon(self, trace: Trace) -> int:
        """Length of the shortest signal this formula reads."""
        raise NotImplementedError

    # operator sugar:  phi & psi, phi | psi, ~phi
    def __and__(self, other: "Formula") -> "And":
        return And(self, other)

    def __or__(self, other: "Formula") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)


# ----------------------------------------------------------------------
class Predicate(Formula):
    """Atomic predicate  signal >= c  or  signal <= c.
    ``scale`` divides the robustness, e.g. scale=capacity gives a dimensionless
    margin so predicates in different units can be combined with And."""

    def __init__(self, signal: str, op: str, c: Threshold, scale: Threshold = 1.0,
                 label: str | None = None):
        if op not in (">=", "<="):
            raise ValueError("op must be '>=' or '<='")
        self.signal, self.op, self.c, self.scale = signal, op, c, scale
        self.label = label

    @staticmethod
    def _at(v: Threshold, t: int) -> float:
        return float(v[t]) if np.ndim(v) else float(v)

    def rho(self, trace: Trace, t: int = 0) -> float:
        x = float(trace[self.signal][t])
        c, k = self._at(self.c, t), self._at(self.scale, t)
        return ((x - c) if self.op == ">=" else (c - x)) / k

    def margin_signal(self, trace: Trace) -> np.ndarray:
        """Vectorised rho over all t (fast path used by the monitor)."""
        x = np.asarray(trace[self.signal], dtype=float)
        c = np.broadcast_to(np.asarray(self.c, dtype=float), x.shape) if np.ndim(self.c) == 0 \
            else np.asarray(self.c, dtype=float)[: x.size]
        k = np.asarray(self.scale, dtype=float)
        k = k if k.ndim == 0 else k[: x.size]
        return ((x - c) if self.op == ">=" else (c - x)) / k

    def horizon(self, trace: Trace) -> int:
        return len(trace[self.signal])

    def __str__(self) -> str:
        if self.label:
            return self.label
        c = "c(t)" if np.ndim(self.c) else f"{self.c:g}"
        return f"({self.signal} {self.op} {c})"


class Not(Formula):
    def __init__(self, phi: Formula):
        self.phi = phi

    def rho(self, trace: Trace, t: int = 0) -> float:
        return -self.phi.rho(trace, t)

    def horizon(self, trace: Trace) -> int:
        return self.phi.horizon(trace)

    def __str__(self) -> str:
        return f"¬{self.phi}"


class And(Formula):
    def __init__(self, *phis: Formula):
        if not phis:
            raise ValueError("And needs at least one operand")
        self.phis = phis

    def rho(self, trace: Trace, t: int = 0) -> float:
        return min(p.rho(trace, t) for p in self.phis)

    def horizon(self, trace: Trace) -> int:
        return min(p.horizon(trace) for p in self.phis)

    def __str__(self) -> str:
        if len(self.phis) > 4:
            return f"AND_{{{len(self.phis)} terms}}({self.phis[0]}, ...)"
        return "(" + " ∧ ".join(str(p) for p in self.phis) + ")"


class Or(Formula):
    def __init__(self, *phis: Formula):
        if not phis:
            raise ValueError("Or needs at least one operand")
        self.phis = phis

    def rho(self, trace: Trace, t: int = 0) -> float:
        return max(p.rho(trace, t) for p in self.phis)

    def horizon(self, trace: Trace) -> int:
        return min(p.horizon(trace) for p in self.phis)

    def __str__(self) -> str:
        return "(" + " ∨ ".join(str(p) for p in self.phis) + ")"


class _Temporal(Formula):
    symbol = "?"
    reduce = staticmethod(min)
    empty = np.inf

    def __init__(self, phi: Formula, a: int, b: int):
        if a < 0 or b < a:
            raise ValueError("need 0 <= a <= b")
        self.phi, self.a, self.b = phi, int(a), int(b)

    def window(self, trace: Trace, t: int) -> range:
        end = min(t + self.b, self.phi.horizon(trace) - 1)
        return range(t + self.a, end + 1)

    def rho(self, trace: Trace, t: int = 0) -> float:
        w = self.window(trace, t)
        if len(w) == 0:
            return float(self.empty)
        if isinstance(self.phi, Predicate):                       # fast path
            m = self.phi.margin_signal(trace)[w.start:w.stop]
            return float(m.min() if self.reduce is min else m.max())
        return float(self.reduce(self.phi.rho(trace, tau) for tau in w))

    def critical_time(self, trace: Trace, t: int = 0) -> int | None:
        """Step tau in the window that determines rho (argmin for G, argmax for F)."""
        w = self.window(trace, t)
        if len(w) == 0:
            return None
        vals = np.array([self.phi.rho(trace, tau) for tau in w])
        k = int(np.argmin(vals) if self.reduce is min else np.argmax(vals))
        return w.start + k

    def horizon(self, trace: Trace) -> int:
        return self.phi.horizon(trace)

    def __str__(self) -> str:
        return f"{self.symbol}_[{self.a},{self.b}] {self.phi}"


class Always(_Temporal):
    """G_[a,b] phi : phi holds at every step of the window."""
    symbol = "G"
    reduce = staticmethod(min)
    empty = np.inf


class Eventually(_Temporal):
    """F_[a,b] phi : phi holds at some step of the window."""
    symbol = "F"
    reduce = staticmethod(max)
    empty = -np.inf
