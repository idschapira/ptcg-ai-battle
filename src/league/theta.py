"""Theta: the parameter vector that IS the genome of the league.

A deck module declares a `ThetaSchema` (an ordered list of `ParamSpec`s:
name, default, bounds, integrality). A `Theta` is one immutable point in
that space — mapping-like for readable rule code (`theta["low_deck"]`)
and vector-like for the Phase 2 mutation operators (`to_vector` /
`from_vector`, bounds always enforced).

Two invariants matter downstream:

1. DEFAULTS ARE EXACT. `schema.defaults()` stores the declared float
   verbatim, so a module whose defaults are the shipped constants
   reproduces the shipped agent bit-for-bit (see the CrustleAgent
   equivalence test) — no rounding, no rescaling, no normalization.
2. EVERY Theta IS IN BOUNDS. Construction clips; mutation therefore
   cannot walk a rule out of its safe band (e.g. a score that must stay
   below the attach band).

JSON round-trips by NAME, not position, so adding a parameter to a
schema does not invalidate stored Hall of Fame entries: unknown names
are dropped, missing names fall back to the default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final, Iterator, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One tunable knob: its default, its legal band, its meaning."""

    name: str
    default: float
    low: float
    high: float
    integral: bool = False
    doc: str = ""

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"{self.name}: low {self.low} > high {self.high}")
        if not (self.low <= self.default <= self.high):
            raise ValueError(
                f"{self.name}: default {self.default} outside "
                f"[{self.low}, {self.high}]")

    def clip(self, value: float) -> float:
        """Clamp into the legal band; integral specs snap to whole numbers.

        The default is returned VERBATIM when it survives clipping, which
        is what keeps `defaults()` bit-exact against the shipped code."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return self.default
        if number != number:  # NaN
            return self.default
        if self.integral:
            number = float(round(number))
        if number <= self.low:
            return self.low
        if number >= self.high:
            return self.high
        return number


class ThetaSchema:
    """An ordered, name-indexed set of ParamSpecs (one per deck module)."""

    __slots__ = ("_specs", "_by_name")

    def __init__(self, specs: Sequence[ParamSpec] = ()) -> None:
        self._specs: tuple[ParamSpec, ...] = tuple(specs)
        self._by_name: dict[str, int] = {}
        for i, spec in enumerate(self._specs):
            if spec.name in self._by_name:
                raise ValueError(f"duplicate parameter {spec.name!r}")
            self._by_name[spec.name] = i

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __iter__(self) -> Iterator[ParamSpec]:
        return iter(self._specs)

    @property
    def specs(self) -> tuple[ParamSpec, ...]:
        return self._specs

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs)

    def spec(self, name: str) -> ParamSpec:
        return self._specs[self._by_name[name]]

    def index_of(self, name: str) -> int:
        return self._by_name[name]

    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Per-position (low, high) — the search box for Phase 2."""
        return tuple((spec.low, spec.high) for spec in self._specs)

    def defaults(self) -> "Theta":
        return Theta(self, tuple(spec.default for spec in self._specs))

    def from_vector(self, vector: Sequence[float]) -> "Theta":
        """Positional construction (mutation output); clipped, padded with
        defaults, and truncated — never raises on a wrong-length vector."""
        values = []
        for i, spec in enumerate(self._specs):
            raw = vector[i] if i < len(vector) else spec.default
            values.append(spec.clip(raw))
        return Theta(self, tuple(values))

    def from_dict(self, mapping: Mapping[str, Any] | None) -> "Theta":
        """Name-keyed construction: unknown keys ignored, missing keys
        default. This is what makes stored genomes forward-compatible."""
        source = mapping or {}
        values = []
        for spec in self._specs:
            raw = source.get(spec.name, spec.default)
            values.append(spec.clip(raw))
        return Theta(self, tuple(values))


class Theta(Mapping[str, float]):
    """One immutable, in-bounds point in a ThetaSchema's space."""

    __slots__ = ("_schema", "_values")

    def __init__(self, schema: ThetaSchema, values: Sequence[float]) -> None:
        if len(values) != len(schema):
            raise ValueError(
                f"expected {len(schema)} values, got {len(values)}")
        self._schema = schema
        self._values: tuple[float, ...] = tuple(float(v) for v in values)

    # ---- mapping protocol (readable rule code) ---- #

    def __getitem__(self, name: str) -> float:
        return self._values[self._schema.index_of(name)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._schema.names)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Theta):
            return (self._schema is other._schema
                    and self._values == other._values)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((id(self._schema), self._values))

    def __repr__(self) -> str:
        inner = ", ".join(f"{n}={v:g}" for n, v in zip(self._schema.names,
                                                       self._values))
        return f"Theta({inner})"

    # ---- typed accessors ---- #

    @property
    def schema(self) -> ThetaSchema:
        return self._schema

    def i(self, name: str) -> int:
        """Integer view of a knob used as a count/threshold."""
        return int(self[name])

    # ---- vector protocol (Phase 2 mutation) ---- #

    def to_vector(self) -> tuple[float, ...]:
        return self._values

    def replace(self, **overrides: float) -> "Theta":
        """A new Theta with some knobs moved (clipped into their bands).

        Unknown names raise: a typo in a mutation operator must fail loud
        rather than silently evolve nothing."""
        values = list(self._values)
        for name, value in overrides.items():
            if name not in self._schema:
                raise KeyError(f"unknown parameter {name!r}")
            i = self._schema.index_of(name)
            values[i] = self._schema.specs[i].clip(value)
        return Theta(self._schema, tuple(values))

    def distance(self, other: "Theta") -> float:
        """Band-normalized L1 distance — how far apart two genomes are
        in units of their own legal ranges (diversity metric for Phase 2)."""
        total = 0.0
        for spec, mine, theirs in zip(self._schema.specs, self._values,
                                      other.to_vector()):
            span = spec.high - spec.low
            total += abs(mine - theirs) / span if span else 0.0
        return total

    # ---- persistence ---- #

    def to_dict(self) -> dict[str, float]:
        return dict(zip(self._schema.names, self._values))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def diff_from_defaults(self) -> dict[str, tuple[float, float]]:
        """{name: (default, current)} for knobs that actually moved —
        the readable summary of what evolution changed."""
        out: dict[str, tuple[float, float]] = {}
        for spec, value in zip(self._schema.specs, self._values):
            if value != spec.default:
                out[spec.name] = (spec.default, value)
        return out


EMPTY_SCHEMA: Final[ThetaSchema] = ThetaSchema(())

__all__ = ["ParamSpec", "Theta", "ThetaSchema", "EMPTY_SCHEMA"]
