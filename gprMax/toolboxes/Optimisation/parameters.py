"""Describe the variables the optimiser may change.

Users put Real, Integer and Categorical definitions in a dictionary; see
examples/start_here.py. Values reach build_model/evaluate in these same units.
This module validates bounds; it does not choose mesh sizes or build geometry.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral
from numbers import Real as RealNumber
from typing import Any, Mapping


def _number(value, name):
    """Accept a finite numerical value; reject booleans even though Python treats them as integers."""
    if isinstance(value, bool) or not isinstance(value, RealNumber) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class Real:
    """A continuous variable between lower and upper, inclusive.

    Example: Real(0.10, 0.18, "m") for a length supplied to the model in metres.
    ``unit`` is a label, not a conversion. ``scale="log"`` suits positive values
    spanning orders of magnitude. The model receives physical values, not logs.
    """

    lower: float
    upper: float
    unit: str = ""
    scale: str = "linear"

    def __post_init__(self):
        """Validate bounds and scaling once, when the search variable is declared."""
        lo, hi = _number(self.lower, "lower"), _number(self.upper, "upper")
        if lo >= hi or self.scale not in ("linear", "log") or (self.scale == "log" and lo <= 0):
            raise ValueError("Real requires increasing bounds and linear or positive log scaling")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    def validate(self, value):
        """Check a proposed physical value without scaling, snapping or unit conversion."""
        value = _number(value, "value")
        if not self.lower <= value <= self.upper:
            raise ValueError(f"value must lie in [{self.lower}, {self.upper}]")
        return value

    def from_unit(self, value):
        """Decode a unit coordinate; ordinary candidates already use physical units."""
        u = _number(value, "unit coordinate")
        if not 0 <= u <= 1:
            raise ValueError("unit coordinate must lie in [0, 1]")
        if u in (0, 1):
            return self.lower if u == 0 else self.upper
        if self.scale == "log":
            return math.exp((1 - u) * math.log(self.lower) + u * math.log(self.upper))
        return (1 - u) * self.lower + u * self.upper


@dataclass(frozen=True)
class Integer:
    """An integer from ``lower, lower + step, ..., upper``, inclusive.

    Example: Integer(50, 80) for a cell count. This type does not itself mean
    cells: the user decides whether an integer represents cells, layers, etc.
    ``Integer(26, 36, "mm", step=2)`` supplies 26, 28, ..., 36 millimetres.
    Both bounds must lie on this lattice. The default step is one.
    """

    lower: int
    upper: int
    unit: str = ""
    step: int = 1

    def __post_init__(self):
        """Validate inclusive integer bounds; equal bounds represent a fixed value."""
        if any(
            isinstance(x, bool) or not isinstance(x, Integral) for x in (self.lower, self.upper)
        ):
            raise ValueError("Integer bounds must be integers")
        if self.lower > self.upper:
            raise ValueError("Integer bounds must be ordered")
        if isinstance(self.step, bool) or not isinstance(self.step, Integral) or self.step < 1:
            raise ValueError("Integer step must be a positive integer")
        if (self.upper - self.lower) % self.step:
            raise ValueError("Integer upper bound must equal lower plus a whole number of steps")
        object.__setattr__(self, "lower", int(self.lower))
        object.__setattr__(self, "upper", int(self.upper))
        object.__setattr__(self, "step", int(self.step))

    def validate(self, value):
        """Require an allowed integer; do not silently snap an invalid candidate."""
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not self.lower <= value <= self.upper
        ):
            raise ValueError(f"value must be an integer in [{self.lower}, {self.upper}]")
        if (value - self.lower) % self.step:
            raise ValueError(f"value must follow step {self.step} from {self.lower}")
        return int(value)

    def from_unit(self, value):
        """Round a unit coordinate to the nearest allowed value, with ties upward.

        This is optimiser decoding in the declared units. It does not change
        the simulation mesh. A model receives the resulting physical integer.
        """
        u = _number(value, "unit coordinate")
        if not 0 <= u <= 1:
            raise ValueError("unit coordinate must lie in [0, 1]")
        intervals = (self.upper - self.lower) // self.step
        # Multiplying a represented half-step can land one ULP below a tie.
        index = math.floor(math.nextafter(u * intervals + 0.5, math.inf))
        return self.lower + min(intervals, index) * self.step


@dataclass(frozen=True)
class Categorical:
    """One string from a tuple, e.g. Categorical(("sand", "clay")).

    Your model maps the chosen name to its material/geometry settings.
    The RF and TPE adapters support categorical parameters.
    """

    choices: tuple[str, ...]

    def __post_init__(self):
        """Store a unique, nonempty catalogue of string choices."""
        values = tuple(self.choices)
        if (
            isinstance(self.choices, str)
            or not values
            or any(not isinstance(x, str) for x in values)
        ):
            raise ValueError("Categorical choices must be a nonempty sequence of strings")
        if len(set(values)) != len(values):
            raise ValueError("Categorical choices must be unique")
        object.__setattr__(self, "choices", values)

    def validate(self, value):
        """Require one of the declared choices without interpreting its physical meaning."""
        if not isinstance(value, str) or value not in self.choices:
            raise ValueError(f"value must be one of {self.choices}")
        return value


class ParameterSpace:
    """Advanced container for a parameter dictionary and its validation rules.

    optimise(parameters=...) creates this automatically. Most users only need
    the dictionary; custom adapters and Campaign code use this container.
    """

    def __init__(self, parameters: Mapping[str, Real | Integer | Categorical]):
        """Keep the declared parameter order and reject unsupported variable definitions."""
        # Insertion order becomes the coordinate order in numerical adapters.
        # The user and builder continue to use names, so they need not track indices.
        self._parameters = dict(parameters)
        if not self._parameters:
            raise ValueError("At least one parameter is required")
        for name, definition in self._parameters.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Parameter names must be nonempty strings")
            if not isinstance(definition, (Real, Integer, Categorical)):
                raise TypeError(f"Unsupported parameter definition for {name}")

    def validate(self, candidate: Mapping[str, Any]) -> dict:
        """Return a validated copy of a complete candidate dictionary.

        Every declared variable must be supplied, even when several variables
        change together. Missing or extra keys usually indicate a mismatch
        between the optimiser adapter and the model's parameter declaration.
        """
        if not isinstance(candidate, Mapping):
            raise TypeError("A candidate must be a mapping of physical parameter values")
        missing = set(self._parameters) - set(candidate)
        extra = set(candidate) - set(self._parameters)
        if missing or extra:
            raise ValueError(
                f"Parameter names mismatch: missing={sorted(missing)}, extra={list(extra)}"
            )
        result = {}
        for name, definition in self._parameters.items():
            try:
                result[name] = definition.validate(candidate[name])
            except ValueError as exc:
                raise ValueError(f"Parameter {name}: {exc}") from exc
        return result

    def to_dict(self):
        """Describe bounds, types, units and scales for adapters and campaign records."""
        return {
            name: {"type": type(value).__name__, **asdict(value)}
            for name, value in self._parameters.items()
        }
