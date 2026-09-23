"""Minimal n-dimensional numeric tensor (CPU, standard library only)."""

from __future__ import annotations

import math


def _normalize(data):
    """Validate *data* and return ``(shape, deep-copied nested lists)``."""
    if isinstance(data, bool):
        raise ValueError("Tensor data must be numeric; booleans are not allowed")
    if isinstance(data, (int, float)):
        return [], data
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("Tensor data must not be empty")
        shape = None
        items = []
        for item in data:
            sub_shape, value = _normalize(item)
            if shape is None:
                shape = sub_shape
            elif sub_shape != shape:
                raise ValueError(
                    "Tensor data is ragged: nested sequences must all have the same shape"
                )
            items.append(value)
        return [len(items)] + shape, items
    raise ValueError(
        "Tensor data must be a number or a nested sequence of numbers, "
        f"got {type(data).__name__}"
    )


def _deep_copy(value):
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


class Tensor:
    """An n-dimensional container of numbers with a ``grad`` slot.

    ``grad`` starts as ``None``; layers accumulate gradients onto it and
    ``Sequential.zero_grad()`` resets it to an all-zeros tensor.
    """

    __slots__ = ("_data", "_shape", "grad")

    def __init__(self, data):
        self._shape, self._data = _normalize(data)
        self.grad = None

    @property
    def shape(self):
        return list(self._shape)

    def tolist(self):
        return _deep_copy(self._data)

    def __repr__(self):
        return f"Tensor(shape={self._shape}, data={self._data!r})"

    # -- internal engine hooks (not part of the public README surface) ------

    def _values(self):
        """Return the raw nested storage (not a copy)."""
        return self._data

    def _set_values(self, values):
        """Replace the raw nested storage with an already-validated tree."""
        self._data = values

    def _scaled_subtract_(self, other, scale):
        """Subtract *scale* times *other* in place, element by element.

        The arithmetic happens in a fixed left-to-right order so the
        result is identical to an uninterrupted run.
        """
        self._data = _scaled_subtract(self._data, other._data, float(scale))

    def _all_finite(self):
        return _all_finite(self._data)


def _scaled_subtract(a, b, scale):
    if isinstance(a, list):
        return [_scaled_subtract(x, y, scale) for x, y in zip(a, b)]
    return a - scale * b


def _all_finite(value):
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return isinstance(value, int) or (
        isinstance(value, float) and math.isfinite(value)
    )
