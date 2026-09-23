"""Minimal n-dimensional numeric tensor (CPU, standard library only)."""

from __future__ import annotations


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
