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


def _in_place_step(data, grad_data, lr):
    """Apply ``data <- data - lr * grad_data`` to nested lists in place."""
    for index, item in enumerate(data):
        grad_item = grad_data[index]
        if isinstance(item, list):
            _in_place_step(item, grad_item, lr)
        else:
            data[index] = item - lr * grad_item


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

    def _step_in_place(self, grad, lr):
        """Update every element by ``theta <- theta - lr * g`` in place.

        ``grad`` must be a :class:`Tensor` with the same shape.  The tensor
        object identity (and the layer's hold on it) is preserved.
        """
        if grad._shape != self._shape:
            raise ValueError("gradient shape must match parameter shape")
        if self._shape:
            _in_place_step(self._data, grad._data, lr)
        else:
            self._data = self._data - lr * grad._data

    @classmethod
    def _from_parts(cls, shape, data):
        """Build a tensor from already-validated nested lists (internal)."""
        obj = object.__new__(cls)
        obj._shape = shape
        obj._data = data
        obj.grad = None
        return obj

    def _replace_data(self, data, shape):
        """Restore validated values produced by the checkpoint codec."""
        self._data = data
        self._shape = shape

    def __repr__(self):
        return f"Tensor(shape={self._shape}, data={self._data!r})"
