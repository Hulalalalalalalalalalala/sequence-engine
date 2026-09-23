"""CPU, standard-library-only multidimensional array.

A :class:`Tensor` wraps a rectangular nested ``list`` of Python floats with
an immutable shape.  Parameter tensors carry an accumulating :attr:`grad`;
activation tensors (inputs, outputs and hidden cells) carry no gradient
history at all -- this is what lets a hidden cell cross a chunk boundary
carrying numbers without ever carrying gradient tape.

Gradient accumulation is plain ``float.__add__`` in call order, so the same
sequence of gradient values always lands in the exact same accumulator.
"""


def _validate_data(data):
    """Return ``(flat_floats, shape)`` for a valid nested numeric ``list``.

    Raises :class:`ValueError` for empty data, ragged or depth-uneven
    nesting, booleans and non-numeric leaves.  Integers are accepted and
    widened to ``float``.
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Tensor data must be a non-empty nested list")

    flat = []

    def shape_of(node):
        """Recursively establish one uniform shape for this subtree."""
        if isinstance(node, list):
            if len(node) == 0:
                raise ValueError("Tensor data must not contain empty lists")
            child_shapes = {shape_of(child) for child in node}
            if len(child_shapes) != 1:
                raise ValueError("Tensor data must have a rectangular shape")
            return (len(node),) + child_shapes.pop()
        # ``bool`` is a subclass of ``int``; reject it explicitly.
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            raise ValueError(
                "Tensor leaves must be int or float, "
                f"not {type(node).__name__}"
            )
        return ()

    shape = shape_of(data)

    def flatten(node):
        if isinstance(node, list):
            for child in node:
                flatten(child)
        else:
            flat.append(float(node))

    flatten(data)
    return flat, shape


class Tensor:
    """Nested-list tensor with an optional :attr:`grad` accumulator."""

    def __init__(self, data):
        flat, shape = _validate_data(data)
        self._flat = flat
        self._shape = shape
        self._grad = None

    # -- constructors used inside the engine --------------------------------

    @classmethod
    def _from_flat(cls, flat, shape):
        """Build a tensor directly from a flat list of floats."""
        self = cls.__new__(cls)
        self._flat = list(flat)
        self._shape = tuple(shape)
        self._grad = None
        return self

    @classmethod
    def zeros(cls, shape):
        """Return an all-zero tensor with the given shape.

        A zero-sized leading dimension is allowed (it represents an empty
        batch, which the container rejects); a scalar-shaped or negatively
        sized tensor is not.
        """
        shape = tuple(shape)
        if len(shape) == 0 or any(size < 0 for size in shape):
            raise ValueError("Tensor shape must be non-empty and non-negative")
        return cls._from_flat([0.0] * product(shape), shape)

    # -- public surface (see README) ----------------------------------------

    @property
    def shape(self):
        return list(self._shape)

    def tolist(self):
        """Return a nested ``list`` copy of the tensor's values."""
        return nest(self._flat, self._shape)

    @property
    def grad(self):
        """Accumulated parameter gradient (a :class:`Tensor`) or ``None``."""
        return self._grad

    # -- gradient bookkeeping used by layers --------------------------------

    def _zero_grad(self):
        """Reset the accumulated gradient to an observable zero tensor."""
        self._grad = Tensor._from_flat(
            [0.0] * product(self._shape), self._shape
        )

    def _accumulate_grad(self, other):
        """Add ``other`` to this tensor's gradient, ordinary float addition.

        Called once per gradient contribution in call order (reverse time
        inside a chunk, then chunk after chunk across calls).
        """
        flat = other._flat
        if self._grad is None:
            self._grad = Tensor._from_flat(flat, self._shape)
        else:
            acc = self._grad._flat
            for i in range(len(acc)):
                acc[i] = acc[i] + flat[i]

    def _copy(self):
        """Independent copy of the values (never the grad)."""
        return Tensor._from_flat(self._flat, self._shape)


def product(shape):
    total = 1
    for size in shape:
        total *= size
    return total


def nest(flat, shape):
    if len(shape) == 1:
        return list(flat)
    step = product(shape[1:])
    return [
        nest(flat[i * step:(i + 1) * step], shape[1:])
        for i in range(shape[0])
    ]
