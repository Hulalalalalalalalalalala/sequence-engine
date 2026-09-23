"""Sequential container with explicit truncated backpropagation.

The container wires caller-supplied layers together in registration order.
It is deliberately graph-free: a :class:`Tensor` never remembers how it was
produced.  Each layer privately caches whatever it needs for its own
backward pass, and that cache is overwritten on the next forward and
consumed by backward, so a backward pass can structurally never reach
across a chunk boundary -- only the hidden-state *values* the caller keeps
are handed on.

Layer protocol (exactly three required capabilities)
----------------------------------------------------

Each layer object provides:

* ``forward(x, h) -> (y, h_new)``: ``x`` is the input chunk, ``h`` is the
  layer's own hidden cell (``None`` on a cold start -- the layer then begins
  from an all-zero state), and ``y`` / ``h_new`` are freshly produced
  :class:`~sequence_engine.tensor.Tensor` values.  The layer may stash a
  private cache for its backward pass.
* ``backward(dy) -> dx``: propagates the upstream gradient through the
  cached pass, accumulates parameter gradients on its own parameter
  tensors, drops its private cache and returns the gradient wrt its input.
* ``parameters() -> list[Tensor]``: the layer's parameters in a stable
  order (may be empty).

The chain hidden state is a plain ``list`` of one cell per layer, stacked
in registration order.  Cells are activation tensors and therefore carry
no gradient history of their own.
"""

from .tensor import Tensor, product

_CAPABILITIES = ("forward", "backward", "parameters")


class Sequential:
    """Runs layers in registration order, one chunk at a time."""

    def __init__(self, modules):
        try:
            layers = list(modules)
        except TypeError:
            raise ValueError("Sequential requires a non-empty list of layers")
        if len(layers) == 0:
            raise ValueError("Sequential requires at least one layer")
        for index, module in enumerate(layers):
            missing = [
                name
                for name in _CAPABILITIES
                if not callable(getattr(module, name, None))
            ]
            if missing:
                raise ValueError(
                    f"layer {index} ({type(module).__name__}) is missing "
                    f"capabilities: {', '.join(missing)}"
                )
        self._modules = layers
        # Whether a forward pass is currently awaiting its one backward.
        self._pending = False
        self._output_shape = None

    # -- public API ---------------------------------------------------------

    def forward(self, batch, hidden=None):
        """Run one chunk through every layer.

        ``hidden`` is ``None`` for a cold start (every layer starts from an
        all-zero cell) or a list containing one hidden tensor per registered
        layer.  Returns ``(output, new_hidden)``; ``new_hidden`` is a fresh
        stack of value-only cells, independent of any gradient history.
        Calling ``forward`` again before ``backward`` resets the pending
        history in place.
        """
        if not isinstance(batch, Tensor):
            raise ValueError("forward expects a Tensor batch")
        shape = batch.shape
        if len(shape) == 0 or any(size == 0 for size in shape):
            raise ValueError("forward received an empty batch")

        if hidden is None:
            cells = [None] * len(self._modules)
        else:
            if not isinstance(hidden, (list, tuple)):
                raise ValueError("hidden must be a list of per-layer cells")
            if len(hidden) != len(self._modules):
                raise ValueError(
                    f"hidden has {len(hidden)} cells but the container has "
                    f"{len(self._modules)} layers"
                )
            cells = []
            for index, cell in enumerate(hidden):
                if not isinstance(cell, Tensor):
                    raise ValueError(f"hidden cell {index} must be a Tensor")
                # Only values cross the chunk boundary: hand each layer a
                # private snapshot, so no aliased tensor can carry history.
                cells.append(cell._copy())

        x = batch
        new_hidden = []
        # A new chunk starts here: any previous chunk still awaiting backward
        # is reset in place before any layer runs, so a failure partway
        # through can never leave its stale history eligible for backward.
        self._pending = False
        self._output_shape = None
        for module, h in zip(self._modules, cells):
            y, h_new = module.forward(x, h)
            if not isinstance(y, Tensor) or not isinstance(h_new, Tensor):
                raise ValueError(
                    f"{type(module).__name__}.forward must return "
                    "(output Tensor, hidden Tensor)"
                )
            new_hidden.append(h_new)
            x = y

        self._output_shape = tuple(x.shape)
        self._pending = True
        return x, new_hidden

    def backward(self, loss):
        """Back-propagate the caller-computed scalar ``loss``.

        The scalar terminal seeds a gradient of one per element of the chunk
        output (the standard scalar-node seed), and the registered layers are
        visited in reverse, each pushing the upstream gradient further back
        and leaving parameter gradients on its own tensors.  Exactly one
        ``backward`` is allowed per ``forward``; the pending history is
        consumed immediately, so the call can never add gradients to an
        earlier chunk.
        """
        if not self._pending:
            raise RuntimeError(
                "backward called without a pending forward pass"
            )
        if not isinstance(loss, Tensor):
            raise ValueError("backward expects a scalar loss Tensor")
        if len(loss._flat) != 1:
            raise ValueError("backward expects a scalar loss (one element)")

        # Consume the pass before any layer runs: a second backward call can
        # never be eligible, regardless of what a layer does below.
        self._pending = False
        output_shape = self._output_shape
        self._output_shape = None

        upstream = Tensor._from_flat([1.0] * product(output_shape), output_shape)
        for module in reversed(self._modules):
            upstream = module.backward(upstream)

    def zero_grad(self):
        """Reset accumulated gradients on every parameter tensor to zero."""
        for param in self.parameters():
            param._zero_grad()

    def parameters(self):
        """All parameters in registration order: layers in order, then each
        layer's own parameter order."""
        params = []
        for module in self._modules:
            params.extend(module.parameters())
        return params
