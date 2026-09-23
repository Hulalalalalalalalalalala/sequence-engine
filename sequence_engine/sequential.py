"""Sequential container: chains caller-provided layers in registration order."""

from __future__ import annotations

from .tensor import Tensor

_REQUIRED_CAPABILITIES = ("forward", "backward", "parameters")


def _zeros(shape):
    if not shape:
        return 0.0
    return [_zeros(shape[1:]) for _ in range(shape[0])]


class Sequential:
    """Chains caller-provided layers in registration order.

    Each layer must provide ``forward``, ``backward`` and ``parameters``.
    ``forward(batch, hidden=None)`` runs the batch through every layer in
    order; each layer consumes the previous layer's output plus its own
    hidden-state slot (``None`` when no hidden state is supplied, meaning
    the layer restarts from zeros) and returns an ``(output, hidden_slot)``
    pair. The slots stack in registration order into the chain's hidden
    state.

    The engine keeps no gradient history across ``forward`` calls: hidden
    states carry values only, so a backward pass on a later segment can
    never add gradients to an earlier one. ``backward(loss)`` starts from
    the caller-computed scalar loss and walks the layers in reverse order,
    each layer passing the upstream gradient back and accumulating
    parameter gradients on its own tensors.
    """

    def __init__(self, modules):
        if isinstance(modules, (str, bytes)):
            raise ValueError("Sequential expects a non-empty sequence of layers")
        try:
            modules = list(modules)
        except TypeError:
            raise ValueError(
                "Sequential expects a non-empty sequence of layers"
            ) from None
        if len(modules) == 0:
            raise ValueError("Sequential requires at least one layer")
        for index, module in enumerate(modules):
            missing = [
                name
                for name in _REQUIRED_CAPABILITIES
                if not callable(getattr(module, name, None))
            ]
            if missing:
                raise ValueError(
                    f"layer at index {index} is missing required "
                    f"capabilit{'ies' if len(missing) > 1 else 'y'}: "
                    + ", ".join(missing)
                )
        self._modules = modules
        self._pending_backward = False
        self._hidden_shapes = None

    def forward(self, batch, hidden=None):
        if not isinstance(batch, Tensor):
            raise ValueError("batch must be a Tensor")
        batch_shape = batch.shape
        if len(batch_shape) == 0 or batch_shape[0] <= 0:
            raise ValueError("batch must be non-empty")
        slots = self._prepare_hidden(hidden)
        x = batch
        new_hidden = []
        for module, slot in zip(self._modules, slots):
            result = module.forward(x, slot)
            try:
                x, slot_out = result
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "each layer's forward must return an (output, hidden) pair"
                ) from exc
            if not isinstance(x, Tensor) or not isinstance(slot_out, Tensor):
                raise ValueError("each layer's forward must return (Tensor, Tensor)")
            new_hidden.append(slot_out)
        self._hidden_shapes = [slot.shape for slot in new_hidden]
        self._pending_backward = True
        return x, new_hidden

    def backward(self, loss):
        if not self._pending_backward:
            raise RuntimeError(
                "backward() requires a preceding forward() and may only be "
                "called once per forward()"
            )
        upstream = loss
        for module in reversed(self._modules):
            upstream = module.backward(upstream)
        self._pending_backward = False

    def zero_grad(self):
        for param in self.parameters():
            param.grad = Tensor(_zeros(param.shape))

    def parameters(self):
        params = []
        for module in self._modules:
            params.extend(module.parameters())
        return params

    def _prepare_hidden(self, hidden):
        count = len(self._modules)
        if hidden is None:
            return [None] * count
        if not isinstance(hidden, (list, tuple)) or len(hidden) != count:
            raise ValueError(
                f"hidden must be a list of {count} tensors (one slot per layer)"
            )
        slots = list(hidden)
        for index, slot in enumerate(slots):
            if slot is None:
                continue
            if not isinstance(slot, Tensor):
                raise ValueError(f"hidden slot {index} must be a Tensor")
            expected = self._hidden_shapes[index] if self._hidden_shapes else None
            if expected is not None and slot.shape != expected:
                raise ValueError(
                    f"hidden slot {index} has shape {slot.shape}, expected {expected}"
                )
        return slots
