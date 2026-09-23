"""Sequential container: chains caller-provided layers in registration order."""

from __future__ import annotations

from . import checkpoint as _checkpoint
from .tensor import Tensor

_REQUIRED_CAPABILITIES = ("forward", "backward", "parameters")


def _zeros(shape):
    if not shape:
        return 0.0
    return [_zeros(shape[1:]) for _ in range(shape[0])]


def _layer_kind(module):
    kind = getattr(module, "checkpoint_kind", None)
    if kind is None:
        kind = type(module).__name__
    if not isinstance(kind, str) or not kind:
        raise ValueError("layer checkpoint_kind must be a non-empty string")
    return kind


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

    Besides the forward/backward cycle the container offers:

    * ``update(learning_rate)`` -- one in-place gradient step
      ``theta <- theta - lr * grad`` on every parameter; gradients are
      not cleared (``zero_grad`` still does that).
    * ``save(target)`` / ``load(source)`` -- atomic, versioned snapshots of
      parameters, accumulated gradients and slice-boundary hidden state,
      to a filesystem path or to an in-memory ``bytearray``.
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
        self._last_output = None
        self._last_hidden = None

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
        self._last_output = x
        self._last_hidden = new_hidden
        self._pending_backward = True
        return x, new_hidden

    def backward(self, loss):
        if not self._pending_backward:
            raise RuntimeError(
                "backward() requires a preceding forward() and may only be "
                "called once per forward()"
            )
        upstream = self._parse_loss(loss)
        # Snapshot gradients first: if any layer raises mid-pass, roll the
        # partial accumulation back so the state is exactly as if backward
        # had never run and the caller may retry the same forward's backward
        # (that retry is not a second backward pass).
        grad_snapshot = [
            None if param.grad is None else param.grad.tolist()
            for param in self.parameters()
        ]
        try:
            for module in reversed(self._modules):
                upstream = module.backward(upstream)
        except BaseException:
            for param, saved in zip(self.parameters(), grad_snapshot):
                param.grad = None if saved is None else Tensor(saved)
            raise
        self._pending_backward = False

    def _parse_loss(self, loss):
        if isinstance(loss, bool):
            raise ValueError("loss must be numeric, not a boolean")
        if isinstance(loss, (int, float, Tensor)):
            return loss
        if isinstance(loss, tuple):
            if len(loss) != 2:
                raise ValueError(
                    "tuple loss must be (output, upstream): exactly two elements"
                )
            output, upstream = loss
            if output is not self._last_output:
                raise ValueError(
                    "tuple loss must carry the Tensor returned by the "
                    "preceding forward() call"
                )
            if isinstance(upstream, bool) or not isinstance(
                upstream, (int, float, Tensor)
            ):
                raise ValueError(
                    "tuple loss upstream must be a number or a Tensor"
                )
            return upstream
        raise ValueError(
            "loss must be a scalar, a Tensor, or an (output, upstream) tuple"
        )

    def zero_grad(self):
        for param in self.parameters():
            param.grad = Tensor(_zeros(param.shape))

    def parameters(self):
        params = []
        for module in self._modules:
            params.extend(module.parameters())
        return params

    def update(self, learning_rate):
        """Perform one in-place step ``theta <- theta - learning_rate * grad``.

        Every parameter is updated in fixed registration order using its
        accumulated gradient; parameters without a gradient slot are left
        untouched (their gradient is implicitly zero). Gradients are not
        cleared -- ``zero_grad`` remains the only way to reset them.
        """
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float)
        ):
            raise ValueError("learning rate must be a number")
        if learning_rate != learning_rate or learning_rate in (
            float("inf"),
            float("-inf"),
        ):
            raise ValueError("learning rate must be finite")
        for param in self.parameters():
            if param.grad is not None:
                param._scaled_subtract_(param.grad, learning_rate)

    # -- checkpoints --------------------------------------------------------

    def save(self, target):
        """Fix the full training state into *target*.

        *target* is either a filesystem path (``str``/``os.PathLike``) or a
        ``bytearray`` used as an in-memory buffer. The snapshot records
        every parameter tensor, every accumulated gradient (zeros for
        parameters that have never received one), the current
        slice-boundary hidden state, the layer order with per-layer
        parameter shapes and the format version.

        A snapshot may only be taken at a segment boundary -- i.e. when no
        backward pass is pending -- because a forward's intermediate
        activations live inside the caller's layers and cannot be fixed by
        the engine. Saving mutates no existing state and repeated saves
        produce identical bytes.
        """
        if self._pending_backward:
            raise RuntimeError(
                "save() requires a completed segment: call backward() for the "
                "pending forward() first"
            )
        params = self.parameters()
        document = {
            "params": [
                {"s": param.shape, "v": param.tolist()} for param in params
            ],
            "grads": [
                {
                    "s": param.shape,
                    "v": param.grad.tolist()
                    if param.grad is not None
                    else _zeros(param.shape),
                }
                for param in params
            ],
            "hidden": None
            if self._last_hidden is None
            else [
                {"s": slot.shape, "v": slot.tolist()} for slot in self._last_hidden
            ],
            "layers": [
                {
                    "kind": _layer_kind(module),
                    "shapes": [param.shape for param in module.parameters()],
                }
                for module in self._modules
            ],
            "pending": False,
        }
        return _checkpoint.save_bytes(document, target)

    def load(self, source):
        """Restore state previously written by ``save``.

        *source* is a filesystem path or a bytes-like buffer. The whole
        checkpoint is validated before anything is applied: the format
        version, every tensor shape and the layer order (count, kinds and
        per-layer parameter shapes) must match this container exactly. A
        missing path raises ``FileNotFoundError``; any structural problem
        rejects the entire checkpoint with ``ValueError`` and leaves the
        container untouched.

        On success returns the restored slice-boundary hidden-state list
        (one tensor per layer), or ``None`` when the checkpoint fixed the
        start-of-training state. Feed it back into the next
        ``forward(batch, hidden)`` to continue the sequence.
        """
        document = _checkpoint.load_bytes(source)
        self._validate_against_model(document)

        params = self.parameters()
        for param, entry in zip(params, document["params"]):
            param._set_values(_rebuild_tree(entry["v"], entry["s"]))
        for param, entry in zip(params, document["grads"]):
            param.grad = Tensor(_rebuild_tree(entry["v"], entry["s"]))
        if document["hidden"] is None:
            self._last_hidden = None
            self._hidden_shapes = None
            restored_hidden = None
        else:
            slots = [
                Tensor(_rebuild_tree(entry["v"], entry["s"]))
                for entry in document["hidden"]
            ]
            self._last_hidden = slots
            self._hidden_shapes = [slot.shape for slot in slots]
            restored_hidden = [Tensor(slot.tolist()) for slot in slots]
        self._last_output = None
        self._pending_backward = False
        return restored_hidden

    def _validate_against_model(self, document):
        if not isinstance(document, dict):
            raise ValueError("checkpoint is not a valid state document")
        params = self.parameters()
        saved_params = document.get("params")
        saved_grads = document.get("grads")
        if (
            not isinstance(saved_params, list)
            or not isinstance(saved_grads, list)
            or len(saved_params) != len(params)
            or len(saved_grads) != len(params)
        ):
            raise ValueError(
                "checkpoint parameter count does not match the current model"
            )
        for index, (param, p_entry, g_entry) in enumerate(
            zip(params, saved_params, saved_grads)
        ):
            if not isinstance(p_entry, dict) or not isinstance(g_entry, dict):
                raise ValueError(f"checkpoint parameter {index} is malformed")
            if p_entry.get("s") != param.shape or g_entry.get("s") != param.shape:
                raise ValueError(
                    f"checkpoint parameter {index} shape does not match the "
                    f"current model"
                )

        saved_layers = document.get("layers")
        if not isinstance(saved_layers, list) or len(saved_layers) != len(
            self._modules
        ):
            raise ValueError(
                "checkpoint layer count does not match the current model"
            )
        offset = 0
        for index, (module, saved) in enumerate(zip(self._modules, saved_layers)):
            if not isinstance(saved, dict):
                raise ValueError(f"checkpoint layer {index} is malformed")
            if saved.get("kind") != _layer_kind(module):
                raise ValueError(
                    f"checkpoint layer {index} kind {saved.get('kind')!r} does "
                    f"not match the current model"
                )
            current_shapes = [p.shape for p in module.parameters()]
            if saved.get("shapes") != current_shapes:
                raise ValueError(
                    f"checkpoint layer {index} parameter shapes do not match "
                    f"the current model"
                )
            offset += len(current_shapes)
        if offset != len(params):
            raise ValueError("checkpoint layer order does not match parameters")

        saved_hidden = document.get("hidden")
        if saved_hidden is not None:
            if not isinstance(saved_hidden, list) or len(saved_hidden) != len(
                self._modules
            ):
                raise ValueError(
                    "checkpoint hidden slot count does not match the current model"
                )
            for index, entry in enumerate(saved_hidden):
                if not isinstance(entry, dict) or "s" not in entry:
                    raise ValueError(f"checkpoint hidden slot {index} is malformed")
                if self._hidden_shapes is not None and (
                    entry["s"] != self._hidden_shapes[index]
                ):
                    raise ValueError(
                        f"checkpoint hidden slot {index} shape does not match "
                        f"the current model"
                    )

        if document.get("pending") is not False:
            raise ValueError("checkpoint pending flag has an unexpected value")

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


def _rebuild_tree(value, shape):
    """Validate decoded leaves against *shape* and return a fresh nested tree."""
    if shape:
        if not isinstance(value, list) or len(value) != shape[0]:
            raise ValueError("checkpoint tensor values do not match their shape")
        return [_rebuild_tree(item, shape[1:]) for item in value]
    if isinstance(value, (list, bool)) or not isinstance(value, (int, float)):
        raise ValueError("checkpoint tensor values do not match their shape")
    return value
