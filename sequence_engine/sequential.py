"""Sequential container: chains caller-provided layers in registration order."""

from __future__ import annotations

import math
import os

from . import checkpoint as _checkpoint
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
        self._last_hidden = new_hidden
        self._pending_backward = True
        return x, new_hidden

    def backward(self, loss):
        """Run one backward pass for the most recent forward.

        ``loss`` is the caller-computed scalar upstream seed; it may be
        given directly or wrapped as a one-element tuple ``(loss,)``.

        Raises ``ValueError`` for a malformed tuple argument regardless of
        state. Raises ``RuntimeError`` when a valid seed is given with no
        preceding forward, or when backward is called twice for one
        forward. If a layer raises while the pass is in progress the pass
        is considered not consumed: the flag stays set and a retry of
        ``backward`` is not a second call.
        """
        if isinstance(loss, tuple):
            if len(loss) != 1:
                raise ValueError(
                    "tuple loss must be a one-element tuple (loss,)"
                )
            loss = loss[0]
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

    def update(self, lr):
        """Apply one gradient step in place: ``theta <- theta - lr * g``.

        Uses each parameter's accumulated gradient; a parameter without a
        gradient yet is treated as having a zero gradient. Raises
        ``ValueError`` for a non-finite or non-numeric learning rate.
        """
        if isinstance(lr, bool) or not isinstance(lr, (int, float)):
            raise ValueError("learning rate must be a finite number")
        if not math.isfinite(lr):
            raise ValueError("learning rate must be finite")
        for param in self.parameters():
            if param.grad is None:
                continue
            param._step_in_place(param.grad, lr)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save(self, target=None):
        """Freeze the full training state into a checkpoint.

        The snapshot contains every parameter tensor, every accumulated
        gradient (parameters without a gradient are written as zeros) and
        the numeric values of the segment-boundary hidden state from the
        most recent forward.

        With a path-like *target* the bytes are committed atomically
        (temp file plus ``os.replace``; an existing file is overwritten
        without a backup) and ``None`` is returned. With *target* omitted
        the checkpoint bytes are returned for in-memory use.
        """
        document = self._build_document()
        blob = _checkpoint.encode(document)
        if target is None:
            return blob
        if isinstance(target, (str, os.PathLike)):
            _checkpoint.write_atomic(os.fspath(target), blob)
            return None
        raise ValueError(
            "save() target must be a file path or None for an in-memory buffer"
        )

    def load(self, source):
        """Restore a checkpoint produced by :meth:`save`.

        *source* is a file path or checkpoint bytes. The whole checkpoint
        is rejected (``ValueError``) when the version, fields, tensor
        shapes or the layer/parameter ordering do not match the current
        model; in that case no state is mutated. A missing path raises
        ``FileNotFoundError``.

        Returns the restored segment-boundary hidden state as a list of
        tensors (one slot per layer, ``None`` for slots the checkpoint did
        not contain), ready to pass to :meth:`forward`.
        """
        if isinstance(source, (str, os.PathLike)):
            with open(os.fspath(source), "rb") as handle:
                blob = handle.read()
        elif isinstance(source, (bytes, bytearray, memoryview)):
            blob = bytes(source)
        else:
            raise ValueError(
                "load() source must be a file path or checkpoint bytes"
            )
        document = _checkpoint.decode(blob)
        self._check_compatibility(document)
        return self._apply_document(document)

    def _build_document(self):
        layer_records = []
        for index, module in enumerate(self._modules):
            params = module.parameters()
            param_records = []
            for param in params:
                shape = param.shape
                grad_data = (
                    param.grad.tolist()
                    if param.grad is not None
                    else _zeros(shape)
                )
                param_records.append(
                    {
                        "shape": shape,
                        "data": param.tolist(),
                        "grad": grad_data,
                    }
                )
            if self._last_hidden is None:
                hidden = None
            else:
                slot = self._last_hidden[index]
                hidden = {"shape": slot.shape, "data": slot.tolist()}
            layer_records.append({"params": param_records, "hidden": hidden})
        return {
            "ver": _checkpoint._DOCUMENT_VERSION,
            "hshapes": (
                [list(shape) for shape in self._hidden_shapes]
                if self._hidden_shapes is not None
                else [None] * len(self._modules)
            ),
            "layers": layer_records,
        }

    def _check_compatibility(self, document):
        saved_layers = document["layers"]
        if len(saved_layers) != len(self._modules):
            raise ValueError(
                f"checkpoint has {len(saved_layers)} layers but the current "
                f"model has {len(self._modules)}"
            )
        current = [module.parameters() for module in self._modules]
        for index, (record, params) in enumerate(zip(saved_layers, current)):
            saved_params = record["params"]
            if len(saved_params) != len(params):
                raise ValueError(
                    f"layer {index}: checkpoint has {len(saved_params)} "
                    f"parameters but the current model has {len(params)}"
                )
            for p_index, (saved, param) in enumerate(zip(saved_params, params)):
                if list(saved["shape"]) != param.shape:
                    raise ValueError(
                        f"layer {index} parameter {p_index}: checkpoint shape "
                        f"{list(saved['shape'])} does not match current shape "
                        f"{param.shape}"
                    )

    def _apply_document(self, document):
        restored_hidden = []
        for module, record in zip(self._modules, document["layers"]):
            for param, saved in zip(module.parameters(), record["params"]):
                shape = list(saved["shape"])
                param._replace_data(saved["data"], shape)
                param.grad = Tensor._from_parts(shape, saved["grad"])
            slot = record["hidden"]
            if slot is None:
                restored_hidden.append(None)
            else:
                restored_hidden.append(
                    Tensor._from_parts(list(slot["shape"]), slot["data"])
                )
        # A checkpoint is a clean segment boundary: layer-internal forward
        # caches cannot be serialized, so the restored model always starts
        # with no forward in flight; the caller replays a segment with the
        # returned boundary hidden state before calling backward.
        self._pending_backward = False
        cached = document["hshapes"]
        if any(shape is None for shape in cached):
            self._hidden_shapes = None
        else:
            self._hidden_shapes = [list(shape) for shape in cached]
        self._last_hidden = (
            list(restored_hidden)
            if any(slot is not None for slot in restored_hidden)
            else None
        )
        return restored_hidden

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
