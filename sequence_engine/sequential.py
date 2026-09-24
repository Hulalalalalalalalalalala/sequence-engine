"""Sequential container: chains caller-provided layers in registration order.

Thread-safety
-------------

A single :class:`Sequential` instance may be shared by multiple threads
that interleave ``forward`` / ``backward`` / ``update`` / ``zero_grad`` /
``save`` / ``load`` calls.  Every public method is serialized by one
re-entrant lock, so any interleaving is equivalent to some serial order
of the calls: an ``update`` is atomic with respect to every other
mutation and to a snapshot, a ``save`` always fixes a complete slice
boundary (never two threads' half-written progress), and a ``load`` is
validated fully before its new state is committed in one critical
section -- a concurrent ``update`` can never observe a half-restored
container.
"""

from __future__ import annotations

import threading

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


def _valid_hidden_shape(shape):
    """Hidden tensors are always batched: a non-empty shape of positive ints."""
    if not isinstance(shape, list) or not shape:
        return False
    return all(
        isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
        for dim in shape
    )


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
    * ``save_incremental(directory)`` / ``load_incremental(directory)`` --
      an incremental chain that only writes changed layers; reassembly
      reproduces the full snapshot bit for bit.
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
        # Version of the checkpoint this container's state last came
        # from (FORMAT_VERSION for native state; 1 after loading an old
        # file that was migrated on read).
        self._loaded_src = _checkpoint.FORMAT_VERSION
        # One re-entrant lock orders every individual public call.
        self._lock = threading.RLock()
        # The segment lock makes the whole ``forward ... backward``
        # window atomic across threads: forward acquires it (and keeps it
        # across the return), the matching backward releases it. Parameter
        # mutators (update/zero_grad/load) take it too, so they always run
        # at a slice boundary and can never interleave with a backward's
        # use of the cached activations. save() deliberately does NOT take
        # it: a mid-segment save is refused immediately by the boundary
        # check rather than blocking.
        self._segment_lock = threading.RLock()

    def forward(self, batch, hidden=None):
        if not isinstance(batch, Tensor):
            raise ValueError("batch must be a Tensor")
        batch_shape = batch.shape
        if len(batch_shape) == 0 or batch_shape[0] <= 0:
            raise ValueError("batch must be non-empty")
        # Open the segment before touching layer state. Held until the
        # matching backward() completes, so another thread cannot start a
        # second forward on top of this one's cached activations.
        self._segment_lock.acquire()
        try:
            with self._lock:
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
                        raise ValueError(
                            "each layer's forward must return (Tensor, Tensor)"
                        )
                    new_hidden.append(slot_out)
                self._hidden_shapes = [slot.shape for slot in new_hidden]
                self._last_output = x
                self._last_hidden = new_hidden
                self._pending_backward = True
                # Return deep copies of the slots: the list handed out must
                # keep carrying the boundary values even if the caller (or
                # another thread) immediately starts another segment.
                return x, [Tensor(slot.tolist()) for slot in new_hidden]
        except BaseException:
            # A forward that raised never opened a pending segment.
            self._segment_lock.release()
            raise

    def backward(self, loss):
        with self._lock:
            if not self._pending_backward:
                # No open segment: do not touch the segment lock.
                raise RuntimeError(
                    "backward() requires a preceding forward() and may only be "
                    "called once per forward()"
                )
            upstream = self._parse_loss(loss)
            # Snapshot gradients first: if any layer raises mid-pass, roll the
            # partial accumulation back so the state is exactly as if backward
            # had never run and the caller may retry the same forward's backward
            # (that retry is not a second backward pass). The segment lock stays
            # held across a failed attempt so the retry owns the segment.
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
        # Segment complete: only now let another thread's forward or a
        # boundary mutator proceed.
        self._segment_lock.release()

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
        # Runs at a segment boundary: blocks while another thread is
        # inside forward...backward, so it cannot replace gradients that
        # a concurrent backward is accumulating.
        with self._segment_lock:
            with self._lock:
                for param in self.parameters():
                    param.grad = Tensor(_zeros(param.shape))

    def parameters(self):
        with self._lock:
            params = []
            for module in self._modules:
                params.extend(module.parameters())
            return params

    def update(self, learning_rate):
        """Perform one in-place step ``theta <- theta - learning_rate * grad``.

        The step runs at a segment boundary (blocking while another
        thread is between forward and backward) as a single critical
        section: every parameter is updated in fixed registration order
        using its accumulated gradient, and concurrent snapshots, loads
        or segments see either the pre-update or the post-update state,
        never a partially updated tensor. Parameters without a gradient
        slot are left untouched (their gradient is implicitly zero).
        Gradients are not cleared -- ``zero_grad`` remains the only way
        to reset them.
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
        with self._segment_lock:
            with self._lock:
                for param in self.parameters():
                    if param.grad is not None:
                        param._scaled_subtract_(param.grad, learning_rate)

    # -- full checkpoints ---------------------------------------------------

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
        the engine. The boundary is the only restriction; nothing about
        earlier forwards or updates matters. The values are copied under
        the container lock and serialized afterwards, so a concurrent
        ``update``/``backward`` can never leave the snapshot half-written,
        and saving mutates no existing state. Repeated saves of the same
        state produce identical bytes.
        """
        document = self._snapshot_document()
        if isinstance(target, bytearray):
            # The in-memory buffer is shared caller state: serialize and
            # replace its bytes under the lock so two saves into one
            # bytearray cannot interleave clear()/extend().
            with self._lock:
                return _checkpoint.save_bytes(document, target)
        # Path writes only need the copied document; the bytes land via a
        # temp file + atomic replace, so holding the lock during IO is
        # neither necessary nor desirable.
        return _checkpoint.save_bytes(document, target)

    def load(self, source):
        """Restore state previously written by ``save``.

        *source* is a filesystem path or a bytes-like buffer. The whole
        checkpoint (including a version-1 file, migrated item by item) is
        parsed and validated, and the new tensors are built, before any
        live state changes; the commit itself is one critical section, so
        a concurrent ``update`` never observes a half-restored container.
        The format version, every tensor shape and the layer order
        (count, kinds and per-layer parameter shapes) must match this
        container exactly; hidden-state shapes are checked on the spot,
        even on a model that has never run a forward. A missing path
        raises ``FileNotFoundError``; any structural problem rejects the
        entire checkpoint with ``ValueError`` and leaves the container
        untouched.

        On success returns the restored slice-boundary hidden-state list
        (one tensor per layer), or ``None`` when the checkpoint fixed the
        start-of-training state. Feed it back into the next
        ``forward(batch, hidden)`` to continue the sequence.
        """
        document = _checkpoint.load_bytes(source)
        return self._commit_document(document)

    # -- incremental chains -------------------------------------------------

    def save_incremental(self, directory):
        """Append the current state to the incremental chain at *directory*.

        Only layers whose parameter or gradient content changed relative
        to the previous chain entry are written; unchanged layers are
        referenced by a content digest and copied during reassembly. The
        first call creates ``base.ckp``, ``00000001.delta`` and
        ``manifest.json``; later calls append one numbered delta each and
        publish the manifest atomically. Returns the new sequence id.

        Segment-boundary and locking rules are identical to ``save``;
        concurrent writers to the same chain directory are additionally
        serialized with an advisory file lock, so two simultaneous saves
        leave exactly one complete extra entry on disk, never a mixture.
        """
        document = self._snapshot_document()
        return _checkpoint.save_chain(document, directory)

    def load_incremental(self, directory):
        """Reassemble the incremental chain at *directory* and restore it.

        Layers are merged in chain order; the result is bitwise identical
        to a full ``save`` at the latest chain entry. A truncated,
        corrupt, missing or malformed chain member rejects the whole
        chain with ``ValueError`` and leaves the container untouched. A
        missing *directory* raises ``FileNotFoundError``. Returns the
        restored hidden-state list (or ``None``), exactly like ``load``.
        """
        document = _checkpoint.load_chain(directory)
        return self._commit_document(document)

    # -- snapshot/build/apply helpers --------------------------------------

    def _snapshot_document(self):
        with self._lock:
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
                    {"s": slot.shape, "v": slot.tolist()}
                    for slot in self._last_hidden
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
            return document

    def _commit_document(self, document):
        """Validate *document* fully, then commit it in one critical section.

        The commit runs at a segment boundary (segment lock then the
        container lock, the same lock order forward/update use).
        Validation, tree rebuilding and the state swap are atomic, so a
        racing ``update``/``save``/segment sees either the old state or
        the whole restored state -- never a partially validated or
        half-applied checkpoint.
        """
        with self._segment_lock:
            with self._lock:
                return self._apply_document(document)

    def _apply_document(self, document):
        self._validate_against_model(document)
        params = self.parameters()
        new_params = [
            _rebuild_tree(entry["v"], entry["s"])
            for entry in document["params"]
        ]
        new_grads = [
            _rebuild_tree(entry["v"], entry["s"])
            for entry in document["grads"]
        ]
        if document["hidden"] is None:
            new_hidden = None
        else:
            new_hidden = [
                _rebuild_tree(entry["v"], entry["s"])
                for entry in document["hidden"]
            ]
        src = int(document.get("src", _checkpoint.FORMAT_VERSION))

        for param, tree in zip(params, new_params):
            param._set_values(tree)
        for param, grad_tree in zip(params, new_grads):
            param.grad = Tensor(grad_tree)
        if new_hidden is None:
            self._last_hidden = None
            self._hidden_shapes = None
            restored_hidden = None
        else:
            slots = [Tensor(tree) for tree in new_hidden]
            self._last_hidden = slots
            self._hidden_shapes = [slot.shape for slot in slots]
            # Return copies: callers may feed the list into another
            # container while this one keeps training.
            restored_hidden = [Tensor(slot.tolist()) for slot in slots]
        self._last_output = None
        self._pending_backward = False
        self._loaded_src = src
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
            if not _tree_matches_shape(p_entry.get("v"), p_entry.get("s")):
                raise ValueError(
                    f"checkpoint parameter {index} values do not match their shape"
                )
            if not _tree_matches_shape(g_entry.get("v"), g_entry.get("s")):
                raise ValueError(
                    f"checkpoint gradient {index} values do not match their shape"
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
                if not isinstance(entry, dict) or "s" not in entry or "v" not in entry:
                    raise ValueError(f"checkpoint hidden slot {index} is malformed")
                # Checked on the spot, including a model that has never run
                # a forward: one well-formed batched tensor per layer, with
                # values matching its declared shape.
                if not _valid_hidden_shape(entry["s"]):
                    raise ValueError(
                        f"checkpoint hidden slot {index} shape is not a valid "
                        f"batched tensor shape"
                    )
                if not _tree_matches_shape(entry["v"], entry["s"]):
                    raise ValueError(
                        f"checkpoint hidden slot {index} values do not match "
                        f"its shape"
                    )
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


def _tree_matches_shape(value, shape):
    """Structural check used before any live state is touched."""
    if not isinstance(shape, list):
        return False
    if shape:
        if not isinstance(value, list) or len(value) != shape[0]:
            return False
        dim = shape[0]
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            return False
        return all(_tree_matches_shape(item, shape[1:]) for item in value)
    return (
        not isinstance(value, (bool, list))
        and isinstance(value, (int, float))
    )
