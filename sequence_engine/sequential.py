"""Sequential container: chains caller-provided layers in registration order."""

from __future__ import annotations

import hashlib
import math
import os
import struct
import threading

from . import checkpoint as _checkpoint
from .tensor import Tensor
from .tensor import _deep_copy as _copy_tree

_REQUIRED_CAPABILITIES = ("forward", "backward", "parameters")

# Adam coefficients are fixed by the engine; callers only choose the rate.
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPSILON = 1e-8


def _zeros(shape):
    if not shape:
        return 0.0
    return [_zeros(shape[1:]) for _ in range(shape[0])]


def _parameters_fingerprint(params):
    """A deterministic bitwise fingerprint of every parameter's values."""
    digest = hashlib.sha256()
    for param in params:
        for dim in param.shape:
            digest.update(struct.pack("<q", dim))
        _fingerprint_tree(param._values(), digest)
    return digest.digest()


def _fingerprint_tree(value, digest):
    if isinstance(value, list):
        digest.update(b"[")
        for item in value:
            _fingerprint_tree(item, digest)
        digest.update(b"]")
    elif isinstance(value, bool):
        digest.update(b"?")
    elif isinstance(value, int):
        digest.update(b"i")
        digest.update(str(value).encode("ascii"))
        digest.update(b";")
    else:
        digest.update(b"f")
        digest.update(struct.pack("<d", value))


def _adam_update_trees(theta, m, v, grad, lr, m_correction, v_correction):
    """One Adam step on matching trees; returns ``(theta, m, v)`` updated.

    The arithmetic order is fixed -- m and v first, then the bias
    corrections, then the parameter -- so a resumed run is bitwise
    identical to an uninterrupted one.
    """
    if isinstance(theta, list):
        new_theta, new_m, new_v = [], [], []
        for t_item, m_item, v_item, g_item in zip(theta, m, v, grad):
            nt, nm, nv = _adam_update_trees(
                t_item, m_item, v_item, g_item, lr, m_correction, v_correction
            )
            new_theta.append(nt)
            new_m.append(nm)
            new_v.append(nv)
        return new_theta, new_m, new_v
    m_next = _ADAM_BETA1 * m + 0.1 * grad
    v_next = _ADAM_BETA2 * v + 0.001 * (grad * grad)
    m_hat = m_next / m_correction
    v_hat = v_next / v_correction
    return theta - lr * m_hat / (math.sqrt(v_hat) + _ADAM_EPSILON), m_next, v_next


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
    * ``adam_step(learning_rate)`` -- one in-place Adam step with the
      engine's fixed coefficients; the first/second moments and the step
      count ride along in every checkpoint.
    * ``set_recompute(enabled)`` -- bounded-memory mode: a segment keeps
      only its boundary hidden state plus a few anchors, and the backward
      pass recomputes the activations from them.
    * ``save(target)`` / ``load(source)`` -- atomic, versioned snapshots of
      parameters, accumulated gradients, optimizer state and
      slice-boundary hidden state, to a filesystem path, an
      incremental-chain directory or to an in-memory ``bytearray``.
    * ``compact(target, up_to=None)`` -- folds an incremental chain's
      basis and a prefix of its deltas into one new basis segment,
      crash-safely; the reassembled state is bit for bit unchanged.
    * ``derive(source, target, at=None)`` -- forks an incremental chain
      into a new chain directory (or a new ``MemoryChain``) that shares
      the segments up to the fork point and then evolves independently.
    * ``drop(target)`` -- removes a chain, reclaiming only the segments
      no remaining chain of its family can reach.

    All public operations are serialised by one re-entrant lock, so
    several threads may interleave ``forward``, ``backward``, ``update``,
    ``save`` and ``load`` calls: every observed parameter state is one a
    serial execution of the same calls could have produced, ``update`` is
    atomic and a snapshot always corresponds to one complete slice
    boundary -- never two passes half mixed.
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
        # Recompute mode: the configured switch plus the per-segment latch
        # with its anchors (segment input + incoming hidden values) and the
        # parameter fingerprint taken at forward time.
        self._recompute = False
        self._segment_recompute = False
        self._anchors = None
        self._forward_fingerprint = None
        # Retry anchors (segment input + incoming hidden values) kept in
        # every mode while a backward is pending: if a layer raises
        # mid-backward the segment's forward is replayed from them, so
        # layer caches a failed pass destroyed are rebuilt before the
        # caller retries.
        self._retry_anchors = None
        # Optimizer state: first/second moment trees per parameter plus the
        # step count.  Lazily sized to the parameter list on first use.
        self._adam_t = 0
        self._adam_m = None
        self._adam_v = None
        # Version a checkpoint was loaded from (None until the first load).
        self._loaded_from_version = None
        self._lock = threading.RLock()

    @property
    def loaded_from_version(self):
        """Format version the last ``load`` migrated from (``None`` before).

        A version-1 or version-2 checkpoint reports ``1``/``2``; a native
        version-3 file reports ``3``. The source version is only updated on
        a successful load -- a checkpoint rejected mid-migration leaves the
        previous value untouched.
        """
        return self._loaded_from_version

    def forward(self, batch, hidden=None):
        with self._lock:
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
                    raise ValueError(
                        "each layer's forward must return (Tensor, Tensor)"
                    )
                new_hidden.append(slot_out)
            self._hidden_shapes = [slot.shape for slot in new_hidden]
            self._last_output = x
            self._last_hidden = new_hidden
            self._retry_anchors = (
                batch.tolist(),
                [None if slot is None else slot.tolist() for slot in slots],
            )
            if self._recompute:
                # Bounded-memory mode: keep only the anchors needed to
                # recompute this segment's activations at backward time --
                # the segment input, the incoming hidden values and a
                # fingerprint of the parameters.
                self._segment_recompute = True
                self._anchors = (
                    batch.tolist(),
                    [None if slot is None else slot.tolist() for slot in slots],
                )
                self._forward_fingerprint = _parameters_fingerprint(
                    self.parameters()
                )
            else:
                self._segment_recompute = False
                self._anchors = None
                self._forward_fingerprint = None
            self._pending_backward = True
            return x, new_hidden

    def set_recompute(self, enabled):
        """Switch bounded-memory (activation-recompute) mode on or off.

        With the mode on, a ``forward`` keeps only the slice-boundary
        hidden state and a few anchors (the segment input and the incoming
        hidden values); the matching ``backward`` recomputes the layer
        activations from those anchors instead of relying on anything the
        layers retained.  Forward outputs and parameter gradients are
        bitwise identical to the default mode, and peak activation memory
        no longer grows with the number of segments.

        The switch may only be flipped between segments: changing it while
        a backward is still pending raises ``RuntimeError`` and leaves the
        container state untouched.
        """
        with self._lock:
            enabled = bool(enabled)
            if self._pending_backward and enabled != self._segment_recompute:
                raise RuntimeError(
                    "set_recompute() cannot change the mode in the middle of "
                    "a segment (a backward is still pending)"
                )
            self._recompute = enabled

    def backward(self, loss):
        with self._lock:
            if not self._pending_backward:
                raise RuntimeError(
                    "backward() requires a preceding forward() and may only be "
                    "called once per forward()"
                )
            upstream = self._parse_loss(loss)
            if self._segment_recompute:
                self._recompute_activations()
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
            except BaseException as layer_exc:
                for param, saved in zip(self.parameters(), grad_snapshot):
                    param.grad = None if saved is None else Tensor(saved)
                # A layer's failed backward may also have destroyed caches a
                # retry would need (its own, or a layer's that already ran).
                # Replay the segment's forward from the retry anchors so every
                # layer cache is rebuilt exactly as the recorded forward left
                # it.  The replay only re-runs layer forwards -- parameters,
                # gradients and the recorded boundary state stay untouched.
                # If the replay itself fails that failure must not be
                # swallowed: it surfaces as a ValueError naming the rebuild
                # stage, chained onto the original layer exception, which
                # stays the exception handed to the caller (it is neither
                # masked nor replaced).
                try:
                    self._replay_forward()
                except BaseException as replay_exc:
                    raise layer_exc from self._cache_rebuild_error(replay_exc)
                raise
            self._pending_backward = False
            self._anchors = None
            self._forward_fingerprint = None
            self._retry_anchors = None

    def _cache_rebuild_error(self, replay_exc):
        """Wrap a failed cache-rebuild replay as a stage-named ValueError.

        The original backward exception is what the caller receives; this
        ValueError is chained onto it (``__cause__``) so the rebuild
        failure is neither silent nor able to mask the original, yet
        always names the rebuild stage and carries the replay error.
        """
        return ValueError(
            "failed to rebuild layer caches: replaying the segment's "
            "forward after the backward error raised "
            f"{type(replay_exc).__name__}: {replay_exc}"
        )

    def _replay_forward(self):
        """Re-run the in-flight segment's forward to rebuild layer caches.

        Uses the retry anchors recorded at forward time.  The container's
        own recorded state (``_last_output``, ``_last_hidden``, the
        anchors) is left exactly as it was, so a retried backward observes
        the same boundary and the tuple-loss identity check still passes.
        """
        if self._retry_anchors is None:
            return
        batch_values, slot_values = self._retry_anchors
        x = Tensor(batch_values)
        slots = [None if value is None else Tensor(value) for value in slot_values]
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

    def _recompute_activations(self):
        """Rebuild the in-flight segment's activations from its anchors.

        Runs before the backward walk in recompute mode.  The parameters
        must be bitwise identical to their forward-time values -- an
        external rewrite in between would silently corrupt the recomputed
        activations, so it raises ``RuntimeError`` instead, before any
        gradient is touched.
        """
        if self._anchors is None or self._forward_fingerprint is None:
            raise RuntimeError("recompute anchors are missing for this segment")
        if _parameters_fingerprint(self.parameters()) != self._forward_fingerprint:
            raise RuntimeError(
                "parameters were modified while the segment was in flight; "
                "cannot recompute its activations"
            )
        batch_values, slot_values = self._anchors
        x = Tensor(batch_values)
        slots = [None if value is None else Tensor(value) for value in slot_values]
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
        if _checkpoint._trees_differ(
            x.tolist(), self._last_output.tolist(), x.shape
        ):
            raise RuntimeError(
                "recomputed activations diverge from the recorded forward "
                "output; the segment cannot be back-propagated"
            )

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

        Every parameter is updated in fixed registration order using its
        accumulated gradient; parameters without a gradient slot are left
        untouched (their gradient is implicitly zero). Gradients are not
        cleared -- ``zero_grad`` remains the only way to reset them.

        The whole step runs under the container lock, so a concurrent
        ``save`` or ``load`` can never observe a half-updated model.
        """
        with self._lock:
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

    def _ensure_optim_state(self, count):
        if self._adam_m is None or len(self._adam_m) != count:
            # Un-stepped state starts from the initial values: t = 0 and
            # zero moments shaped like the parameters.
            params = self.parameters()
            self._adam_t = 0
            self._adam_m = [_zeros(param.shape) for param in params]
            self._adam_v = [_zeros(param.shape) for param in params]

    def adam_step(self, learning_rate):
        """Perform one in-place Adam step on every parameter.

        The coefficients are fixed by the engine (beta1 = 0.9, beta2 =
        0.999, epsilon = 1e-8); only the learning rate is chosen by the
        caller.  Each call first updates the moments
        ``m <- 0.9*m + 0.1*g`` and ``v <- 0.999*v + 0.001*g^2``, advances
        the step count ``t`` (1 on the first call), then applies the
        bias-corrected update ``theta <- theta - lr * m_hat /
        (sqrt(v_hat) + 1e-8)`` with ``m_hat = m / (1 - 0.9**t)`` and
        ``v_hat = v / (1 - 0.999**t)``.  Parameters without a gradient are
        left untouched; gradients are not cleared.  A state that never
        stepped starts from ``t = 0`` and zero moments.  A non-finite or
        non-numeric learning rate raises ``ValueError``.

        The moments and the step count are part of every checkpoint, so a
        resumed run continues bitwise identically.  The whole step runs
        under the container lock, like ``update``.
        """
        with self._lock:
            if isinstance(learning_rate, bool) or not isinstance(
                learning_rate, (int, float)
            ):
                raise ValueError("learning rate must be a number")
            if learning_rate != learning_rate or learning_rate in (
                float("inf"),
                float("-inf"),
            ):
                raise ValueError("learning rate must be finite")
            params = self.parameters()
            self._ensure_optim_state(len(params))
            self._adam_t += 1
            step = self._adam_t
            lr = float(learning_rate)
            m_correction = 1.0 - _ADAM_BETA1 ** step
            v_correction = 1.0 - _ADAM_BETA2 ** step
            # Build every updated tree before touching any live state, so
            # the step is all-or-nothing.
            new_values = []
            new_m = []
            new_v = []
            for index, param in enumerate(params):
                grad = param.grad
                if grad is None:
                    new_values.append(None)
                    new_m.append(self._adam_m[index])
                    new_v.append(self._adam_v[index])
                    continue
                theta, m_tree, v_tree = _adam_update_trees(
                    param._values(),
                    self._adam_m[index],
                    self._adam_v[index],
                    grad._values(),
                    lr,
                    m_correction,
                    v_correction,
                )
                new_values.append(theta)
                new_m.append(m_tree)
                new_v.append(v_tree)
            for param, values in zip(params, new_values):
                if values is not None:
                    param._set_values(values)
            self._adam_m = new_m
            self._adam_v = new_v

    # -- checkpoints --------------------------------------------------------

    def _snapshot_document(self):
        params = self.parameters()
        self._ensure_optim_state(len(params))
        return {
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
            "optim": {
                "t": self._adam_t,
                "m": [
                    {"s": param.shape, "v": _copy_tree(tree)}
                    for param, tree in zip(params, self._adam_m)
                ],
                "v": [
                    {"s": param.shape, "v": _copy_tree(tree)}
                    for param, tree in zip(params, self._adam_v)
                ],
            },
            "layers": [
                {
                    "kind": _layer_kind(module),
                    "shapes": [param.shape for param in module.parameters()],
                }
                for module in self._modules
            ],
            "pending": False,
        }

    def save(self, target):
        """Fix the current slice-boundary training state into *target*.

        *target* is either a filesystem path (``str``/``os.PathLike``), an
        existing directory used as an incremental checkpoint chain, or a
        ``bytearray`` used as an in-memory buffer. The snapshot records
        every parameter tensor, every accumulated gradient (zeros for
        parameters that have never received one), the optimizer state
        (first/second moments and step count; zeros and ``t = 0`` when
        ``adam_step`` never ran), the current slice-boundary hidden state,
        the layer order with per-layer parameter shapes and the format
        version.

        A boundary exists before the first segment and after every
        ``forward`` (the hidden state returned by that forward *is* the
        boundary), so saving is allowed even while the latest forward has
        not been back-propagated yet: the snapshot fixes the boundary
        state, not the in-flight activations. Saving mutates no existing
        state and repeated saves of the same state produce identical
        bytes. The whole snapshot is taken under the container lock, so a
        concurrent ``update`` can never leave a half-written tensor in it.
        """
        with self._lock:
            document = self._snapshot_document()
            return _checkpoint.save_bytes(document, target)

    def load(self, source):
        """Restore state previously written by ``save``.

        *source* is a filesystem path, an incremental-chain directory or a
        bytes-like buffer. The whole checkpoint is validated before
        anything is applied: the format version (versions 1 and 2 are
        migrated item by item -- a missing optimizer state starts at
        ``t = 0`` with zero moments -- and the origin is recorded), every
        tensor shape and the layer order (count, kinds and per-layer
        parameter shapes) must match this container exactly. Hidden-state shapes are verified on
        the spot, including for a model that has never run a forward. A
        missing path raises ``FileNotFoundError``; any structural problem
        rejects the entire checkpoint with ``ValueError`` and leaves the
        container untouched -- nothing is partially applied or silently
        filled in.

        On success returns the restored slice-boundary hidden-state list
        (one tensor per layer), or ``None`` when the checkpoint fixed the
        start-of-training state. Feed it back into the next
        ``forward(batch, hidden)`` to continue the sequence.
        """
        with self._lock:
            document, source_version = _checkpoint.load_bytes_with_source(source)
            self._validate_against_model(document)

            # Build every replacement tree before touching any live state,
            # so a failure in the middle still leaves the model exactly as
            # it was (validation above already rejects malformed input, but
            # the rebuild is kept on the apply side as a hard guarantee).
            params = self.parameters()
            new_values = [
                _rebuild_tree(entry["v"], entry["s"])
                for entry in document["params"]
            ]
            new_grads = [
                Tensor(_rebuild_tree(entry["v"], entry["s"]))
                for entry in document["grads"]
            ]
            if document["hidden"] is None:
                new_hidden = None
                new_hidden_shapes = None
            else:
                rebuilt_slots = [
                    Tensor(_rebuild_tree(entry["v"], entry["s"]))
                    for entry in document["hidden"]
                ]
                new_hidden = rebuilt_slots
                new_hidden_shapes = [slot.shape for slot in rebuilt_slots]
            optim = document["optim"]
            new_optim_t = optim["t"]
            new_optim_m = [
                _rebuild_tree(entry["v"], entry["s"]) for entry in optim["m"]
            ]
            new_optim_v = [
                _rebuild_tree(entry["v"], entry["s"]) for entry in optim["v"]
            ]

            for param, values in zip(params, new_values):
                param._set_values(values)
            for param, grad in zip(params, new_grads):
                param.grad = grad
            self._adam_t = new_optim_t
            self._adam_m = new_optim_m
            self._adam_v = new_optim_v
            self._last_hidden = new_hidden
            self._hidden_shapes = new_hidden_shapes
            self._last_output = None
            self._pending_backward = False
            self._segment_recompute = False
            self._anchors = None
            self._forward_fingerprint = None
            self._retry_anchors = None
            self._loaded_from_version = source_version
            if new_hidden is None:
                return None
            return [Tensor(slot.tolist()) for slot in new_hidden]

    def compact(self, target, up_to=None):
        """Compact an incremental checkpoint chain in place.

        *target* is an existing chain directory or a ``MemoryChain``.  The
        basis segment and the deltas through *up_to* (the current head
        when omitted) are folded into one new basis segment and the
        remaining deltas are renumbered after it.  The state the chain
        reassembles to -- parameters, gradients, optimizer moments and
        step count, hidden state -- is bit for bit identical before and
        after, and compaction advances no optimizer step; only the
        segment count changes, decreasing deterministically by the merged
        range.  Repeating the same compaction is a no-op, as is a chain
        with nothing to merge.  Old-version segments participate exactly
        as on load and the compacted chain is rewritten in the current
        format version.

        A process killed mid-compaction leaves either the old or the new
        head reachable; the next open of the chain finishes the
        roll-forward, so the directory always holds one complete chain.
        A missing directory raises ``FileNotFoundError``, an unwritable
        directory or a full disk raises ``OSError``, and any corrupt or
        inconsistent segment rejects the whole compaction with
        ``ValueError`` before anything is written.
        """
        with self._lock:
            if isinstance(target, _checkpoint.MemoryChain):
                return _checkpoint.compact_chain_memory(target, up_to)
            if isinstance(target, (str, os.PathLike)):
                return _checkpoint.compact_chain(target, up_to)
            raise TypeError(
                "compact target must be a chain directory or a MemoryChain"
            )

    def verify(self, source):
        """Verify an incremental checkpoint chain without changing it.

        *source* is an existing chain directory (or a ``MemoryChain``).
        Every segment -- basis then each delta through the ``head`` -- is
        checked read-only for completeness (framing/CRC), segment order,
        its basis reference and tensor/layer shapes.  A sound chain
        returns a success report; otherwise a ``ValueError`` names the
        first bad segment's position and the reason.  The chain is only
        read, so a chain compaction that was interrupted on disk is
        inspected in place and the directory is left byte for byte
        unchanged.

        A missing directory raises ``FileNotFoundError``; any truncation,
        missing field, ordering, reference or shape defect rejects the
        whole chain with ``ValueError``.
        """
        with self._lock:
            if isinstance(source, _checkpoint.MemoryChain):
                return _checkpoint.verify_chain_memory(source)
            if isinstance(source, (str, os.PathLike)):
                return _checkpoint.verify_chain(source)
            raise TypeError(
                "verify source must be a chain directory or a MemoryChain"
            )

    def derive(self, source, target=None, at=None):
        """Fork an incremental checkpoint chain into a new chain.

        *source* is an existing chain directory or a ``MemoryChain``;
        *at* is the segment to fork from (the source head when omitted).
        The new chain shares every segment up to the fork point with the
        source chain -- the segment files are shared, never copied -- and
        then evolves independently: each chain maintains only its own
        head and its own appended deltas, saves, loads, verifications and
        streaming compactions on either chain never disturb the other,
        and the reassembled state of each chain stays bit for bit the
        state an unforked chain with the same saves would hold.  A shared
        segment is reclaimed exactly when no chain's head can reach it
        any more, so dropping a branch or compacting the source never
        disturbs the remaining chains.

        For a directory source, *target* is the new chain directory (it
        must not exist yet) and the call returns ``None``; for a
        ``MemoryChain`` source the new ``MemoryChain`` is returned.  A
        fork point that is not a committed segment boundary or names a
        segment beyond the source head, and an already existing target
        directory, raise ``ValueError``; a missing source directory or
        referenced segment raises ``FileNotFoundError``; an unwritable
        destination or a full disk raises ``OSError``.
        """
        with self._lock:
            if isinstance(source, _checkpoint.MemoryChain):
                if target is not None:
                    raise TypeError(
                        "deriving a MemoryChain returns the new chain; "
                        "target must be None"
                    )
                return _checkpoint.derive_chain_memory(source, at)
            if isinstance(source, (str, os.PathLike)):
                if target is None:
                    raise TypeError(
                        "deriving a chain directory requires a target "
                        "directory"
                    )
                return _checkpoint.derive_chain(source, target, at)
            raise TypeError(
                "derive source must be a chain directory or a MemoryChain"
            )

    def drop(self, target):
        """Remove a chain, reclaiming only unreachable shared segments.

        *target* is a chain directory or a ``MemoryChain``.  Segments
        shared with other chains of the family stay alive through those
        chains' own references; everything else is reclaimed.  A missing
        directory raises ``FileNotFoundError``.
        """
        with self._lock:
            if isinstance(target, _checkpoint.MemoryChain):
                return _checkpoint.drop_chain_memory(target)
            if isinstance(target, (str, os.PathLike)):
                return _checkpoint.drop_chain(target)
            raise TypeError(
                "drop target must be a chain directory or a MemoryChain"
            )

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
            # Eagerly confirm the leaves agree with the declared shapes
            # before any live state is touched.
            _rebuild_tree(p_entry["v"], p_entry["s"])
            _rebuild_tree(g_entry["v"], g_entry["s"])

        saved_layers = document.get("layers")
        if not isinstance(saved_layers, list) or len(saved_layers) != len(
            self._modules
        ):
            raise ValueError(
                "checkpoint layer count does not match the current model"
            )
        saved_optim = document.get("optim")
        if not isinstance(saved_optim, dict) or set(saved_optim) != {"t", "m", "v"}:
            raise ValueError("checkpoint optimizer state is missing or malformed")
        optim_t = saved_optim["t"]
        if (
            isinstance(optim_t, bool)
            or not isinstance(optim_t, int)
            or optim_t < 0
        ):
            raise ValueError("checkpoint optimizer step count is invalid")
        optim_m = saved_optim["m"]
        optim_v = saved_optim["v"]
        if (
            not isinstance(optim_m, list)
            or not isinstance(optim_v, list)
            or len(optim_m) != len(params)
            or len(optim_v) != len(params)
        ):
            raise ValueError(
                "checkpoint optimizer state does not match the current model"
            )
        for index, (param, m_entry, v_entry) in enumerate(
            zip(params, optim_m, optim_v)
        ):
            if not isinstance(m_entry, dict) or not isinstance(v_entry, dict):
                raise ValueError(f"checkpoint optimizer moment {index} is malformed")
            if m_entry.get("s") != param.shape or v_entry.get("s") != param.shape:
                raise ValueError(
                    f"checkpoint optimizer moment {index} shape does not match "
                    f"the current model"
                )
            _rebuild_tree(m_entry["v"], m_entry["s"])
            _rebuild_tree(v_entry["v"], v_entry["s"])

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
            checkpoint_shapes = []
            for index, entry in enumerate(saved_hidden):
                if not isinstance(entry, dict) or "s" not in entry or "v" not in entry:
                    raise ValueError(f"checkpoint hidden slot {index} is malformed")
                # Verify the slot concretely on the spot: a well-formed
                # positive shape whose values conform. This runs even for a
                # model that has never executed a forward and therefore has
                # no recorded hidden shapes yet; the first load pins them.
                rebuilt = _rebuild_tree(entry["v"], entry["s"])
                slot_shape = Tensor(rebuilt).shape
                checkpoint_shapes.append(slot_shape)
                if self._hidden_shapes is not None and (
                    slot_shape != self._hidden_shapes[index]
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
