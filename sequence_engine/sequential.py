"""Sequential container: chains caller-provided layers in registration order."""

from __future__ import annotations

import math
import struct
import threading

from . import checkpoint as _checkpoint
from .tensor import Tensor

_REQUIRED_CAPABILITIES = ("forward", "backward", "parameters")

# Adam coefficients are fixed on purpose: adam_step takes only the learning
# rate, so no caller can retune the moments, the bias correction or the
# denominator floor, and no other optimizer kind can be selected. The
# moment update uses the spec's literal decimal coefficients 0.1 / 0.001
# (which are not bit-identical to ``1 - 0.9`` / ``1 - 0.999`` in float64);
# the bias-correction divisors are genuinely ``1 - beta ** t``.
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_ONE_MINUS_BETA1 = 0.1
_ADAM_ONE_MINUS_BETA2 = 0.001
_ADAM_EPSILON = 1e-8


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


def _walk_leaves(tree):
    if isinstance(tree, list):
        for item in tree:
            yield from _walk_leaves(item)
    else:
        yield tree


def _parameter_fingerprint(params):
    """Bit-exact identity token for every parameter leaf.

    Floats are tokenised by their raw IEEE-754 bits so even -0.0 vs +0.0
    counts; the token is a tuple of small immutable values and therefore
    costs only the parameter count, never anything proportional to the
    streamed sequence length.
    """
    token = []
    for param in params:
        for leaf in _walk_leaves(param._values()):
            if isinstance(leaf, bool):
                token.append((0, int(leaf)))
            elif isinstance(leaf, int):
                token.append((0, leaf))
            elif isinstance(leaf, float):
                token.append((1, struct.pack("<d", leaf)))
            else:
                token.append((2, repr(type(leaf))))
    return tuple(token)


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

    Long sequences stay sliced into segments by the caller; each slice is
    one ``forward``/``backward`` pair. With ``set_recompute(True)`` the
    container retains, for a pending slice, only the boundary inputs (the
    slice input batch and the hidden slots carried into it), the boundary
    output and a parameter fingerprint. Per-layer activations are
    reconstructed by replaying the forward pass during ``backward``, so
    retained memory is independent of the streamed sequence length; the
    forward output and the parameter gradients come out bit for bit
    identical to the ordinary path.

    Besides the forward/backward cycle the container offers:

    * ``update(learning_rate)`` -- one in-place gradient step
      ``theta <- theta - lr * grad`` on every parameter; gradients are
      not cleared (``zero_grad`` still does that).
    * ``adam_step(learning_rate)`` -- one bias-corrected Adam step with
      fixed coefficients, carrying its own first/second moment state and
      step count. It never touches the plain gradient-step path.
    * ``save(target)`` / ``load(source)`` -- atomic, versioned snapshots of
      parameters, accumulated gradients, optimizer state and
      slice-boundary hidden state, to a filesystem path, an
      incremental-chain directory or to an in-memory ``bytearray``.

    All public operations are serialised by one re-entrant lock, so
    several threads may interleave ``forward``, ``backward``,
    ``update``/``adam_step``, ``save`` and ``load`` calls: every observed
    parameter state is one a serial execution of the same calls could have
    produced, a step is atomic and a snapshot always corresponds to one
    complete slice boundary -- never two passes half mixed.
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
        # Bounded-memory recomputation switch plus the anchors of the
        # pending slice (only populated while recompute is enabled).
        self._recompute = False
        self._replay_batch = None
        self._replay_hidden = None
        self._forward_fingerprint = None
        # Adam state: moments are allocated lazily at the first step; a
        # never-stepped container snapshots t == 0 with all-zero moments.
        self._adam_t = 0
        self._adam_m = None
        self._adam_v = None
        # Version a checkpoint was loaded from (None until the first load).
        self._loaded_from_version = None
        self._lock = threading.RLock()

    @property
    def loaded_from_version(self):
        """Format version the last ``load`` migrated from (``None`` before).

        A version-1 checkpoint reports ``1``, a version-2 file ``2`` and a
        native version-3 file ``3``. The source version is only updated on
        a successful load -- a checkpoint rejected mid-migration leaves
        the previous value untouched.
        """
        return self._loaded_from_version

    def set_recompute(self, enabled):
        """Toggle bounded-memory activation recomputation.

        With the switch on, a pending ``forward`` retains only the slice
        boundary (carried-in hidden slots and the output boundary), the
        slice input batch and a parameter fingerprint; ``backward``
        replays the layer forwards to rebuild the per-layer activations it
        needs. Outputs and parameter gradients are identical bit for bit
        to the non-recomputing path, while peak retained memory no longer
        grows with the streamed sequence length.

        Toggling the switch while a slice is in flight (after
        ``forward`` and before its ``backward``) raises ``RuntimeError``
        and changes nothing -- the flag keeps its old value and no state
        is half-converted.
        """
        with self._lock:
            if not isinstance(enabled, bool):
                raise ValueError("set_recompute expects a boolean")
            if self._pending_backward and enabled != self._recompute:
                raise RuntimeError(
                    "cannot toggle recompute between a forward() and its "
                    "backward(): finish or discard the pending slice first"
                )
            self._recompute = enabled

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
                if self._recompute:
                    # The boundary output is kept; everything else the
                    # layer cached for its backward is not, so a layer
                    # that knows how to drop its cached activations does
                    # so immediately. Layers without the hook keep their
                    # own cache; the container itself holds nothing
                    # per-layer either way.
                    release = getattr(module, "release_activations", None)
                    if callable(release):
                        release()
            self._hidden_shapes = [slot.shape for slot in new_hidden]
            self._last_output = x
            self._last_hidden = new_hidden
            self._pending_backward = True
            if self._recompute:
                # Keep just the anchors needed to rebuild every layer
                # activation during backward. The anchors are frozen
                # copies of the slice input and the carried-in boundary
                # slots, so a caller mutating the tensors it passed in
                # after the forward cannot change the replay; nothing
                # per-layer is held, and the cost stays a slice boundary.
                self._replay_batch = Tensor(batch.tolist())
                self._replay_hidden = [
                    None if slot is None else Tensor(slot.tolist())
                    for slot in slots
                ]
                self._forward_fingerprint = _parameter_fingerprint(
                    self.parameters()
                )
            else:
                self._replay_batch = None
                self._replay_hidden = None
                self._forward_fingerprint = None
            return x, new_hidden

    def backward(self, loss):
        with self._lock:
            if not self._pending_backward:
                raise RuntimeError(
                    "backward() requires a preceding forward() and may only be "
                    "called once per forward()"
                )
            upstream = self._parse_loss(loss)
            if self._recompute:
                # The replay must run against exactly the parameters the
                # original forward saw: an external rewrite between the
                # slice forward and its recomputation cannot be repaired
                # by replaying, so refuse the whole pass before anything
                # accumulates (no half state is left behind).
                current_fingerprint = _parameter_fingerprint(self.parameters())
                if current_fingerprint != self._forward_fingerprint:
                    raise RuntimeError(
                        "parameters were modified after forward() and before "
                        "the recomputing backward(); the slice is stale"
                    )
                self._backward_with_recompute(upstream)
            else:
                self._backward_direct(upstream)

    def _backward_direct(self, upstream):
        # Snapshot gradients first: if any layer raises mid-pass, roll the
        # partial accumulation back so the state is exactly as if backward
        # had never run and the caller may retry the same forward's backward
        # (that retry is not a second backward pass).
        grad_snapshot = self._grad_snapshot()
        try:
            for module in reversed(self._modules):
                upstream = module.backward(upstream)
        except BaseException:
            self._restore_grad_snapshot(grad_snapshot)
            raise
        self._finish_backward()

    def _backward_with_recompute(self, upstream):
        # Snapshot gradients for the same rollback-and-retry contract as
        # the direct path: a failure (also during a replay forward) rolls
        # every partial accumulation back, and retrying the same backward
        # is not counted as a second pass.
        grad_snapshot = self._grad_snapshot()
        try:
            modules = self._modules
            incoming = self._replay_hidden
            for index in range(len(modules) - 1, -1, -1):
                # Rebuild this layer's activation just in time: replay the
                # prefix 0..index from the slice input and the carried-in
                # boundary slots, then back-propagate only this layer.
                x = self._replay_batch
                for prefix_index in range(index + 1):
                    x, _hidden_out = modules[prefix_index].forward(
                        x, incoming[prefix_index]
                    )
                upstream = modules[index].backward(upstream)
                # The replay just repopulated this layer's cache; release
                # it again so nothing per-slice survives the boundary.
                release = getattr(modules[index], "release_activations", None)
                if callable(release):
                    release()
        except BaseException:
            self._restore_grad_snapshot(grad_snapshot)
            raise
        self._finish_backward()

    def _finish_backward(self):
        self._pending_backward = False
        self._replay_batch = None
        self._replay_hidden = None
        self._forward_fingerprint = None

    def _grad_snapshot(self):
        return [
            None if param.grad is None else param.grad.tolist()
            for param in self.parameters()
        ]

    def _restore_grad_snapshot(self, grad_snapshot):
        for param, saved in zip(self.parameters(), grad_snapshot):
            param.grad = None if saved is None else Tensor(saved)

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

    @staticmethod
    def _validate_learning_rate(learning_rate):
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float)
        ):
            raise ValueError("learning rate must be a number")
        if learning_rate != learning_rate or learning_rate in (
            float("inf"),
            float("-inf"),
        ):
            raise ValueError("learning rate must be finite")
        return float(learning_rate)

    def update(self, learning_rate):
        """Perform one in-place step ``theta <- theta - learning_rate * grad``.

        Every parameter is updated in fixed registration order using its
        accumulated gradient; parameters without a gradient slot are left
        untouched (their gradient is implicitly zero). Gradients are not
        cleared -- ``zero_grad`` remains the only way to reset them.

        The whole step runs under the container lock, so a concurrent
        ``save`` or ``load`` can never observe a half-updated model. The
        Adam path (``adam_step``) is separate: this method never reads or
        changes the optimizer moments or the step count.
        """
        with self._lock:
            lr = self._validate_learning_rate(learning_rate)
            for param in self.parameters():
                if param.grad is not None:
                    param._scaled_subtract_(param.grad, lr)

    def adam_step(self, learning_rate):
        """Perform one bias-corrected Adam step with fixed coefficients.

        The coefficients are not caller-selectable: the moments use
        ``0.9`` and ``0.999``, bias correction uses the step count ``t``
        (starting at 1) and the denominator floor is ``1e-8``. For every
        parameter leaf, in registration order::

            g  = accumulated gradient (zero when the parameter has none)
            m  <- 0.9 * m + 0.1 * g
            v  <- 0.999 * v + 0.001 * g * g
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            theta <- theta - lr * m_hat / (sqrt(v_hat) + 1e-8)

        Gradients are not cleared (``zero_grad`` still does that), and the
        plain ``update`` path, its formula and gradient accumulation are
        untouched -- the two never share state. A non-numeric or
        non-finite *learning_rate* raises ``ValueError``. The moments and
        step count travel with every checkpoint, so a resumed run follows
        the uninterrupted parameter trajectory bit for bit. The step is
        applied atomically under the container lock.
        """
        with self._lock:
            lr = self._validate_learning_rate(learning_rate)
            params = self.parameters()
            if self._adam_m is None:
                self._adam_m = [Tensor(_zeros(param.shape)) for param in params]
                self._adam_v = [Tensor(_zeros(param.shape)) for param in params]
            moments = self._adam_m
            velocities = self._adam_v
            if len(moments) != len(params) or len(velocities) != len(params):
                raise RuntimeError("optimizer moment count does not match parameters")

            t = self._adam_t + 1
            bias1 = 1.0 - _ADAM_BETA1**t
            bias2 = 1.0 - _ADAM_BETA2**t
            new_param_trees = []
            new_m_trees = []
            new_v_trees = []
            for param, moment, velocity in zip(params, moments, velocities):
                grad_tree = (
                    param.grad._values()
                    if param.grad is not None
                    else _zeros(param.shape)
                )
                new_p, new_m, new_v = _adam_trees(
                    param._values(),
                    moment._values(),
                    velocity._values(),
                    grad_tree,
                    lr,
                    t,
                    bias1,
                    bias2,
                )
                new_param_trees.append(new_p)
                new_m_trees.append(new_m)
                new_v_trees.append(new_v)

            # Commit the fully computed trees first; the step count is
            # published last, so any failure leaves the pre-step state.
            for param, moment, velocity, p_tree, m_tree, v_tree in zip(
                params, moments, velocities,
                new_param_trees, new_m_trees, new_v_trees,
            ):
                param._set_values(p_tree)
                moment._set_values(m_tree)
                velocity._set_values(v_tree)
            self._adam_t = t

    # -- checkpoints --------------------------------------------------------

    def _snapshot_document(self):
        params = self.parameters()
        moments = self._adam_m
        velocities = self._adam_v
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
            "optim": {
                "t": self._adam_t,
                "m": [
                    {
                        "s": param.shape,
                        "v": moments[index].tolist()
                        if moments is not None
                        else _zeros(param.shape),
                    }
                    for index, param in enumerate(params)
                ],
                "v": [
                    {
                        "s": param.shape,
                        "v": velocities[index].tolist()
                        if velocities is not None
                        else _zeros(param.shape),
                    }
                    for index, param in enumerate(params)
                ],
            },
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

    def save(self, target):
        """Fix the current slice-boundary training state into *target*.

        *target* is either a filesystem path (``str``/``os.PathLike``), an
        existing directory used as an incremental checkpoint chain, or a
        ``bytearray`` used as an in-memory buffer. The snapshot records
        every parameter tensor, every accumulated gradient (zeros for
        parameters that have never received one), the optimizer state
        (step count plus both moment tensors, all zero at ``t == 0``
        before the first ``adam_step``), the current slice-boundary
        hidden state, the layer order with per-layer parameter shapes and
        the format version.

        A boundary exists before the first segment and after every
        ``forward`` (the hidden state returned by that forward *is* the
        boundary), so saving is allowed even while the latest forward has
        not been back-propagated yet: the snapshot fixes the boundary
        state, not the in-flight activations. Saving mutates no existing
        state and repeated saves of the same state produce identical
        bytes. The whole snapshot is taken under the container lock, so a
        concurrent step can never leave a half-written tensor in it.
        """
        with self._lock:
            document = self._snapshot_document()
            return _checkpoint.save_bytes(document, target)

    def load(self, source):
        """Restore state previously written by ``save``.

        *source* is a filesystem path, an incremental-chain directory or a
        bytes-like buffer. The whole checkpoint is validated before
        anything is applied: the format version (versions 1 and 2 are
        migrated item by item, the missing optimizer state starting at
        ``t == 0`` with all-zero moments, and the origin is recorded),
        every tensor and optimizer-moment shape and the layer order must
        match this container exactly. Hidden-state shapes are verified on
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
            new_moments = [
                Tensor(_rebuild_tree(entry["v"], entry["s"]))
                for entry in document["optim"]["m"]
            ]
            new_velocities = [
                Tensor(_rebuild_tree(entry["v"], entry["s"]))
                for entry in document["optim"]["v"]
            ]
            new_t = document["optim"]["t"]
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

            for param, values in zip(params, new_values):
                param._set_values(values)
            for param, grad in zip(params, new_grads):
                param.grad = grad
            self._adam_t = new_t
            self._adam_m = new_moments
            self._adam_v = new_velocities
            self._last_hidden = new_hidden
            self._hidden_shapes = new_hidden_shapes
            self._last_output = None
            self._pending_backward = False
            self._replay_batch = None
            self._replay_hidden = None
            self._forward_fingerprint = None
            self._loaded_from_version = source_version
            if new_hidden is None:
                return None
            return [Tensor(slot.tolist()) for slot in new_hidden]

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

        saved_optim = document.get("optim")
        if not isinstance(saved_optim, dict):
            raise ValueError("checkpoint optimizer state is malformed")
        saved_t = saved_optim.get("t")
        if isinstance(saved_t, bool) or not isinstance(saved_t, int) or saved_t < 0:
            raise ValueError("checkpoint optimizer step count must be a non-negative int")
        for group_name in ("m", "v"):
            group = saved_optim.get(group_name)
            if not isinstance(group, list) or len(group) != len(params):
                raise ValueError(
                    f"checkpoint optimizer {group_name} count does not match "
                    "the current model"
                )
            for index, (param, entry) in enumerate(zip(params, group)):
                if not isinstance(entry, dict) or entry.get("s") != param.shape:
                    raise ValueError(
                        f"checkpoint optimizer {group_name} {index} shape does "
                        "not match the current model"
                    )
                _rebuild_tree(entry["v"], entry["s"])
        if saved_t == 0:
            for group_name in ("m", "v"):
                for index, entry in enumerate(saved_optim[group_name]):
                    if any(leaf != 0 for leaf in _walk_leaves(entry["v"])):
                        raise ValueError(
                            f"checkpoint optimizer {group_name} {index} must be "
                            "all zeros at step count 0"
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
                    f"match the current model"
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


def _adam_trees(param_tree, moment_tree, velocity_tree, grad_tree,
                lr, t, bias1, bias2):
    """Return fresh ``(param, m, v)`` trees after one Adam update.

    The recursion order is the single fixed leaf order used on every step
    and after every resume, so a continued run computes each float in the
    exact same order as an uninterrupted one.
    """
    if isinstance(param_tree, list):
        p_out = []
        m_out = []
        v_out = []
        for p, m, v, g in zip(param_tree, moment_tree, velocity_tree, grad_tree):
            np_, nm, nv = _adam_trees(p, m, v, g, lr, t, bias1, bias2)
            p_out.append(np_)
            m_out.append(nm)
            v_out.append(nv)
        return p_out, m_out, v_out
    g = float(grad_tree)
    m_new = _ADAM_BETA1 * moment_tree + _ADAM_ONE_MINUS_BETA1 * g
    g_squared = g * g
    v_new = _ADAM_BETA2 * velocity_tree + _ADAM_ONE_MINUS_BETA2 * g_squared
    m_hat = m_new / bias1
    v_hat = v_new / bias2
    param_new = param_tree - lr * m_hat / (math.sqrt(v_hat) + _ADAM_EPSILON)
    return param_new, m_new, v_new


def _rebuild_tree(value, shape):
    """Validate decoded leaves against *shape* and return a fresh nested tree."""
    if shape:
        if not isinstance(value, list) or len(value) != shape[0]:
            raise ValueError("checkpoint tensor values do not match their shape")
        return [_rebuild_tree(item, shape[1:]) for item in value]
    if isinstance(value, (list, bool)) or not isinstance(value, (int, float)):
        raise ValueError("checkpoint tensor values do not match their shape")
    return value
