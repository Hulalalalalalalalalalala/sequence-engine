"""Built-in self-checks for the engine. Runs in memory; writes no files.

The concurrency/chain groups use temporary directories that are removed
before the process returns, so no files are left behind.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import threading

from . import checkpoint as _checkpoint
from .sequential import Sequential
from .tensor import Tensor


class _SelfTestFailure(Exception):
    pass


def _check(condition, message):
    if not condition:
        raise _SelfTestFailure(message)


def _expect(error, fn, message):
    try:
        fn()
    except error:
        return
    except Exception as exc:
        raise _SelfTestFailure(
            f"{message} (raised {type(exc).__name__}: {exc})"
        ) from exc
    raise _SelfTestFailure(f"{message} (no error raised)")


# ---------------------------------------------------------------------------
# Demo caller-provided layer: one tanh RNN step over a batch,
# y = tanh(x @ Wxh + h @ Whh + b); the hidden slot is y itself.
# A scalar upstream g means d(loss)/d(output) = g for every output element.
# ---------------------------------------------------------------------------


class _RNNStep:
    def __init__(self, n_in, n_hid, wxh, whh, bias):
        self.n_in = n_in
        self.n_hid = n_hid
        self.wxh = Tensor(wxh)
        self.whh = Tensor(whh)
        self.bias = Tensor(bias)
        self._cache = None

    def parameters(self):
        return [self.wxh, self.whh, self.bias]

    def forward(self, x, hidden):
        xs = x.tolist()
        rows = len(xs)
        if hidden is None:
            hs = [[0.0] * self.n_hid for _ in range(rows)]
        else:
            hs = hidden.tolist()
            if len(hs) != rows or any(len(row) != self.n_hid for row in hs):
                raise ValueError("hidden state slot has the wrong shape")
        wxh, whh, b = self.wxh.tolist(), self.whh.tolist(), self.bias.tolist()
        y = []
        for r in range(rows):
            row = []
            for j in range(self.n_hid):
                acc = b[j]
                for i in range(self.n_in):
                    acc += xs[r][i] * wxh[i][j]
                for k in range(self.n_hid):
                    acc += hs[r][k] * whh[k][j]
                row.append(math.tanh(acc))
            y.append(row)
        self._cache = (xs, hs, y)
        return Tensor(y), Tensor(y)

    def backward(self, upstream):
        xs, hs, y = self._cache
        rows = len(xs)
        if isinstance(upstream, Tensor):
            dy = upstream.tolist()
        elif isinstance(upstream, bool):
            raise ValueError("upstream gradient must be numeric")
        elif isinstance(upstream, (int, float)):
            dy = [[float(upstream)] * self.n_hid for _ in range(rows)]
        else:
            dy = upstream
        wxh, whh = self.wxh.tolist(), self.whh.tolist()
        d = [
            [dy[r][j] * (1.0 - y[r][j] ** 2) for j in range(self.n_hid)]
            for r in range(rows)
        ]
        g_wxh = [
            [sum(xs[r][i] * d[r][j] for r in range(rows)) for j in range(self.n_hid)]
            for i in range(self.n_in)
        ]
        g_whh = [
            [sum(hs[r][k] * d[r][j] for r in range(rows)) for j in range(self.n_hid)]
            for k in range(self.n_hid)
        ]
        g_b = [sum(d[r][j] for r in range(rows)) for j in range(self.n_hid)]
        _accumulate(self.wxh, g_wxh)
        _accumulate(self.whh, g_whh)
        _accumulate(self.bias, g_b)
        dx = [
            [sum(d[r][j] * wxh[i][j] for j in range(self.n_hid)) for i in range(self.n_in)]
            for r in range(rows)
        ]
        return Tensor(dx)


def _accumulate(param, grad):
    if param.grad is None:
        param.grad = Tensor(grad)
    else:
        param.grad = Tensor(_elementwise_add(param.grad.tolist(), grad))


def _elementwise_add(a, b):
    if isinstance(a, list):
        return [_elementwise_add(x, y) for x, y in zip(a, b)]
    return a + b


# ---------------------------------------------------------------------------
# Deterministic fixtures (no randomness anywhere).
# ---------------------------------------------------------------------------

_N_IN, _N_H1, _N_H2, _BATCH = 2, 3, 2, 2
_KEYS = ("wxh1", "whh1", "b1", "wxh2", "whh2", "b2")

_SEG1 = [[0.5, -0.25], [0.125, 0.75]]
_SEG2 = [[-0.4, 0.3], [0.6, 0.2]]


def _base_weights():
    return {
        "wxh1": [[0.1, -0.2, 0.3], [0.4, 0.05, -0.1]],
        "whh1": [[0.2, -0.1, 0.0], [0.1, 0.3, -0.2], [-0.3, 0.1, 0.2]],
        "b1": [0.01, -0.02, 0.03],
        "wxh2": [[0.2, -0.3], [0.1, 0.4], [-0.2, 0.1]],
        "whh2": [[0.3, -0.1], [0.2, 0.2]],
        "b2": [-0.01, 0.02],
    }


def _build(weights):
    layers = [
        _RNNStep(_N_IN, _N_H1, weights["wxh1"], weights["whh1"], weights["b1"]),
        _RNNStep(_N_H1, _N_H2, weights["wxh2"], weights["whh2"], weights["b2"]),
    ]
    return Sequential(layers), layers


def _flatten(nested):
    flat = []

    def rec(value):
        if isinstance(value, list):
            for item in value:
                rec(item)
        else:
            flat.append(value)

    rec(nested)
    return flat


def _flat_weights(weights):
    return _flatten([weights[key] for key in _KEYS])


def _unflatten(flat):
    iterator = iter(flat)

    def rec(template):
        if isinstance(template, list):
            return [rec(item) for item in template]
        return next(iterator)

    base = _base_weights()
    return {key: rec(base[key]) for key in _KEYS}


def _total(out):
    return sum(value for row in out.tolist() for value in row)


def _objective(weights, segment):
    # The caller's scalar loss is L = sum(outputs); the seed passed to
    # backward() is L itself, so the gradients produced are those of
    # F = 1/2 * L^2. The finite-difference check below verifies exactly
    # that objective.
    seq, _ = _build(weights)
    out, _ = seq.forward(Tensor(segment))
    loss = _total(out)
    return 0.5 * loss * loss


# ---------------------------------------------------------------------------
# Check groups.
# ---------------------------------------------------------------------------


def _check_tensor_basics():
    t = Tensor([[1, 2], [3, 4]])
    _check(t.shape == [2, 2], "shape must reflect nesting")
    _check(t.tolist() == [[1, 2], [3, 4]], "tolist must round-trip the data")
    copied = t.tolist()
    copied[0][0] = 99
    _check(t.tolist()[0][0] == 1, "tolist must return a copy")
    scalar = Tensor(2.5)
    _check(scalar.shape == [] and scalar.tolist() == 2.5, "scalar tensor")
    _check(t.grad is None, "a fresh tensor carries no gradient")


def _check_tensor_validation():
    _expect(ValueError, lambda: Tensor([]), "empty tensor data")
    _expect(ValueError, lambda: Tensor([[]]), "empty inner tensor data")
    _expect(ValueError, lambda: Tensor([[1.0], [2.0, 3.0]]), "ragged tensor data")
    _expect(ValueError, lambda: Tensor([1.0, [2.0]]), "uneven nesting depth")
    _expect(ValueError, lambda: Tensor(True), "boolean tensor data")
    _expect(ValueError, lambda: Tensor([1.0, False]), "nested boolean tensor data")
    _expect(ValueError, lambda: Tensor("nope"), "non-numeric tensor data")


def _check_container_validation():
    _expect(ValueError, lambda: Sequential([]), "empty container")
    _expect(ValueError, lambda: Sequential([object()]), "layer without capabilities")

    class _ForwardOnly:
        def forward(self, x, hidden):
            return x, hidden

    _expect(
        ValueError,
        lambda: Sequential([_ForwardOnly()]),
        "layer missing backward/parameters",
    )

    seq, _ = _build(_base_weights())
    _expect(ValueError, lambda: seq.forward("nope"), "non-tensor batch")
    _expect(ValueError, lambda: seq.forward(Tensor(1.0)), "scalar (empty) batch")
    _expect(
        ValueError,
        lambda: seq.forward(Tensor(_SEG1), [Tensor([[0.0, 0.0]])]),
        "wrong number of hidden slots",
    )
    _expect(
        ValueError,
        lambda: seq.forward(
            Tensor(_SEG1), [Tensor([[0.0] * _N_H1] * _BATCH), "not a tensor"]
        ),
        "non-tensor hidden slot",
    )
    seq.forward(Tensor(_SEG1))
    bad_shape = [
        Tensor([[0.0] * _N_H1] * _BATCH),
        Tensor([[0.0] * _N_H1] * _BATCH),
    ]
    _expect(
        ValueError,
        lambda: seq.forward(Tensor(_SEG2), bad_shape),
        "hidden slot shape mismatch",
    )


def _check_forward_and_hidden():
    seq, _ = _build(_base_weights())
    out, hidden = seq.forward(Tensor(_SEG1))
    _check(
        isinstance(out, Tensor) and out.shape == [_BATCH, _N_H2],
        "output is a tensor whose shape carries the batch size",
    )
    _check(
        isinstance(hidden, list) and len(hidden) == 2,
        "hidden state stacks one slot per layer",
    )
    _check(
        hidden[0].shape == [_BATCH, _N_H1] and hidden[1].shape == [_BATCH, _N_H2],
        "hidden slot shapes",
    )
    _check(
        hidden[1].tolist() == out.tolist(),
        "last layer's hidden slot equals its output in this demo",
    )
    out_again, _ = seq.forward(Tensor(_SEG1))
    _check(
        out_again.tolist() == out.tolist(),
        "dropping the hidden state restarts from zeros deterministically",
    )
    out_carried, _ = seq.forward(Tensor(_SEG2), hidden)
    fresh, _ = _build(_base_weights())
    out_zero, _ = fresh.forward(Tensor(_SEG2))
    _check(
        out_carried.tolist() != out_zero.tolist(),
        "carried hidden state must influence the next segment",
    )


def _check_backward_contract():
    seq, _ = _build(_base_weights())
    _expect(RuntimeError, lambda: seq.backward(1.0), "backward before forward")
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    _expect(
        RuntimeError, lambda: seq.backward(1.0), "second backward for one forward"
    )
    for param in seq.parameters():
        _check(param.grad is not None, "backward must leave gradients on parameters")


def _check_gradients_numeric():
    weights = _base_weights()
    seq, _ = _build(weights)
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    analytic = _flatten([p.grad.tolist() for p in seq.parameters()])
    flat = _flat_weights(weights)
    eps = 1e-6
    for index in range(len(flat)):
        plus = list(flat)
        plus[index] += eps
        minus = list(flat)
        minus[index] -= eps
        numeric = (
            _objective(_unflatten(plus), _SEG1) - _objective(_unflatten(minus), _SEG1)
        ) / (2.0 * eps)
        _check(
            abs(numeric - analytic[index])
            <= 1e-4 * max(1.0, abs(numeric), abs(analytic[index])),
            f"gradient mismatch at flat index {index}: "
            f"analytic {analytic[index]}, numeric {numeric}",
        )


def _check_truncation():
    weights = _base_weights()
    seq, _ = _build(weights)
    out1, hidden1 = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out1))
    grads_seg1 = [p.grad.tolist() for p in seq.parameters()]
    out2, _ = seq.forward(Tensor(_SEG2), hidden1)
    seq.backward(_total(out2))
    grads_both = [p.grad.tolist() for p in seq.parameters()]

    # Reference: segment 2 alone, hidden carried as detached values.
    ref, _ = _build(weights)
    detached = [Tensor(slot.tolist()) for slot in hidden1]
    out2_ref, _ = ref.forward(Tensor(_SEG2), detached)
    _check(
        out2_ref.tolist() == out2.tolist(),
        "carried hidden feeds values identically across the segment boundary",
    )
    ref.backward(_total(out2_ref))
    grads_seg2 = [p.grad.tolist() for p in ref.parameters()]

    # Accumulation is plain ordered float addition of the two segments.
    for first, second, total in zip(grads_seg1, grads_seg2, grads_both):
        _check(
            _elementwise_add(first, second) == total,
            "accumulated gradients must equal seg1 + seg2 exactly",
        )

    # No leak: a stack that only ever backprops segment 2 must see nothing
    # of segment 1, bitwise.
    fresh, _ = _build(weights)
    _, hidden1_fresh = fresh.forward(Tensor(_SEG1))
    out2_fresh, _ = fresh.forward(Tensor(_SEG2), hidden1_fresh)
    fresh.backward(_total(out2_fresh))
    grads_fresh = [p.grad.tolist() for p in fresh.parameters()]
    _check(
        grads_fresh == grads_seg2,
        "backward on a later segment must not add gradients for an earlier one",
    )


def _check_zero_grad_and_parameters():
    seq, layers = _build(_base_weights())
    params = seq.parameters()
    expected = [
        layers[0].wxh,
        layers[0].whh,
        layers[0].bias,
        layers[1].wxh,
        layers[1].whh,
        layers[1].bias,
    ]
    _check(
        len(params) == len(expected) and all(a is b for a, b in zip(params, expected)),
        "parameters() must list layer parameters in registration order",
    )
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    _check(
        any(value != 0 for p in params for value in _flatten(p.grad.tolist())),
        "gradients should be non-zero before zero_grad",
    )
    seq.zero_grad()
    for param in params:
        _check(
            param.grad is not None
            and all(value == 0 for value in _flatten(param.grad.tolist())),
            "zero_grad must reset every gradient to zeros",
        )


# ---------------------------------------------------------------------------
# New capabilities: one-step update, tuple loss, retry contract, checkpoints.
# ---------------------------------------------------------------------------

_LR = 0.1
_SEG3 = [[0.3, -0.7], [-0.2, 0.4]]


def _fresh_stack(weights=None):
    return _build(weights if weights is not None else _base_weights())


def _check_update_step():
    seq, _ = _fresh_stack()
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    before = [p.tolist() for p in seq.parameters()]
    grads_before = [p.grad.tolist() for p in seq.parameters()]
    seq.update(_LR)
    for param, old, grad in zip(seq.parameters(), before, grads_before):
        expected = _elementwise_scale_sub(old, grad, _LR)
        _check(
            param.tolist() == expected,
            "update must apply theta <- theta - lr*grad in place",
        )
        _check(
            param.grad.tolist() == grad,
            "update must not clear accumulated gradients (zero_grad does)",
        )
    for bad_lr in (float("nan"), float("inf"), float("-inf")):
        _expect(
            ValueError, lambda bad=bad_lr: seq.update(bad), "non-finite learning rate"
        )
    _expect(ValueError, lambda: seq.update("0.1"), "non-numeric learning rate")


def _elementwise_scale_sub(values, grads, lr):
    if isinstance(values, list):
        return [
            _elementwise_scale_sub(a, b, lr) for a, b in zip(values, grads)
        ]
    return values - lr * grads


class _BoomOnceRNN(_RNNStep):
    """RNN step whose first backward attempt raises, then succeeds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._boom = True

    def backward(self, upstream):
        if self._boom:
            self._boom = False
            raise RuntimeError("transient failure during backward")
        return super().backward(upstream)


def _check_tuple_loss_and_retry():
    # Tuple form (output, upstream) is equivalent to the scalar seed form.
    seq, _ = _fresh_stack()
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward((out, 1.5))
    ref, _ = _fresh_stack()
    ref_out, _ = ref.forward(Tensor(_SEG1))
    ref.backward(1.5)
    _check(
        [p.grad.tolist() for p in seq.parameters()]
        == [p.grad.tolist() for p in ref.parameters()],
        "tuple loss (output, upstream) must seed backward identically",
    )
    _expect(
        RuntimeError, lambda: seq.backward((out, 1.0)), "second backward after tuple"
    )
    stale, _ = _fresh_stack()
    stale.forward(Tensor(_SEG1))
    other, _ = _fresh_stack()
    other_out, _ = other.forward(Tensor(_SEG1))
    _expect(
        ValueError,
        lambda: stale.backward((other_out, 1.0)),
        "tuple loss must carry the tensor of the preceding forward",
    )
    _expect(
        ValueError,
        lambda: stale.backward((out,)),
        "one-element tuple loss is rejected",
    )
    _expect(
        ValueError,
        lambda: stale.backward((out, float("nan"))),
        "non-numeric tuple upstream is rejected",
    )

    # A failed backward must be retriable, and must not count as the one
    # backward of the forward; partial accumulation is rolled back first.
    weights = _base_weights()
    boom = _BoomOnceRNN(
        _N_IN, _N_H1, weights["wxh1"], weights["whh1"], weights["b1"]
    )
    quiet = _RNNStep(_N_H1, _N_H2, weights["wxh2"], weights["whh2"], weights["b2"])
    broken = Sequential([boom, quiet])
    broken.forward(Tensor(_SEG1))
    _expect(RuntimeError, lambda: broken.backward(1.0), "first attempt fails")
    # reverse order ran the quiet layer first; its partial grad must be gone
    _check(quiet.wxh.grad is None, "failed backward rolls partial gradients back")
    broken.backward(1.0)  # retry is allowed and is not a "second" backward
    good, _ = _fresh_stack(weights)
    good.forward(Tensor(_SEG1))
    good.backward(1.0)
    _check(
        [p.grad.tolist() for p in broken.parameters()]
        == [p.grad.tolist() for p in good.parameters()],
        "retried backward must produce the same gradients as a clean run",
    )
    _expect(
        RuntimeError, lambda: broken.backward(1.0), "the retry consumed the pass"
    )


def _continuity_run(resume_from=None):
    """Three segments with an update between segments two and three."""
    seq, _ = _fresh_stack()
    out1, hidden1 = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out1))
    if resume_from is not None:
        seq, _ = _fresh_stack()
        hidden1 = seq.load(resume_from)
    out2, hidden2 = seq.forward(Tensor(_SEG2), hidden1)
    seq.backward(_total(out2))
    seq.update(_LR)
    carry = [Tensor(slot.tolist()) for slot in hidden2]
    out3, _ = seq.forward(Tensor(_SEG3), carry)
    seq.backward(_total(out3))
    return (
        [p.tolist() for p in seq.parameters()],
        [p.grad.tolist() for p in seq.parameters()],
    )


def _check_checkpoint_continuity():
    uninterrupted = _continuity_run()

    seq, _ = _fresh_stack()
    out1, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out1))

    buffer = bytearray()
    seq.save(buffer)
    first_bytes = bytes(buffer)
    # Repeated saves fix the same state byte for byte and change nothing.
    params_before = [p.tolist() for p in seq.parameters()]
    grads_before = [p.grad.tolist() for p in seq.parameters()]
    seq.save(buffer)
    seq.save(buffer)
    _check(bytes(buffer) == first_bytes, "repeated saves must be identical")
    _check(
        [p.tolist() for p in seq.parameters()] == params_before
        and [p.grad.tolist() for p in seq.parameters()] == grads_before,
        "save must not mutate parameters or gradients",
    )

    resumed = _continuity_run(resume_from=buffer)
    _check(
        resumed == uninterrupted,
        "save -> load -> continue slices, backward and update must match an "
        "uninterrupted run bit for bit",
    )


def _check_checkpoint_roundtrip():
    # Never-trained model: gradients are absent yet saved and read back zero.
    seq, _ = _fresh_stack()
    buffer = bytearray()
    seq.save(buffer)
    restored, _ = _fresh_stack()
    restored.zero_grad()
    hidden = restored.load(buffer)
    _check(hidden is None, "checkpoint before any forward fixes no hidden state")
    for param in restored.parameters():
        _check(
            all(value == 0 for value in _flatten(param.grad.tolist())),
            "parameters without gradients round-trip as zero gradients",
        )

    # Bitwise float fidelity, including negative zero, across memory save.
    tricky, _ = _fresh_stack()
    tricky.parameters()[0]._set_values(
        [[1.5, -0.0, 3.25], [100000000000000.0, -2.5, 0.0625]]
    )
    tricky.zero_grad()
    tricky.parameters()[0].grad._set_values(
        [[-0.0, 2.0, -3.0], [0.0, 1.0, -1.0]]
    )
    raw = bytearray()
    tricky.save(raw)  # segment boundary: no forward pending
    target, _ = _fresh_stack()
    target.load(bytes(raw))
    values = target.parameters()[0].tolist()
    grad_values = target.parameters()[0].grad.tolist()
    _check(
        values == [[1.5, -0.0, 3.25], [100000000000000.0, -2.5, 0.0625]],
        "param values restore",
    )
    _check(
        grad_values == [[-0.0, 2.0, -3.0], [0.0, 1.0, -1.0]],
        "gradient values restore",
    )
    _check(
        struct.pack("<d", values[0][1]) == struct.pack("<d", -0.0),
        "negative zero in parameters keeps its sign",
    )
    _check(
        struct.pack("<d", grad_values[0][0]) == struct.pack("<d", -0.0),
        "negative zero in gradients keeps its sign",
    )

    # After a completed segment the checkpoint also fixes boundary hidden.
    out2, hidden_state = tricky.forward(Tensor(_SEG1))
    tricky.backward(_total(out2))
    raw2 = bytearray()
    tricky.save(raw2)
    target2, _ = _fresh_stack()
    restored_hidden = target2.load(bytes(raw2))
    _check(
        [s.tolist() for s in restored_hidden]
        == [s.tolist() for s in hidden_state],
        "load restores the slice-boundary hidden tensors",
    )

    # Structural rejection on the in-memory form: whole document refused.
    def expect_rejected(mutated, message):
        victim, _ = _fresh_stack()
        _expect(ValueError, lambda: victim.load(mutated), message)

    expect_rejected(b"", "empty buffer")
    expect_rejected(b"not a checkpoint at all", "foreign bytes")
    expect_rejected(bytes(raw2)[: len(raw2) // 2], "half-written (torn) buffer")
    expect_rejected(bytes(raw2)[:-1], "truncated trailer")
    flipped = bytearray(raw2)
    flipped[len(flipped) // 2] ^= 0xFF
    expect_rejected(bytes(flipped), "flipped payload byte fails the CRC")
    expect_rejected(
        bytes(raw2)[:8] + struct.pack("<I", 999) + bytes(raw2)[12:],
        "unsupported version",
    )

    # A checkpoint from a model with different parameter shapes is refused.
    other_weights = _base_weights()
    other_weights["b1"] = [0.01, -0.02]  # shape [2] instead of [3]
    different, _ = _fresh_stack(other_weights)
    other_buffer = bytearray()
    different.save(other_buffer)
    victim, _ = _fresh_stack()
    _expect(
        ValueError,
        lambda: victim.load(bytes(other_buffer)),
        "tensor shape mismatch rejects the whole checkpoint",
    )

    # Layer-order mismatch: same parameter shapes, different layer kinds.
    model, _ = _fresh_stack()
    model.zero_grad()
    wrong_kind_doc = {
        "params": [{"s": p.shape, "v": p.tolist()} for p in model.parameters()],
        "grads": [
            {"s": p.shape, "v": p.grad.tolist()} for p in model.parameters()
        ],
        "hidden": None,
        "layers": [
            {"kind": "SomethingElse", "shapes": [[2, 3], [3, 3], [3]]},
            {"kind": "_RNNStep", "shapes": [[3, 2], [2, 2], [2]]},
        ],
        "pending": False,
    }
    victim2, _ = _fresh_stack()
    _expect(
        ValueError,
        lambda: victim2.load(_checkpoint.build_bytes(wrong_kind_doc)),
        "layer kind/order mismatch rejects the whole checkpoint",
    )


# ---------------------------------------------------------------------------
# Concurrency checks
# ---------------------------------------------------------------------------


def _fixed_gradient_state():
    """A trained boundary state with fixed, nonzero gradients."""
    seq, _ = _fresh_stack()
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    buf = bytearray()
    seq.save(buf)
    base_params = [p.tolist() for p in seq.parameters()]
    grads = [p.grad.tolist() for p in seq.parameters()]
    return seq, bytes(buf), base_params, grads


def _subtract_path(base, grads, lr, steps):
    """Reference path P_k = the engine's own repeated theta <- theta - lr*grad."""
    current = [_deep(base_j) for base_j in base]
    path = [[_deep(j) for j in current]]
    for _ in range(steps):
        current = [_elementwise_scale_sub(p, g, lr) for p, g in zip(current, grads)]
        path.append([_deep(j) for j in current])
    return path


def _deep(value):
    if isinstance(value, list):
        return [_deep(item) for item in value]
    return value


def _find_path_k(observed_params, path):
    """Return the unique k whose parameter vector equals *observed_params*."""
    matches = []
    for k, snapshot in enumerate(path):
        if observed_params == snapshot:
            matches.append(k)
    if len(matches) != 1:
        return None
    return matches[0]


def _check_concurrent_update_save_load():
    # Fixed gradients: every update subtracts lr * grad along one
    # deterministic path; a load() resets the model to k = 0. Any
    # snapshot taken concurrently must therefore sit at a single integer
    # position k on that path for EVERY parameter -- a half-applied
    # update or load would put different parameters at different k.
    seq, buf_b, base_params, grads = _fixed_gradient_state()
    lr = 0.05
    n_updates = 60
    n_loads = 25
    n_snaps = 400
    path = _subtract_path(base_params, grads, lr, n_updates + 5)

    snapshots = []
    errors = []

    def observer_stepped(barrier):
        # Deterministic alternation: snapshot after every single update.
        for _ in range(n_updates):
            barrier.wait()
            buf = bytearray()
            seq.save(buf)
            snapshots.append(bytes(buf))

    def updater_stepped(barrier):
        for _ in range(n_updates):
            seq.update(lr)
            barrier.wait()

    barrier = threading.Barrier(2)
    t_obs = threading.Thread(target=observer_stepped, args=(barrier,))
    t_upd = threading.Thread(target=updater_stepped, args=(barrier,))
    t_obs.start()
    t_upd.start()
    t_obs.join()
    t_upd.join()
    _check(not errors, f"stepped race raised: {errors!r}")
    # Every interleaved snapshot is a complete boundary at one single
    # position k on the serial path (never different parameters at
    # different k's). When the observer reaches the barrier the next
    # update may already have started, so k can be one step ahead.
    seen_positions = []
    for index, raw in enumerate(snapshots):
        victim, _ = _fresh_stack()
        hidden = victim.load(raw)
        _check(hidden is not None, "raced snapshot must carry boundary hidden state")
        observed = [p.tolist() for p in victim.parameters()]
        k = _find_path_k(observed, path)
        _check(
            k is not None and k in (index + 1, index + 2),
            f"raced snapshot {index} is not one whole state at a serial "
            f"path position (found {k})",
        )
        seen_positions.append(k)

    # Free-running three-way race: updates walk the path, loads reset to
    # k = 0, snapshots must always be one globally consistent k.
    seq, _, _, _ = _fixed_gradient_state()
    snapshots = []

    def updater():
        try:
            for _ in range(n_updates):
                seq.update(lr)
        except Exception as exc:  # pragma: no cover - reported as failure
            errors.append(exc)

    def loader():
        try:
            for _ in range(n_loads):
                seq.load(buf_b)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def free_saver():
        for _ in range(n_snaps):
            buf = bytearray()
            seq.save(buf)
            snapshots.append(bytes(buf))

    threads = [
        threading.Thread(target=updater),
        threading.Thread(target=loader),
        threading.Thread(target=free_saver),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _check(not errors, f"free race raised: {errors!r}")
    _check(len(snapshots) == n_snaps, "every concurrent save must complete")
    bad = 0
    for raw in snapshots:
        victim, _ = _fresh_stack()
        victim.load(raw)
        observed = [p.tolist() for p in victim.parameters()]
        if _find_path_k(observed, path) is None:
            bad += 1
    _check(bad == 0, f"{bad}/{len(snapshots)} concurrent snapshots were torn")


def _check_concurrent_writers():
    # Two threads repeatedly save FULL checkpoints to the same path while
    # a third walks parameters; the file must always be one complete
    # checkpoint -- never half of one save mixed with another.
    seq, _ = _fresh_stack()
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))

    errors = []
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "shared.ckp")
        seq.save(path)

        def full_writer():
            try:
                for _ in range(40):
                    try:
                        seq.save(path)
                    except RuntimeError:
                        # The trainer may be mid-segment (forward without
                        # backward); a boundary refusal is legal, retry.
                        pass
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def chain_writer():
            try:
                chain_dir = os.path.join(td, "chain")
                for _ in range(40):
                    try:
                        seq.save_incremental(chain_dir)
                    except RuntimeError:
                        # Another thread may hold the container mid-segment
                        # during its own forward/backward; a refused save is
                        # a legal boundary outcome, just retry.
                        pass
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def trainer():
            try:
                for k in range(40):
                    out, _ = seq.forward(Tensor(_SEG2 if k % 2 else _SEG1))
                    seq.backward(_total(out))
                    if k % 5 == 0:
                        seq.update(0.01)
                    seq.save(path)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=full_writer),
            threading.Thread(target=chain_writer),
            threading.Thread(target=trainer),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        _check(not errors, f"concurrent writers raised: {errors!r}")

        # Final full file is one complete, loadable boundary checkpoint.
        final_raw = open(path, "rb").read()
        victim, _ = _fresh_stack()
        victim.load(final_raw)
        _check(
            _checkpoint.load_bytes(path)["pending"] is False,
            "concurrent full saves always land on a segment boundary",
        )
        # The incremental chain reassembles without error.
        rechain, _ = _fresh_stack()
        rechain.load_incremental(os.path.join(td, "chain"))
        full = bytearray()
        rechain.save(full)
        _check(
            _checkpoint.load_bytes(bytes(full))["pending"] is False,
            "concurrent incremental chain reassembles to a boundary state",
        )


# ---------------------------------------------------------------------------
# Version 1 reading / migration
# ---------------------------------------------------------------------------


def _forge_v1(document):
    header, payload = _checkpoint._freeze(
        document, version=1, source_version=None
    )
    return _checkpoint._frame(header, payload, 1)


def _repack_full(raw, mutate_header, version=None):
    """Rewrite a full checkpoint with a mutated header and valid CRC."""
    if version is None:
        version = struct.unpack("<I", raw[8:12])[0]
    (hlen,) = struct.unpack("<Q", raw[12:20])
    header = __import__("json").loads(raw[20 : 20 + hlen])
    payload = raw[20 + hlen : raw.rfind(_checkpoint.END_MAGIC)]
    mutate_header(header)
    new_header = __import__("json").dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    body = (
        raw[:8]
        + struct.pack("<I", version)
        + struct.pack("<Q", len(new_header))
        + new_header
        + payload
    )
    crc = __import__("zlib").crc32(body[20:])
    leaf_count = len(payload) // 9
    return body + _checkpoint.END_MAGIC + struct.pack("<QI", leaf_count, crc)


def _check_version1_migration():
    seq, _ = _fresh_stack()
    out1, hidden1 = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out1))
    v2_buf = bytearray()
    seq.save(v2_buf)
    document = _checkpoint.load_bytes(bytes(v2_buf))
    _check(document["src"] == 2, "native checkpoints record source version 2")

    raw_v1 = _forge_v1(document)
    _check(struct.unpack("<I", raw_v1[8:12])[0] == 1, "forged file really is v1")
    migrated = _checkpoint.parse_bytes(raw_v1)
    _check(migrated["src"] == 1, "loaded v1 file records source version 1")

    # Item-by-item migration reproduces state and boundary hidden exactly.
    restored, _ = _fresh_stack()
    restored_hidden = restored.load(raw_v1)
    _check(restored._loaded_src == 1, "model remembers the migration source")
    _check(
        [p.tolist() for p in restored.parameters()]
        == [p.tolist() for p in seq.parameters()],
        "v1 parameters migrate unchanged",
    )
    _check(
        [p.grad.tolist() for p in restored.parameters()]
        == [p.grad.tolist() for p in seq.parameters()],
        "v1 gradients migrate unchanged",
    )
    _check(
        [s.tolist() for s in restored_hidden] == [s.tolist() for s in hidden1],
        "v1 boundary hidden migrates unchanged",
    )

    # Continuity through a migrated checkpoint is bitwise identical.
    migrated_buf = bytearray()
    restored.save(migrated_buf)
    resumed = _continuity_run(resume_from=raw_v1)
    _check(
        resumed == _continuity_run(),
        "training continued after a v1 load matches an uninterrupted run",
    )

    # Migration failures reject the WHOLE file; the model is untouched.
    victim, _ = _fresh_stack()
    before = [p.tolist() for p in victim.parameters()]

    def reject(raw, message):
        _expect(ValueError, lambda: _checkpoint.parse_bytes(raw), message)
        fresh, _ = _fresh_stack()
        _expect(ValueError, lambda: fresh.load(raw), message + " via model.load")

    # shape/value disagreement introduced mid-document
    reject(
        _repack_full(raw_v1, lambda h: h["params"].__setitem__(0, [7])),
        "half-applicable v1 migration (parameter shape disagreement)",
    )
    # field added and field removed at the version boundary
    reject(
        _repack_full(raw_v1, lambda h: h.update(src=1)),
        "v1 header with an unexpected new field",
    )
    reject(
        _repack_full(raw_v1, lambda h: h.pop("pending")),
        "v1 header missing a field",
    )
    reject(
        _repack_full(bytes(v2_buf), lambda h: h.pop("src"), version=2),
        "v2 header missing the source version",
    )
    reject(
        _repack_full(bytes(v2_buf), lambda h: h.update(extra=1), version=2),
        "v2 header with an unknown field",
    )

    # Non-finite leaf smuggled past the framing CRC is caught item by item.
    import zlib

    broken = bytearray(raw_v1)
    (hlen,) = struct.unpack("<Q", bytes(broken[12:20]))
    pstart, pend = 20 + hlen, broken.rfind(_checkpoint.END_MAGIC)
    planted = False
    for i in range(pstart, pend, 9):
        if broken[i] == ord("f"):
            broken[i + 1 : i + 9] = struct.pack("<d", float("inf"))
            planted = True
            break
    _check(planted, "test fixture must contain a float leaf")
    broken[-4:] = struct.pack("<I", zlib.crc32(bytes(broken[20:pend])))
    reject(bytes(broken), "non-finite float in migrated v1 payload")

    _check(
        [p.tolist() for p in victim.parameters()] == before,
        "rejected migrations leave the container untouched",
    )


# ---------------------------------------------------------------------------
# Incremental chain checks
# ---------------------------------------------------------------------------


class _StatelessScale:
    """Layer with ZERO parameters: y = 0.5 * x; the hidden slot is y."""

    def __init__(self):
        self._cache = None

    def parameters(self):
        return []

    def forward(self, x, hidden):
        xs = x.tolist()
        y = [[0.5 * v for v in row] for row in xs]
        self._cache = xs
        return Tensor(y), Tensor(y)

    def backward(self, upstream):
        if isinstance(upstream, Tensor):
            dy = upstream.tolist()
        elif isinstance(upstream, bool):
            raise ValueError("upstream must be numeric")
        else:
            dy = [[float(upstream)] * len(self._cache[0]) for _ in self._cache]
        return Tensor([[0.5 * v for v in row] for row in dy])


def _check_incremental_chains():
    with tempfile.TemporaryDirectory() as td:
        chain_dir = os.path.join(td, "chain")
        seq, _ = _fresh_stack()

        seq1 = seq.save_incremental(chain_dir)
        _check(seq1 == 1, "first chain entry is sequence 1")
        files = sorted(os.listdir(chain_dir))
        _check(
            files == [".seqckp.lock", "00000001.delta", "base.ckp", "manifest.json"],
            f"chain directory layout: {files}",
        )

        # No state change: the delta writes no layer bodies but the chain
        # advances and reassembly is still bitwise identical to a full save.
        no_change_size = os.path.getsize(os.path.join(chain_dir, "00000001.delta"))
        seq.save_incremental(chain_dir)
        delta2_size = os.path.getsize(os.path.join(chain_dir, "00000002.delta"))
        _check(
            delta2_size < os.path.getsize(os.path.join(chain_dir, "base.ckp")),
            "an unchanged snapshot writes far less than a full checkpoint",
        )
        _check(no_change_size > 0, "first delta still fixes references/state")

        def full_bytes(model):
            buf = bytearray()
            model.save(buf)
            return bytes(buf)

        # Walk several segments + updates, chaining between each step.
        history = []
        segments = (_SEG1, _SEG2, _SEG3, _SEG1)
        for index, segment in enumerate(segments):
            out, hidden = seq.forward(Tensor(segment))
            seq.backward(_total(out))
            if index % 2 == 1:
                seq.update(0.1)
            new_seq = seq.save_incremental(chain_dir)
            history.append((new_seq, full_bytes(seq)))
            # Loading the chain at every intermediate step must match.
            reader, _ = _fresh_stack()
            reader.load_incremental(chain_dir)
            _check(
                full_bytes(reader) == full_bytes(seq),
                f"chain reassembly after entry {new_seq} is bitwise identical "
                f"to the full save",
            )
        _check(
            [n for n, _ in history] == [3, 4, 5, 6],
            "chain sequence ids advance one per snapshot",
        )

        # Hidden state survives an incremental-only round trip.
        reader, _ = _fresh_stack()
        restored_hidden = reader.load_incremental(chain_dir)
        _check(
            [s.tolist() for s in restored_hidden] == [s.tolist() for s in hidden],
            "incremental load restores boundary hidden state",
        )

        # Empty-parameter layers take part deterministically.
        weights = _base_weights()
        rnn = _RNNStep(_N_IN, _N_H1, weights["wxh1"], weights["whh1"], weights["b1"])
        stateless = _StatelessScale()
        mixed = Sequential([rnn, stateless])
        mixed_dir = os.path.join(td, "mixed")
        mixed.save_incremental(mixed_dir)
        out, _ = mixed.forward(Tensor(_SEG1))
        mixed.backward(_total(out))
        mixed.update(0.1)
        mixed.save_incremental(mixed_dir)
        got, _ = _fresh_stack()  # wrong model on purpose; build matching one:
        rebuilt = Sequential(
            [
                _RNNStep(
                    _N_IN, _N_H1,
                    weights["wxh1"], weights["whh1"], weights["b1"],
                ),
                _StatelessScale(),
            ]
        )
        rebuilt.load_incremental(mixed_dir)
        expected_buf = bytearray()
        mixed.save(expected_buf)
        actual_buf = bytearray()
        rebuilt.save(actual_buf)
        _check(
            bytes(actual_buf) == bytes(expected_buf),
            "zero-parameter layers are referenced deterministically and "
            "reassemble bitwise",
        )

        # Missing directory/path is FileNotFoundError, not ValueError.
        _expect(
            FileNotFoundError,
            lambda: rebuilt.load_incremental(os.path.join(td, "no-such-chain")),
            "missing incremental chain directory",
        )

        # Incremental save into an impossible directory is OSError (a file
        # blocks the chain's parent path).
        blocker = os.path.join(td, "blocker")
        with open(blocker, "wb") as fh:
            fh.write(b"x")
        _expect(
            OSError,
            lambda: seq.save_incremental(os.path.join(blocker, "chain")),
            "unwritable chain dir",
        )


def _check_incremental_corruption():
    import json
    import zlib

    with tempfile.TemporaryDirectory() as td:
        chain_dir = os.path.join(td, "chain")
        seq, _ = _fresh_stack()
        seq.save_incremental(chain_dir)
        for segment in (_SEG1, _SEG2, _SEG3):
            out, _ = seq.forward(Tensor(segment))
            seq.backward(_total(out))
            seq.update(0.05)
            seq.save_incremental(chain_dir)
        tip_full = bytearray()
        seq.save(tip_full)

        def expect_chain_rejected(message):
            fresh, _ = _fresh_stack()
            _expect(ValueError, lambda: fresh.load_incremental(chain_dir), message)

        # Truncated middle delta.
        delta2 = os.path.join(chain_dir, "00000002.delta")
        good = open(delta2, "rb").read()
        open(delta2, "wb").write(good[: len(good) // 2])
        expect_chain_rejected("truncated delta")
        open(delta2, "wb").write(good)

        # Flipped byte inside a delta (CRC catches it).
        flipped = bytearray(good)
        flipped[len(flipped) // 2] ^= 0xFF
        open(delta2, "wb").write(bytes(flipped))
        expect_chain_rejected("corrupt delta byte")
        open(delta2, "wb").write(good)

        # Missing middle delta file but manifest ahead of it.
        os.unlink(delta2)
        expect_chain_rejected("missing delta member")
        open(delta2, "wb").write(good)

        # Manifest field removed / oversized integer / garbage JSON.
        manifest_path = os.path.join(chain_dir, "manifest.json")
        good_manifest = open(manifest_path, "rb").read()
        manifest = json.loads(good_manifest.decode("utf-8"))
        del manifest["prev_crc"]
        open(manifest_path, "w").write(json.dumps(manifest))
        expect_chain_rejected("manifest missing field")
        manifest["prev_crc"] = 10**40  # oversized integer
        open(manifest_path, "w").write(json.dumps(manifest))
        expect_chain_rejected("manifest oversized integer")
        open(manifest_path, "wb").write(b"{not json")
        expect_chain_rejected("garbage manifest")
        open(manifest_path, "wb").write(good_manifest)

        # Tampered base.
        base_path = os.path.join(chain_dir, "base.ckp")
        good_base = open(base_path, "rb").read()
        open(base_path, "wb").write(good_base[:-8])
        expect_chain_rejected("truncated base")
        open(base_path, "wb").write(good_base)

        # Chain still intact after restoring everything.
        fresh, _ = _fresh_stack()
        fresh.load_incremental(chain_dir)
        buf = bytearray()
        fresh.save(buf)
        _check(bytes(buf) == bytes(tip_full), "chain heals back to the tip")

        # Crash mid-write: an orphan truncated next delta exists but the
        # manifest still points at the previous complete entry.
        orphan = os.path.join(chain_dir, "00000005.delta")
        open(orphan, "wb").write(b"SEQDELTA1\x00\x00")
        fresh, _ = _fresh_stack()
        fresh.load_incremental(chain_dir)  # ignores the unreferenced orphan
        buf = bytearray()
        fresh.save(buf)
        _check(
            bytes(buf) == bytes(tip_full),
            "an interrupted append leaves the previous complete entry usable",
        )
        # A new append atomically replaces the orphan and continues.
        out, _ = seq.forward(Tensor(_SEG2))
        seq.backward(_total(out))
        next_seq = seq.save_incremental(chain_dir)
        _check(next_seq == 5, "chain continues from the last published entry")

        # Non-finite state is refused at write time and leaves no trace.
        poison, _ = _fresh_stack()
        poison.parameters()[0]._set_values(
            [[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        _expect(
            ValueError,
            lambda: poison.save_incremental(os.path.join(td, "poison")),
            "non-finite parameter refused for incremental save",
        )
        _check(
            not os.path.exists(os.path.join(td, "poison")),
            "a refused incremental save creates no chain directory state",
        )


_GROUPS = [
    ("tensor basics", _check_tensor_basics),
    ("tensor validation", _check_tensor_validation),
    ("container validation", _check_container_validation),
    ("forward and hidden state", _check_forward_and_hidden),
    ("backward contract", _check_backward_contract),
    ("numeric gradients", _check_gradients_numeric),
    ("truncated backpropagation", _check_truncation),
    ("zero_grad and parameters", _check_zero_grad_and_parameters),
    ("one-step update", _check_update_step),
    ("tuple loss and backward retry", _check_tuple_loss_and_retry),
    ("checkpoint bitwise continuity", _check_checkpoint_continuity),
    ("checkpoint round-trip and rejection", _check_checkpoint_roundtrip),
    ("concurrent update/save/load", _check_concurrent_update_save_load),
    ("version 1 migration", _check_version1_migration),
    ("incremental chains", _check_incremental_chains),
    ("incremental corruption and recovery", _check_incremental_corruption),
    ("concurrent checkpoint writers", _check_concurrent_writers),
]


def run_selftest():
    for name, group in _GROUPS:
        try:
            group()
        except _SelfTestFailure as exc:
            print(f"selftest: FAILED [{name}]: {exc}", file=sys.stderr)
            return 1
    print(f"selftest: OK ({len(_GROUPS)} check groups)")
    return 0
