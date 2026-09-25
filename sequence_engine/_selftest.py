"""Built-in self-checks for the engine. Runs in memory; writes no files."""

from __future__ import annotations

import json
import math
import struct
import sys
import threading
import zlib

from . import checkpoint as _checkpoint
from . import _v1_golden
from .sequential import Sequential, _zeros
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


class _CacheWreckingRNN(_RNNStep):
    """RNN step whose first backward wrecks its own cache, then raises."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._boom = True

    def backward(self, upstream):
        if self._boom:
            self._boom = False
            self._cache = None  # a failed pass may destroy what it cached
            raise RuntimeError("transient failure during backward")
        return super().backward(upstream)


def _check_backward_retry_restores_caches():
    # A layer whose backward fails after destroying its cache: the retry
    # must still work, because the engine replays the segment's forward to
    # rebuild every layer cache before handing the error back.
    weights = _base_weights()
    wrecker = _CacheWreckingRNN(
        _N_IN, _N_H1, weights["wxh1"], weights["whh1"], weights["b1"]
    )
    quiet = _RNNStep(_N_H1, _N_H2, weights["wxh2"], weights["whh2"], weights["b2"])
    broken = Sequential([wrecker, quiet])
    out, _ = broken.forward(Tensor(_SEG1))
    _expect(
        RuntimeError,
        lambda: broken.backward((out, 1.0)),
        "first attempt fails",
    )
    # The quiet layer ran before the failure; its partial grad is rolled back.
    _check(quiet.wxh.grad is None, "failed backward rolls partial gradients back")
    # The wrecked cache was rebuilt by replaying the segment's forward.
    _check(wrecker._cache is not None, "a failed backward rebuilds layer caches")
    # The tuple-loss identity still refers to the recorded forward output.
    broken.backward((out, 1.0))
    good, _ = _fresh_stack(weights)
    good_out, _ = good.forward(Tensor(_SEG1))
    good.backward((good_out, 1.0))
    _check(
        [p.grad.tolist() for p in broken.parameters()]
        == [p.grad.tolist() for p in good.parameters()],
        "a retried backward matches the uninterrupted pass bit for bit",
    )
    _expect(
        RuntimeError, lambda: broken.backward(1.0), "the retry consumed the pass"
    )

    # The same contract holds in recompute mode.
    wrecker2 = _CacheWreckingRNN(
        _N_IN, _N_H1, weights["wxh1"], weights["whh1"], weights["b1"]
    )
    quiet2 = _RNNStep(_N_H1, _N_H2, weights["wxh2"], weights["whh2"], weights["b2"])
    tuned = Sequential([wrecker2, quiet2])
    tuned.set_recompute(True)
    tuned.forward(Tensor(_SEG1))
    _expect(RuntimeError, lambda: tuned.backward(1.0), "recompute attempt fails")
    tuned.backward(1.0)
    _check(
        [p.grad.tolist() for p in tuned.parameters()]
        == [p.grad.tolist() for p in good.parameters()],
        "a retried backward in recompute mode matches bit for bit",
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
# Format v2, v1 migration, incremental chains, concurrency.
# ---------------------------------------------------------------------------


def _repack_full(raw, mutate, version=_checkpoint.FORMAT_VERSION):
    hlen = struct.unpack("<Q", raw[12:20])[0]
    header = json.loads(raw[20 : 20 + hlen])
    payload = raw[20 + hlen : raw.rfind(_checkpoint.END_MAGIC)]
    mutate(header)
    new_header = json.dumps(
        header, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    leaf_count = len(payload) // 9
    body = (
        raw[:8]
        + struct.pack("<I", version)
        + struct.pack("<Q", len(new_header))
        + new_header
        + payload
    )
    crc = zlib.crc32(body[20:])
    return body + _checkpoint.END_MAGIC + struct.pack("<QI", leaf_count, crc)


def _downgrade_full_to_v2(raw):
    """Rewrite current-version snapshot bytes as a genuine version-2 file.

    The optimizer state (first/second moments) is the last block of
    payload leaves, so dropping that block plus the header field yields a
    faithful old-format file for the migration checks.
    """
    hlen = struct.unpack("<Q", raw[12:20])[0]
    header = json.loads(raw[20 : 20 + hlen])
    end = raw.rfind(_checkpoint.END_MAGIC)
    payload = raw[20 + hlen : end]
    param_leaves = sum(
        math.prod(shape) for shape in header["params"]
    )
    payload_v2 = payload[: -(2 * param_leaves * 9)]
    header.pop("optim")
    header["v"] = 2
    new_header = json.dumps(
        header, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    leaf_count = len(payload_v2) // 9
    body = (
        raw[:8]
        + struct.pack("<I", 2)
        + struct.pack("<Q", len(new_header))
        + new_header
        + payload_v2
    )
    crc = zlib.crc32(body[20:])
    return body + _checkpoint.END_MAGIC + struct.pack("<QI", leaf_count, crc)


def _check_v3_version_and_v1_v2_migration():
    # Native saves carry format version 3.
    seq, _ = _fresh_stack()
    out, h1 = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    raw = bytearray()
    seq.save(raw)
    _check(
        struct.unpack("<I", bytes(raw)[8:12])[0] == 3,
        "new full checkpoints are format version 3",
    )

    # A genuine version-1 file loads, migrates and records source 1.
    v1 = _v1_golden.trained_bytes()
    _check(struct.unpack("<I", v1[8:12])[0] == 1, "golden fixture really is v1")
    migrated, _ = _fresh_stack()
    restored_hidden = migrated.load(bytes(v1))
    _check(migrated.loaded_from_version == 1, "a v1 load records source version 1")
    # The hidden-state check is aligned to the migrated values themselves:
    # the migrated document's hidden slots are exactly what load returns.
    migrated_doc = _checkpoint.parse_bytes(bytes(v1))
    _check(
        [s.tolist() for s in restored_hidden]
        == [entry["v"] for entry in migrated_doc["hidden"]],
        "v1 migration restores slice-boundary hidden state",
    )
    _check(
        [p.tolist() for p in migrated.parameters()]
        == [p.tolist() for p in seq.parameters()],
        "v1 migration restores parameters exactly",
    )
    _check(migrated._adam_t == 0, "v1 migration starts the optimizer at t=0")
    _check(
        all(value == 0 for tree in migrated._adam_m for value in _flatten(tree))
        and all(value == 0 for tree in migrated._adam_v for value in _flatten(tree)),
        "v1 migration starts the optimizer moments at zero",
    )

    # Float fidelity incl. negative zero through v1 migration.
    v1_tricky = _v1_golden.tricky_bytes()
    tricky, _ = _fresh_stack()
    tricky.load(bytes(v1_tricky))
    values = tricky.parameters()[0].tolist()
    grad_values = tricky.parameters()[0].grad.tolist()
    _check(
        values == [[1.5, -0.0, 3.25], [100000000000000.0, -2.5, 0.0625]],
        "v1 tricky parameter values migrate",
    )
    _check(
        struct.pack("<d", values[0][1]) == struct.pack("<d", -0.0)
        and struct.pack("<d", grad_values[0][0]) == struct.pack("<d", -0.0),
        "v1 migration keeps the negative-zero sign",
    )

    # A genuine version-2 file (no optimizer state) migrates to t=0, zero
    # moments, and re-saves natively as version 3.
    v2 = _downgrade_full_to_v2(bytes(raw))
    _check(struct.unpack("<I", v2[8:12])[0] == 2, "downgraded fixture really is v2")
    from_v2, _ = _fresh_stack()
    from_v2.load(bytes(v2))
    _check(from_v2.loaded_from_version == 2, "a v2 load records source version 2")
    _check(from_v2._adam_t == 0, "v2 migration starts the optimizer at t=0")
    rebuf = bytearray()
    from_v2.save(rebuf)
    _check(
        struct.unpack("<I", bytes(rebuf)[8:12])[0] == 3,
        "a migrated checkpoint re-saves natively as v3",
    )
    again, _ = _fresh_stack()
    again.load(bytes(rebuf))
    _check(again.loaded_from_version == 3, "a native v3 load records source version 3")

    # A migrated v1 state re-saves natively as v3 too.
    rebuf_v1 = bytearray()
    migrated.save(rebuf_v1)
    _check(
        struct.unpack("<I", bytes(rebuf_v1)[8:12])[0] == 3,
        "a migrated v1 checkpoint re-saves natively as v3",
    )

    # Migration failure rejects the whole file and records no source.
    torn = bytes(v1)[: len(v1) // 2]
    victim, _ = _fresh_stack()
    _expect(ValueError, lambda: victim.load(torn), "a torn v1 file is rejected")
    _check(victim.loaded_from_version is None, "rejected migration records no version")

    # A non-finite float smuggled into a v1 payload is refused mid-migration.
    smuggled = bytearray(v1_tricky)
    hlen = struct.unpack("<Q", bytes(smuggled)[12:20])[0]
    end = bytes(smuggled).rfind(_checkpoint.END_MAGIC)
    payload_start = 20 + hlen
    payload = bytes(smuggled)[payload_start:end]
    pos = next(i for i in range(0, len(payload), 9) if payload[i] == ord("f"))
    smuggled[payload_start + pos + 1 : payload_start + pos + 9] = struct.pack(
        "<d", float("inf")
    )
    crc = zlib.crc32(bytes(smuggled)[20:end])
    smuggled[-4:] = struct.pack("<I", crc)
    victim2, _ = _fresh_stack()
    _expect(
        ValueError,
        lambda: victim2.load(bytes(smuggled)),
        "a non-finite float in a v1 payload refuses migration",
    )

    # v3 header with a removed or added field is rejected wholesale.
    victim3, _ = _fresh_stack()
    _expect(
        ValueError,
        lambda: victim3.load(_repack_full(bytes(raw), lambda h: h.pop("layers"))),
        "v3 checkpoint missing a field is rejected",
    )
    _expect(
        ValueError,
        lambda: victim3.load(_repack_full(bytes(raw), lambda h: h.pop("optim"))),
        "v3 checkpoint missing the optimizer state is rejected",
    )
    _expect(
        ValueError,
        lambda: _fresh_stack()[0].load(
            _repack_full(bytes(raw), lambda h: h.update(surprise=1))
        ),
        "v3 checkpoint with an extra field is rejected",
    )


def _check_incremental_chain_in_memory():
    # The whole chain mechanism runs against the in-memory store, so the
    # built-in self-check writes no files.
    seq, _ = _fresh_stack()
    chain = _checkpoint.MemoryChain()
    basis_full = bytearray()
    seq.save(basis_full)
    seq.save(chain)  # basis segment
    _check(len(chain) == 1, "the first chain save writes one basis segment")

    # Unchanged model: an empty, deterministic delta.
    seq.save(chain)
    seg1 = chain.read_segment(_checkpoint._segment_name(1))
    _n, _hc, _p, items1 = _checkpoint._parse_delta(seg1, 1)
    _check(items1 == [], "an unchanged save produces an empty delta")

    # Train: the next delta introduces hidden state with all slots.
    out, hidden = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    seq.save(chain)
    seg2 = chain.read_segment(_checkpoint._segment_name(2))
    _n2, hc2, _p2, items2 = _checkpoint._parse_delta(seg2, 2)
    param_count = len(seq.parameters())
    hidden_base = 4 * param_count + 1
    hidden_indices = {i for i, _s, _t in items2 if i >= hidden_base}
    _check(
        hidden_indices == set(range(hidden_base, hidden_base + 2)),
        "the delta that first fixes hidden state carries every hidden slot",
    )

    # Update: parameters change, hidden does not.
    seq.update(_LR)
    full_after = bytearray()
    seq.save(full_after)
    seq.save(chain)  # update changed params -> delta

    head_doc = _checkpoint.load_chain_memory(chain)
    _check(
        _checkpoint.build_bytes(head_doc) == bytes(full_after),
        "reassembling the chain reproduces the full snapshot bit for bit",
    )

    # Adam steps touch the optimizer tensors too: the chain still
    # reassembles to exactly the full snapshot.
    seq.adam_step(_LR)
    seq.adam_step(_LR)
    full_adam = bytearray()
    seq.save(full_adam)
    seq.save(chain)
    head_doc = _checkpoint.load_chain_memory(chain)
    _check(
        _checkpoint.build_bytes(head_doc) == bytes(full_adam),
        "the chain reproduces a full snapshot with optimizer state bit for bit",
    )

    basis_doc = _checkpoint.load_chain_memory(chain, up_to=0)
    _check(
        _checkpoint.build_bytes(basis_doc) == bytes(basis_full),
        "reassembling up to the basis reproduces the original full snapshot",
    )

    # Loading the chain through Sequential gives the same state as a full load.
    target, _ = _fresh_stack()
    restored = target.load(chain)
    _check(
        [p.tolist() for p in target.parameters()]
        == [p.tolist() for p in seq.parameters()],
        "Sequential.load reassembles chain parameters",
    )
    _check(
        [s.tolist() for s in restored] == [s.tolist() for s in hidden],
        "Sequential.load reassembles chain hidden state",
    )

    # A truncated middle segment rejects the whole chain with ValueError.
    good_seg2 = seg2
    chain.write_segment(_checkpoint._segment_name(2), good_seg2[: len(good_seg2) // 2])
    _expect(
        ValueError,
        lambda: _checkpoint.load_chain_memory(chain),
        "a truncated delta segment rejects the whole chain",
    )
    chain.write_segment(_checkpoint._segment_name(2), good_seg2)
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain))
        == bytes(full_adam),
        "the chain recovers once the segment is whole again",
    )

    # A delta missing a required field is rejected.
    raw = seg2
    hlen = struct.unpack("<Q", raw[12:20])[0]
    dheader = json.loads(raw[20 : 20 + hlen])
    dpayload = raw[20 + hlen : raw.rfind(_checkpoint.DELTA_END_MAGIC)]
    dheader.pop("changed")
    new_header = json.dumps(dheader, separators=(",", ":")).encode("utf-8")
    body = (
        raw[:8]
        + struct.pack("<I", 2)
        + struct.pack("<Q", len(new_header))
        + new_header
        + dpayload
    )
    crc = zlib.crc32(body[20:])
    forged = body + _checkpoint.DELTA_END_MAGIC + struct.pack(
        "<QI", len(dpayload) // 9, crc
    )
    chain.write_segment(_checkpoint._segment_name(2), forged)
    _expect(
        ValueError,
        lambda: _checkpoint.load_chain_memory(chain),
        "a delta missing a field rejects the whole chain",
    )


class _IntLayer:
    """Integer arithmetic layer; backward sets a constant integer gradient."""

    checkpoint_kind = "_IntLayer"

    def __init__(self, weights, grad_const):
        self.w = Tensor(weights)
        self._grad = list(grad_const)
        self._cache = None

    def parameters(self):
        return [self.w]

    def forward(self, x, hidden):
        xs = x.tolist()
        ws = self.w.tolist()
        y = [[xs[r][j] + ws[j] for j in range(len(xs[0]))] for r in range(len(xs))]
        self._cache = xs
        return Tensor(y), Tensor(y)

    def backward(self, upstream):
        dy = (
            upstream.tolist()
            if isinstance(upstream, Tensor)
            else [[upstream] * len(self._grad) for _ in self._cache]
        )
        self.w.grad = Tensor(list(self._grad))
        return Tensor(dy)


def _concurrency_model():
    a = _IntLayer([100000, 200000], [3, 7])
    b = _IntLayer([500000, 900000], [1, 2])
    return Sequential([a, b]), a, b


def _assert_single_step_count(probe, baseline, grads):
    # Each element of a coherent state reads baseline[j] - k * g[j] for one
    # shared non-negative integer k (lr=1, every update adds exactly one
    # step); a half-updated tensor or a load applied halfway would leave
    # different elements at different k. Values stay exact integers because
    # the magnitudes here never lose integral precision in float64.
    steps = None
    for values, w0, g in zip(probe, baseline, grads):
        for j, value in enumerate(values):
            number = int(value)
            delta = w0[j] - number
            _check(delta % g[j] == 0, "concurrent weight left its update lattice")
            k = delta // g[j]
            _check(k >= 0, "an incoherent state produced a negative step count")
            steps = k if steps is None else steps
            _check(k == steps, "half-observed update/load mixed two serial states")


def _check_concurrent_updates_saves_loads():
    # Establish a constant gradient once (one clean segment), then enter the
    # concurrent phase. The only mutating call raced thereafter is
    # update(1): each applied step moves every parameter by exactly its
    # constant gradient g, and an atomic load resets the whole state to the
    # phase baseline. Under any thread interleaving a coherent state is
    # therefore baseline - k*g for one shared k; save/load must never expose
    # anything else.
    seq, a, b = _concurrency_model()
    seq.forward(Tensor([[1, 1]]))
    seq.backward(1)
    baseline = [p.tolist() for p in seq.parameters()]
    grads = [a._grad, b._grad]
    base_bytes = bytearray()
    seq.save(base_bytes)
    errors = []

    def updater():
        try:
            for _ in range(300):
                seq.update(1)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def buffer_roundtrip():
        try:
            for _ in range(300):
                buf = bytearray()
                seq.save(buf)
                probe, _, _ = _concurrency_model()
                probe.load(bytes(buf))
                _assert_single_step_count(
                    [p.tolist() for p in probe.parameters()], baseline, grads
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def chain_roundtrip():
        chain = _checkpoint.MemoryChain()
        try:
            for _ in range(300):
                seq.save(chain)
                probe, _, _ = _concurrency_model()
                probe.load(chain)
                _assert_single_step_count(
                    [p.tolist() for p in probe.parameters()], baseline, grads
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def loader_to_base():
        try:
            for _ in range(300):
                seq.load(bytes(base_bytes))  # atomic reset to the baseline
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = (
        [threading.Thread(target=updater) for _ in range(3)]
        + [
            threading.Thread(target=buffer_roundtrip),
            threading.Thread(target=chain_roundtrip),
            threading.Thread(target=loader_to_base),
        ]
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _check(errors == [], f"concurrent run raised: {errors!r}")
    _assert_single_step_count(
        [p.tolist() for p in seq.parameters()], baseline, grads
    )


def _check_boundary_save_and_eager_hidden_shapes():
    # Save after a forward with no backward is allowed: it fixes the
    # post-forward slice boundary, not the in-flight activations.
    seq, _ = _fresh_stack()
    out, hidden = seq.forward(Tensor(_SEG1))
    buf = bytearray()
    seq.save(buf)  # must not raise
    victim, _ = _fresh_stack()
    restored = victim.load(bytes(buf))
    _check(
        [s.tolist() for s in restored] == [s.tolist() for s in hidden],
        "a post-forward (pre-backward) save fixes boundary hidden state",
    )
    # The saving model's still-pending backward is untouched.
    seq.backward(_total(out))
    _check(
        all(p.grad is not None for p in seq.parameters()),
        "saving does not consume the pending backward",
    )

    # A fresh model (never forwarded) checks hidden shapes on the spot:
    # a checkpoint whose hidden slot declares a wrong shape is refused.
    trained, _ = _fresh_stack()
    o, _h = trained.forward(Tensor(_SEG1))
    trained.backward(_total(o))
    good = bytearray()
    trained.save(good)
    forged = _repack_full(bytes(good), lambda h: h["hidden"].__setitem__(0, [9, 9]))
    brand_new, _ = _fresh_stack()
    _expect(
        ValueError,
        lambda: brand_new.load(forged),
        "a fresh model rejects a checkpoint with a wrong hidden shape",
    )

    # A valid checkpoint loaded into a fresh model pins the hidden shapes:
    # a later forward carrying a wrongly-shaped slot is rejected.
    pinned, _ = _fresh_stack()
    pinned.load(bytes(good))
    bad_hidden = [
        Tensor([[0.0] * 5] * 2),  # layer 1 expects 3 hidden units
        Tensor([[0.0] * 2] * 2),
    ]
    _expect(
        ValueError,
        lambda: pinned.forward(Tensor(_SEG1), bad_hidden),
        "the first load pins hidden shapes used by later forwards",
    )


# ---------------------------------------------------------------------------
# Adam steps, bounded-memory recompute, version-3 optimizer state.
# ---------------------------------------------------------------------------

_ADAM_LR = 0.05


def _reference_adam_step(theta, grads, m, v, t, lr):
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def rec(theta, grads, m, v):
        if isinstance(theta, list):
            return [
                rec(a, b, c, d)
                for a, b, c, d in zip(theta, grads, m, v)
            ]
        m_next = 0.9 * m + 0.1 * grads
        v_next = 0.999 * v + 0.001 * (grads * grads)
        m_hat = m_next / (1.0 - beta1**t)
        v_hat = v_next / (1.0 - beta2**t)
        return theta - lr * m_hat / (math.sqrt(v_hat) + eps)

    def moments(grads, m, v, which):
        if isinstance(grads, list):
            return [
                moments(a, b, c, which) for a, b, c in zip(grads, m, v)
            ]
        if which == "m":
            return 0.9 * m + 0.1 * grads
        return 0.999 * v + 0.001 * (grads * grads)

    return (
        rec(theta, grads, m, v),
        moments(grads, m, v, "m"),
        moments(grads, m, v, "v"),
    )


def _check_adam_step():
    seq, _ = _fresh_stack()
    _check(seq._adam_t == 0, "optimizer starts at t=0")
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    before = [p.tolist() for p in seq.parameters()]
    grads = [p.grad.tolist() for p in seq.parameters()]

    m0 = [_zeros(p.shape) for p in seq.parameters()]
    v0 = m0
    expected_theta, expected_m, expected_v = [], [], []
    for theta, grad, m, v in zip(before, grads, m0, v0):
        nt, nm, nv = _reference_adam_step(theta, grad, m, v, 1, _ADAM_LR)
        expected_theta.append(nt)
        expected_m.append(nm)
        expected_v.append(nv)
    seq.adam_step(_ADAM_LR)
    _check(seq._adam_t == 1, "the first adam_step sets t=1")
    _check(
        [p.tolist() for p in seq.parameters()] == expected_theta,
        "adam_step applies the bias-corrected update exactly",
    )
    _check(seq._adam_m == expected_m and seq._adam_v == expected_v,
           "adam_step keeps the updated first/second moments")
    _check(
        [p.grad.tolist() for p in seq.parameters()] == grads,
        "adam_step does not clear accumulated gradients",
    )

    # A second step uses t=2 and the carried moments.
    expected_theta2, expected_m2, expected_v2 = [], [], []
    for index, param in enumerate(seq.parameters()):
        nt, nm, nv = _reference_adam_step(
            param.tolist(), grads[index], seq._adam_m[index],
            seq._adam_v[index], 2, _ADAM_LR,
        )
        expected_theta2.append(nt)
        expected_m2.append(nm)
        expected_v2.append(nv)
    seq.adam_step(_ADAM_LR)
    _check(seq._adam_t == 2, "the second adam_step sets t=2")
    _check(
        [p.tolist() for p in seq.parameters()] == expected_theta2,
        "the second adam_step continues from the carried moments",
    )
    _check(seq._adam_m == expected_m2 and seq._adam_v == expected_v2,
           "the carried moments match the reference")

    # Validation mirrors update(): non-numeric/non-finite lr is ValueError.
    for bad in (float("nan"), float("inf"), float("-inf")):
        _expect(ValueError, lambda bad=bad: seq.adam_step(bad),
                "non-finite adam learning rate")
    _expect(ValueError, lambda: seq.adam_step("0.05"),
            "non-numeric adam learning rate")
    _check(seq._adam_t == 2, "a rejected adam_step advances no step")

    # update() and adam_step() are independent: a plain update changes
    # neither the moments nor t.
    untouched_m = _copy_nested(seq._adam_m)
    untouched_v = _copy_nested(seq._adam_v)
    seq.update(_LR)
    _check(seq._adam_t == 2, "update() leaves the adam step count alone")
    _check(seq._adam_m == untouched_m and seq._adam_v == untouched_v,
           "update() leaves the adam moments alone")

    # Parameters without a gradient are left as-is, but t still advances.
    quiet, _ = _fresh_stack()
    quiet_t0 = [p.tolist() for p in quiet.parameters()]
    quiet.adam_step(_ADAM_LR)
    _check(
        [p.tolist() for p in quiet.parameters()] == quiet_t0,
        "adam_step leaves parameters without gradients untouched",
    )
    _check(quiet._adam_t == 1, "adam_step still advances t without gradients")


def _copy_nested(value):
    if isinstance(value, list):
        return [_copy_nested(item) for item in value]
    return value


def _adam_run(interrupt_target=None):
    """Three segments, Adam between segments two and three, optional resume."""
    seq, _ = _fresh_stack()
    out1, hidden1 = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out1))
    if interrupt_target is not None:
        seq.save(interrupt_target)
        seq, _ = _fresh_stack()
        hidden1 = seq.load(interrupt_target)
    out2, hidden2 = seq.forward(Tensor(_SEG2), hidden1)
    seq.backward(_total(out2))
    seq.adam_step(_ADAM_LR)
    if interrupt_target is not None:
        seq.save(interrupt_target)
        restored_hidden = [Tensor(slot.tolist()) for slot in hidden2]
        seq, _ = _fresh_stack()
        hidden2 = seq.load(interrupt_target)
        _check(
            [s.tolist() for s in hidden2]
            == [s.tolist() for s in restored_hidden],
            "resumed optimizer state keeps the boundary hidden state",
        )
        _check(seq._adam_t == 1, "resumed optimizer state keeps t=1")
    carry = [Tensor(slot.tolist()) for slot in hidden2]
    out3, _ = seq.forward(Tensor(_SEG3), carry)
    seq.backward(_total(out3))
    seq.adam_step(_ADAM_LR)
    return (
        [p.tolist() for p in seq.parameters()],
        [p.grad.tolist() for p in seq.parameters()],
        seq._adam_t,
    )


def _check_adam_checkpoint_continuity():
    uninterrupted = _adam_run()
    resumed_buffer = _adam_run(bytearray())
    _check(
        resumed_buffer == uninterrupted,
        "save/load resume of adam steps matches the uninterrupted run bit "
        "for bit (parameters, gradients, step count)",
    )
    chain = _checkpoint.MemoryChain()
    resumed_chain = _adam_run(chain)
    _check(
        resumed_chain == uninterrupted,
        "an incremental-chain resume matches the uninterrupted adam run",
    )
    # The reassembled chain equals the full snapshot at the same moment.
    seq, _ = _fresh_stack()
    o, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(o))
    seq.adam_step(_ADAM_LR)
    full = bytearray()
    seq.save(full)
    other_chain = _checkpoint.MemoryChain()
    seq.save(other_chain)
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(other_chain))
        == bytes(full),
        "optimizer state travels the incremental chain bit for bit",
    )


def _check_optim_state_guards():
    # An oversized step count can never enter a checkpoint.
    seq, _ = _fresh_stack()
    seq.zero_grad()
    doc = {
        "params": [{"s": p.shape, "v": p.tolist()} for p in seq.parameters()],
        "grads": [
            {"s": p.shape, "v": p.grad.tolist()} for p in seq.parameters()
        ],
        "hidden": None,
        "optim": {
            "t": 2**63,
            "m": [{"s": p.shape, "v": p.grad.tolist()} for p in seq.parameters()],
            "v": [{"s": p.shape, "v": p.grad.tolist()} for p in seq.parameters()],
        },
        "layers": [
            {"kind": "_RNNStep",
             "shapes": [[2, 3], [3, 3], [3]]},
            {"kind": "_RNNStep",
             "shapes": [[3, 2], [2, 2], [2]]},
        ],
        "pending": False,
    }
    _expect(ValueError, lambda: _checkpoint.build_bytes(doc),
            "oversized optimizer step count rejected on save")

    # A non-finite moment is rejected on save too.
    doc["optim"]["t"] = 1
    bad_m = doc["optim"]["m"][0]["v"]
    bad_m[0][0] = float("inf")
    _expect(ValueError, lambda: _checkpoint.build_bytes(doc),
            "non-finite optimizer moment rejected on save")

    # A v3 payload missing the optimizer state, or with a bad moment shape,
    # is rejected wholesale and leaves the container untouched.
    trained, _ = _fresh_stack()
    o, _ = trained.forward(Tensor(_SEG1))
    trained.backward(_total(o))
    trained.adam_step(_ADAM_LR)
    raw = bytearray()
    trained.save(raw)
    victim, _ = _fresh_stack()
    _expect(
        ValueError,
        lambda: victim.load(
            _repack_full(bytes(raw), lambda h: h.pop("optim"))
        ),
        "a v3 checkpoint missing optimizer state is refused",
    )
    _check(victim._adam_t == 0, "a rejected load leaves the optimizer untouched")

    raw2 = bytearray()
    trained.save(raw2)

    def mangle_moment_shape(header):
        header["optim"]["m"][0] = [9, 9]

    victim2, _ = _fresh_stack()
    _expect(
        ValueError,
        lambda: victim2.load(
            _repack_full(bytes(raw2), mangle_moment_shape)
        ),
        "a checkpoint with a wrong optimizer moment shape is refused",
    )


def _segment_stack(recompute):
    seq, _ = _fresh_stack()
    seq.set_recompute(recompute)
    return seq


def _check_recompute_equivalence():
    _check_three_segment_modes(False, False)
    _check_three_segment_modes(True, True)
    _check_three_segment_modes(False, True)
    _check_three_segment_modes(True, False)


def _check_three_segment_modes(first_mode, second_mode):
    plain = _segment_stack(False)
    tuned = _segment_stack(False)
    tuned.set_recompute(first_mode)
    o1p, h1p = plain.forward(Tensor(_SEG1))
    o1t, h1t = tuned.forward(Tensor(_SEG1))
    _check(o1t.tolist() == o1p.tolist(),
           "recompute forward outputs are bitwise identical")
    plain.backward(_total(o1p))
    tuned.backward(_total(o1t))
    _check(
        [p.grad.tolist() for p in tuned.parameters()]
        == [p.grad.tolist() for p in plain.parameters()],
        "recompute parameter gradients are bitwise identical",
    )
    # Switching between segments is allowed and changes nothing numerically.
    tuned.set_recompute(second_mode)
    o2p, h2p = plain.forward(Tensor(_SEG2), h1p)
    o2t, h2t = tuned.forward(Tensor(_SEG2), h1t)
    _check(o2t.tolist() == o2p.tolist(),
           "recompute stays identical after a between-segment switch")
    plain.backward(_total(o2p))
    tuned.backward(_total(o2t))
    _check(
        [p.grad.tolist() for p in tuned.parameters()]
        == [p.grad.tolist() for p in plain.parameters()],
        "gradients stay identical after a between-segment switch",
    )
    plain.adam_step(_ADAM_LR)
    tuned.adam_step(_ADAM_LR)
    _check(
        [p.tolist() for p in tuned.parameters()]
        == [p.tolist() for p in plain.parameters()],
        "the parameter trajectory under adam is identical in both modes",
    )


def _check_recompute_bounded_memory_and_guards():
    seq, layers = _fresh_stack()
    seq.set_recompute(True)
    out, _ = seq.forward(Tensor(_SEG1))
    # Only anchors are retained: the segment input and the incoming hidden
    # values (here null), independent of how many segments came before.
    batch_values, hidden_values = seq._anchors
    _check(batch_values == _SEG1, "recompute anchors keep the segment input")
    _check(hidden_values == [None, None],
            "recompute anchors keep just the incoming hidden values")
    # Layers are free to drop their own caches entirely: backward recomputes.
    for layer in layers:
        layer._cache = None
    ref, _ = _fresh_stack()
    ref_out, _ = ref.forward(Tensor(_SEG1))
    ref.backward(_total(ref_out))
    seq.backward(_total(out))
    _check(
        [p.grad.tolist() for p in seq.parameters()]
        == [p.grad.tolist() for p in ref.parameters()],
        "backward with evicted layer caches still matches the ordinary run",
    )
    _check(seq._anchors is None, "anchors are released once backward completes")

    # A parameter rewritten mid-segment is rejected before any gradient
    # lands, and leaves no half-applied state.
    victim, _ = _fresh_stack()
    victim.set_recompute(True)
    bad_out, _ = victim.forward(Tensor(_SEG1))
    target = victim.parameters()[0]
    rewritten = target.tolist()
    rewritten[0][1] += 1.0
    target._set_values(rewritten)
    _expect(
        RuntimeError, lambda: victim.backward(_total(bad_out)),
        "a parameter rewritten during a segment raises RuntimeError",
    )
    _check(
        all(p.grad is None for p in victim.parameters()),
        "the rejected recompute leaves no partial gradients",
    )
    # The failed backward consumes nothing: a fresh segment works normally.
    recover_out, _ = victim.forward(Tensor(_SEG1))
    victim.backward(_total(recover_out))
    _check(
        all(p.grad is not None for p in victim.parameters()),
        "the container trains normally after rejecting a stale segment",
    )

    # The switch cannot move in the middle of a pending segment.
    toggled, _ = _fresh_stack()
    toggled.set_recompute(True)
    toggled.forward(Tensor(_SEG1))
    _expect(
        RuntimeError, lambda: toggled.set_recompute(False),
        "turning recompute off mid-segment raises RuntimeError",
    )
    ordinary, _ = _fresh_stack()
    ordinary.forward(Tensor(_SEG1))
    _expect(
        RuntimeError, lambda: ordinary.set_recompute(True),
        "turning recompute on mid-segment raises RuntimeError",
    )

    # A long run's retained state never grows with the total length: after
    # every backward the only kept state is the boundary hidden tensors.
    long_run, _ = _fresh_stack()
    long_run.set_recompute(True)
    hidden = None
    for segment in (_SEG1, _SEG2, _SEG3, _SEG1, _SEG2):
        out, hidden = long_run.forward(Tensor(segment), hidden)
        long_run.backward(_total(out))
        _check(long_run._anchors is None,
                "no anchors survive between segments")
        _check(len(hidden) == 2, "only the boundary hidden state is carried")


def _check_v2_chain_migration():
    # Assemble a genuine version-2 chain (basis + empty delta + the delta
    # that first introduces hidden state) entirely in memory.
    chain, hidden = _make_v2_chain()

    # Reassembling the old chain yields the migrated state: optimizer at
    # t=0 with zero moments, hidden slots exactly as written.
    doc = _checkpoint.load_chain_memory(chain)
    _check(doc["optim"]["t"] == 0, "a v2 chain migrates to t=0")
    _check(
        all(value == 0.0 for entry in doc["optim"]["m"]
            for value in _flatten(entry["v"])),
        "a v2 chain migrates to zero first moments",
    )
    _check(
        [entry["v"] for entry in doc["hidden"]]
        == [slot.tolist() for slot in hidden],
        "a v2 chain reassembles its hidden slots correctly",
    )

    victim, _ = _fresh_stack()
    restored = victim.load(chain)
    _check(victim.loaded_from_version == 3,
           "a loaded chain always reports the current document version")
    _check(
        [s.tolist() for s in restored] == [s.tolist() for s in hidden],
        "Sequential.load reassembles a v2 chain's hidden state",
    )
    _check(victim._adam_t == 0, "a v2 chain starts the optimizer at t=0")

    # Re-saving the migrated chain appends native v3 deltas; the head must
    # still reassemble to the full snapshot exactly.
    victim.adam_step(_ADAM_LR)
    full = bytearray()
    victim.save(full)
    victim.save(chain)
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain))
        == bytes(full),
        "a migrated v2 chain continues with v3 deltas bit for bit",
    )


def _make_v2_chain():
    """A genuine version-2 chain (basis + empty delta + hidden-introducing
    delta), assembled in memory; returns ``(chain, hidden)``."""
    seq, _ = _fresh_stack()
    basis_raw = bytearray()
    seq.save(basis_raw)
    basis_v2 = _downgrade_full_to_v2(bytes(basis_raw))
    param_count = len(seq.parameters())

    def v2_delta_frame(number, hc, changed_entries):
        payload = []
        header_entries = []
        for index, shape, tree in changed_entries:
            _checkpoint._freeze_tree(tree, shape, payload)
            header_entries.append({"i": index, "s": shape})
        header = {
            "v": 2,
            "b": _checkpoint._segment_name(0),
            "n": number,
            "hc": hc,
            "changed": header_entries,
            "pending": False,
        }
        return _checkpoint._frame(
            _checkpoint.DELTA_MAGIC, _checkpoint.DELTA_END_MAGIC, header, payload
        )

    chain = _checkpoint.MemoryChain()
    chain.write_segment(_checkpoint._segment_name(0), basis_v2)
    chain.write_segment(_checkpoint._segment_name(1), v2_delta_frame(1, None, []))
    out, hidden = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    hidden_entries = [
        (2 * param_count + slot, slot_tensor.shape, slot_tensor.tolist())
        for slot, slot_tensor in enumerate(hidden)
    ]
    chain.write_segment(
        _checkpoint._segment_name(2), v2_delta_frame(2, 2, hidden_entries)
    )
    chain.write_head(b"2")
    return chain, hidden


def _check_chain_compaction_memory():
    # Full compaction: basis + deltas fold into one native-v3 basis.
    seq, _ = _fresh_stack()
    chain = _checkpoint.MemoryChain()
    seq.save(chain)  # basis: no hidden, t=0
    out, hidden = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    seq.save(chain)  # delta 1: introduces hidden state
    seq.update(_LR)
    seq.save(chain)  # delta 2
    seq.adam_step(_ADAM_LR)
    seq.save(chain)  # delta 3: optimizer state moves
    full_head = bytearray()
    seq.save(full_head)
    _check(len(chain) == 4, "the chain grows one segment per save")

    _checkpoint.compact_chain_memory(chain)
    _check(len(chain) == 1, "full compaction leaves a single basis segment")
    _check(chain.read_head() == b"0", "the compacted head points at the basis")
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain))
        == bytes(full_head),
        "compaction preserves the reassembled state bit for bit",
    )
    basis_raw = chain.read_segment(_checkpoint._segment_name(0))
    _check(
        struct.unpack("<I", basis_raw[8:12])[0] == _checkpoint.FORMAT_VERSION,
        "the compacted basis is written in the current format version",
    )
    _check(
        _checkpoint.load_chain_memory(chain)["optim"]["t"] == 1,
        "compaction preserves the optimizer step count",
    )
    # Repeating the same compaction is a deterministic no-op.
    _checkpoint.compact_chain_memory(chain)
    _check(len(chain) == 1, "repeated compaction changes nothing")
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain))
        == bytes(full_head),
        "repeated compaction keeps the exact state",
    )

    # Partial compaction: fold basis..2, renumber the remaining deltas.
    seq2, _ = _fresh_stack()
    chain2 = _checkpoint.MemoryChain()
    seq2.save(chain2)  # seg 0 (basis, no hidden)
    out2, hidden2 = seq2.forward(Tensor(_SEG1))
    seq2.backward(_total(out2))
    seq2.save(chain2)  # seg 1 (hidden introduced)
    seq2.update(_LR)
    seq2.save(chain2)  # seg 2
    seq2.adam_step(_ADAM_LR)
    seq2.save(chain2)  # seg 3
    seq2.update(_LR)
    seq2.save(chain2)  # seg 4
    full2_head = bytearray()
    seq2.save(full2_head)
    folded_at2 = _checkpoint.build_bytes(
        _checkpoint.load_chain_memory(chain2, up_to=2)
    )
    _checkpoint.compact_chain_memory(chain2, up_to=2)
    _check(len(chain2) == 3, "partial compaction folds exactly the merged range")
    _check(chain2.read_head() == b"2", "partial compaction renumbers the head")
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain2))
        == bytes(full2_head),
        "partial compaction preserves the head state bit for bit",
    )
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain2, up_to=0))
        == folded_at2,
        "the folded basis reassembles to the state at the fold point",
    )
    _check(
        _checkpoint.load_chain_memory(chain2)["optim"]["t"] == 1,
        "partial compaction preserves the optimizer step count",
    )

    # Appending after compaction keeps matching full snapshots bit for bit.
    out3, _ = seq2.forward(Tensor(_SEG2), hidden2)
    seq2.backward(_total(out3))
    seq2.adam_step(_ADAM_LR)
    full2_final = bytearray()
    seq2.save(full2_final)
    seq2.save(chain2)
    _check(len(chain2) == 4, "appending after compaction adds one delta")
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain2))
        == bytes(full2_final),
        "the compacted chain keeps matching full snapshots bit for bit",
    )

    # A chain that never stepped keeps t=0 and zero moments through compaction.
    unstepped, _ = _fresh_stack()
    chain3 = _checkpoint.MemoryChain()
    unstepped.save(chain3)
    unstepped.update(_LR)
    unstepped.save(chain3)
    _checkpoint.compact_chain_memory(chain3)
    doc3 = _checkpoint.load_chain_memory(chain3)
    _check(
        doc3["optim"]["t"] == 0,
        "an unstepped chain stays at t=0 through compaction",
    )
    _check(
        all(value == 0 for entry in doc3["optim"]["m"] for value in _flatten(entry["v"])),
        "an unstepped chain keeps zero moments through compaction",
    )

    # Old-version segments participate and are rewritten as the current version.
    v2_chain, v2_hidden = _make_v2_chain()
    before = _checkpoint.build_bytes(_checkpoint.load_chain_memory(v2_chain))
    _checkpoint.compact_chain_memory(v2_chain)
    _check(len(v2_chain) == 1, "a v2 chain compacts to one segment")
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(v2_chain)) == before,
        "compacting a v2 chain preserves its state bit for bit",
    )
    _check(
        struct.unpack("<I", v2_chain.read_segment(_checkpoint._segment_name(0))[8:12])[0]
        == _checkpoint.FORMAT_VERSION,
        "the compacted v2 chain is rewritten in the current format version",
    )
    doc_v2 = _checkpoint.load_chain_memory(v2_chain)
    _check(doc_v2["optim"]["t"] == 0, "a compacted v2 chain stays at t=0")
    _check(
        [entry["v"] for entry in doc_v2["hidden"]]
        == [slot.tolist() for slot in v2_hidden],
        "a compacted v2 chain keeps its hidden state",
    )

    # A corrupt chain is rejected wholesale and left untouched.
    corrupt = _checkpoint.MemoryChain()
    broken_seq, _ = _fresh_stack()
    broken_seq.save(corrupt)
    broken_seq.update(_LR)
    broken_seq.save(corrupt)
    seg1 = corrupt.read_segment(_checkpoint._segment_name(1))
    corrupt.write_segment(
        _checkpoint._segment_name(1), seg1[: len(seg1) // 2]
    )
    head_before = corrupt.read_head()
    _expect(
        ValueError,
        lambda: _checkpoint.compact_chain_memory(corrupt),
        "a corrupt chain rejects compaction",
    )
    _check(
        corrupt.read_head() == head_before,
        "a rejected compaction leaves the chain untouched",
    )

    # Deterministic edge cases: empty chain, basis-only chain, bad ranges.
    _expect(
        ValueError,
        lambda: _checkpoint.compact_chain_memory(_checkpoint.MemoryChain()),
        "an empty chain has nothing to compact",
    )
    basis_only = _checkpoint.MemoryChain()
    solo, _ = _fresh_stack()
    solo.save(basis_only)
    _checkpoint.compact_chain_memory(basis_only)  # nothing to merge: no-op
    _check(len(basis_only) == 1, "a basis-only chain is left untouched")
    _checkpoint.compact_chain_memory(chain2, up_to=0)  # folds nothing: no-op
    _check(len(chain2) == 4, "up_to=0 is a deterministic no-op")
    for bad in (-1, True, 0.5, "1"):
        _expect(
            ValueError,
            lambda bad=bad: _checkpoint.compact_chain_memory(chain2, bad),
            f"invalid up_to {bad!r} is rejected",
        )
    _expect(
        ValueError,
        lambda: _checkpoint.compact_chain_memory(chain2, 99),
        "up_to beyond the head is rejected",
    )

    # The container-level entry point compacts memory chains too.
    via_seq, _ = _fresh_stack()
    chain4 = _checkpoint.MemoryChain()
    via_seq.save(chain4)
    via_seq.update(_LR)
    via_seq.save(chain4)
    via_seq.compact(chain4)
    _check(len(chain4) == 1, "Sequential.compact folds a memory chain")
    _expect(
        TypeError,
        lambda: via_seq.compact(bytearray()),
        "compact rejects a non-chain target",
    )


def _check_chain_fork_memory():
    # A forked branch shares the prefix with its source chain and then
    # evolves independently; both chains stay bit for bit identical to
    # two chains that never forked.
    seq, _ = _fresh_stack()
    chain = _checkpoint.MemoryChain()
    seq.save(chain)  # seg 0 (basis, no hidden)
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    seq.update(_LR)
    seq.save(chain)  # seg 1 (hidden introduced)
    seq.adam_step(_ADAM_LR)
    seq.save(chain)  # seg 2 (optimizer state moves)

    branch = _checkpoint.fork_chain_memory(chain, up_to=1)
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(branch))
        == _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain, up_to=1)),
        "a fork reassembles exactly the fork-point state",
    )
    _check(branch.read_head() == b"1", "the branch head names the fork point")

    # Both chains append to the same next position without interfering.
    main_seq, _ = _fresh_stack()
    main_seq.load(chain)
    main_seq.update(_LR)
    main_seq.save(chain)  # main seg 3
    branch_seq, _ = _fresh_stack()
    branch_seq.load(branch)
    branch_seq.adam_step(_ADAM_LR)
    branch_seq.save(branch)  # branch seg 2
    _check(len(chain) == 4 and len(branch) == 3, "each chain grows its own tail")

    # Reference chains replayed without a fork must match bit for bit.
    ref_main, _ = _fresh_stack()
    ref_main_chain = _checkpoint.MemoryChain()
    ref_main.save(ref_main_chain)
    out, _ = ref_main.forward(Tensor(_SEG1))
    ref_main.backward(_total(out))
    ref_main.update(_LR)
    ref_main.save(ref_main_chain)
    ref_main.adam_step(_ADAM_LR)
    ref_main.save(ref_main_chain)
    ref_main.update(_LR)
    ref_main.save(ref_main_chain)
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain))
        == _checkpoint.build_bytes(_checkpoint.load_chain_memory(ref_main_chain)),
        "the source chain evolves as if it had never forked",
    )
    ref_branch, _ = _fresh_stack()
    ref_branch_chain = _checkpoint.MemoryChain()
    ref_branch.save(ref_branch_chain)
    out, _ = ref_branch.forward(Tensor(_SEG1))
    ref_branch.backward(_total(out))
    ref_branch.update(_LR)
    ref_branch.save(ref_branch_chain)
    ref_branch.adam_step(_ADAM_LR)
    ref_branch.save(ref_branch_chain)
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(branch))
        == _checkpoint.build_bytes(_checkpoint.load_chain_memory(ref_branch_chain)),
        "the branch evolves as if it had never shared a prefix",
    )

    # Compacting one chain leaves the other's state untouched.
    _checkpoint.compact_chain_memory(chain)
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(branch))
        == _checkpoint.build_bytes(_checkpoint.load_chain_memory(ref_branch_chain)),
        "compacting the source chain leaves the branch bit for bit intact",
    )
    _checkpoint.compact_chain_memory(branch)
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain))
        == _checkpoint.build_bytes(_checkpoint.load_chain_memory(ref_main_chain)),
        "compacting the branch leaves the source chain bit for bit intact",
    )

    # The container-level entry point forks memory chains too.
    via_seq, _ = _fresh_stack()
    chain2 = _checkpoint.MemoryChain()
    via_seq.save(chain2)
    via_seq.update(_LR)
    via_seq.save(chain2)
    branch2 = via_seq.fork(chain2)
    _check(
        isinstance(branch2, _checkpoint.MemoryChain)
        and branch2.read_head() == b"1",
        "Sequential.fork derives a memory branch at the head by default",
    )
    _expect(
        TypeError,
        lambda: via_seq.fork(chain2, "somewhere"),
        "a memory fork takes no target directory",
    )
    _expect(
        TypeError,
        lambda: via_seq.fork(bytearray()),
        "fork rejects a non-chain source",
    )

    # Bad fork points and an empty chain are refused with ValueError.
    for bad in (-1, True, 0.5, "1"):
        _expect(
            ValueError,
            lambda bad=bad: _checkpoint.fork_chain_memory(chain2, bad),
            f"invalid fork point {bad!r} is rejected",
        )
    _expect(
        ValueError,
        lambda: _checkpoint.fork_chain_memory(chain2, 99),
        "a fork point beyond the head is rejected",
    )
    _expect(
        ValueError,
        lambda: _checkpoint.fork_chain_memory(_checkpoint.MemoryChain()),
        "forking a chain with no committed basis is rejected",
    )

    # A family of memory chains verifies through the single-chain entry.
    reports = [
        _checkpoint.verify_chain_memory(chain),
        _checkpoint.verify_chain_memory(branch),
    ]
    _check(all(report.ok for report in reports), "the family members verify")


def _check_chain_delete_memory():
    # Deleting a branch drops exactly its own references; the source
    # chain stays bit for bit intact and advances no optimizer step.
    seq, _ = _fresh_stack()
    chain = _checkpoint.MemoryChain()
    seq.save(chain)  # seg 0 (basis, no hidden)
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    seq.update(_LR)
    seq.save(chain)  # seg 1 (hidden introduced)
    seq.adam_step(_ADAM_LR)
    seq.save(chain)  # seg 2 (optimizer state moves)

    branch = _checkpoint.fork_chain_memory(chain, up_to=1)
    branch_seq, _ = _fresh_stack()
    branch_seq.load(branch)
    branch_seq.update(_LR)
    branch_seq.save(branch)  # branch seg 2 of its own
    source_state = _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain))
    source_t = _checkpoint.load_chain_memory(chain)["optim"]["t"]

    _checkpoint.delete_chain_memory(branch)
    _check(len(branch) == 0, "a deleted chain holds no segments")
    _expect(
        ValueError,
        lambda: _checkpoint.load_chain_memory(branch),
        "a deleted chain has no head pointer",
    )
    _expect(
        ValueError,
        lambda: _checkpoint.verify_chain_memory(branch),
        "a deleted chain fails verification",
    )
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain))
        == source_state,
        "deleting the branch leaves the source chain bit for bit intact",
    )
    _check(
        _checkpoint.load_chain_memory(chain)["optim"]["t"] == source_t,
        "deletion advances no chain's optimizer step count",
    )

    # The emptied store can be reused: its next save writes a fresh basis.
    reused, _ = _fresh_stack()
    reused.save(branch)
    _check(
        len(branch) == 1 and branch.read_head() == b"0",
        "a deleted memory chain accepts a fresh basis",
    )

    # Deleting the source chain leaves the branch one complete state.
    branch2 = _checkpoint.fork_chain_memory(chain, up_to=2)
    branch2_state = _checkpoint.build_bytes(
        _checkpoint.load_chain_memory(branch2)
    )
    _checkpoint.delete_chain_memory(chain)
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(branch2))
        == branch2_state,
        "deleting the source leaves the branch bit for bit intact",
    )
    _check(
        _checkpoint.verify_chain_memory(branch2).ok,
        "the surviving branch still verifies",
    )

    # The container-level entry point and the error taxonomy.
    via_seq, _ = _fresh_stack()
    chain2 = _checkpoint.MemoryChain()
    via_seq.save(chain2)
    via_seq.delete(chain2)
    _check(len(chain2) == 0, "Sequential.delete empties a memory chain")
    _expect(
        ValueError,
        lambda: _checkpoint.delete_chain_memory(chain2),
        "deleting a chain with no committed basis is rejected",
    )
    _expect(
        ValueError,
        lambda: via_seq.delete(chain2),
        "deleting an already-deleted memory chain is rejected",
    )
    _expect(
        TypeError,
        lambda: _checkpoint.delete_chain_memory(object()),
        "delete_chain_memory rejects a non-MemoryChain",
    )
    _expect(
        TypeError,
        lambda: via_seq.delete(bytearray()),
        "delete rejects a non-chain target",
    )
    _expect(
        TypeError,
        lambda: _checkpoint.delete_chain(123),
        "delete_chain rejects a non-path",
    )


def _check_chain_merge_memory():
    # Merging lands the source chain's current state onto the target as
    # one appended target-owned delta; the source is untouched and the
    # two chains keep evolving independently.
    seq, _ = _fresh_stack()
    chain = _checkpoint.MemoryChain()
    seq.save(chain)  # seg 0 (basis, no hidden)
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    seq.update(_LR)
    seq.save(chain)  # seg 1 (hidden introduced)
    seq.adam_step(_ADAM_LR)
    seq.save(chain)  # seg 2 (optimizer state moves)
    source_state = _checkpoint.build_bytes(
        _checkpoint.load_chain_memory(chain)
    )
    source_head = chain.read_head()

    # The target starts from a fork of an earlier point: a family member
    # sharing a prefix with the source but holding a different head.
    branch = _checkpoint.fork_chain_memory(chain, up_to=1)
    branch_seq, _ = _fresh_stack()
    bhidden = branch_seq.load(branch)
    bout, bhidden = branch_seq.forward(Tensor(_SEG2), bhidden)
    branch_seq.backward(_total(bout))
    branch_seq.update(_LR)
    branch_seq.save(branch)  # branch seg 2 of its own, a different state
    branch_before = _checkpoint.build_bytes(_checkpoint.load_chain_memory(branch))
    _check(branch_before != source_state, "the merge target starts in a different state")

    _checkpoint.merge_chain_memory(chain, branch)
    _check(branch.read_head() == b"3", "the merge appends exactly one segment")
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(branch))
        == source_state,
        "after the merge the target reassembles to the source state bit for bit",
    )
    _check(
        chain.read_head() == source_head,
        "the merge leaves the source head exactly as it was",
    )
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain))
        == source_state,
        "the merge leaves the source state untouched",
    )
    # Every target segment it already had is kept; the appended one is
    # the target's own (a fresh bytes object, never the source's).
    _check(len(branch) == 4, "the target keeps all its segments plus one")
    _check(
        branch._objects[_checkpoint._segment_name(3)]
        is not chain._objects[_checkpoint._segment_name(2)],
        "the appended segment belongs to the target alone",
    )

    # Merging the same state again appends an empty delta and changes no
    # state; the appended empty delta parses with no changed tensors.
    _checkpoint.merge_chain_memory(chain, branch)
    _check(branch.read_head() == b"4", "a repeated merge appends one segment")
    raw = branch.read_segment(_checkpoint._segment_name(4))
    _n, _hc, _p, items = _checkpoint._parse_delta(raw, 4)
    _check(items == [], "a repeated merge of the same state appends an empty delta")
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(branch))
        == source_state,
        "the empty merge delta leaves the state exactly as it was",
    )

    # A full save onto the merged chain lands bytes consistent with the
    # merge result and advances no optimizer step.
    merged_seq, _ = _fresh_stack()
    restored_hidden = merged_seq.load(branch)
    t_before = merged_seq._adam_t
    full = bytearray()
    merged_seq.save(full)
    _check(
        bytes(full) == source_state,
        "a full save after the merge matches the merge result bit for bit",
    )
    _check(
        merged_seq._adam_t == t_before,
        "a save after the merge advances no optimizer step",
    )
    _check(
        restored_hidden is not None and len(restored_hidden) == 2,
        "loading the merged target restores the boundary hidden state",
    )

    # After the merge both chains evolve independently again.
    merged_seq.update(_LR)
    merged_seq.save(branch)
    seq.update(_ADAM_LR)  # different operation on the source chain
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(branch))
        != _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain)),
        "the two chains diverge independently after the merge",
    )

    # Independent (unforked) chains of the same model merge too: merging
    # an identical state appends an empty delta and changes nothing else.
    a_chain = _checkpoint.MemoryChain()
    b_chain = _checkpoint.MemoryChain()
    same, _ = _fresh_stack()
    same.save(a_chain)
    same.save(b_chain)
    _checkpoint.merge_chain_memory(a_chain, b_chain)
    _check(b_chain.read_head() == b"1", "an identical-state merge appends one segment")
    empty_raw = b_chain.read_segment(_checkpoint._segment_name(1))
    _n2, _hc2, _p2, empty_items = _checkpoint._parse_delta(empty_raw, 1)
    _check(empty_items == [], "an identical-state merge appends an empty delta")

    # Error taxonomy.
    _expect(
        ValueError,
        lambda: _checkpoint.merge_chain_memory(chain, chain),
        "a chain cannot be merged into itself",
    )
    _expect(
        ValueError,
        lambda: _checkpoint.merge_chain_memory(_checkpoint.MemoryChain(), chain),
        "an empty source chain is rejected",
    )
    _expect(
        ValueError,
        lambda: _checkpoint.merge_chain_memory(chain, _checkpoint.MemoryChain()),
        "an empty target chain is rejected",
    )
    other_weights = _base_weights()
    other_weights["b1"] = [0.01, -0.02]  # [2] instead of [3]
    different, _ = _fresh_stack(other_weights)
    other_chain = _checkpoint.MemoryChain()
    different.save(other_chain)
    fresh_chain = _checkpoint.MemoryChain()
    fresh, _ = _fresh_stack()
    fresh.save(fresh_chain)
    _expect(
        ValueError,
        lambda: _checkpoint.merge_chain_memory(other_chain, fresh_chain),
        "chains whose shapes disagree reject the whole merge",
    )
    _expect(
        TypeError,
        lambda: _checkpoint.merge_chain_memory(chain, object()),
        "merge_chain_memory rejects a non-MemoryChain target",
    )
    _expect(
        TypeError,
        lambda: _checkpoint.merge_chain_memory(object(), chain),
        "merge_chain_memory rejects a non-MemoryChain source",
    )
    via_seq, _ = _fresh_stack()
    _expect(
        TypeError,
        lambda: via_seq.merge(chain, "some-directory"),
        "mixed directory/memory merge arguments are rejected",
    )
    _expect(
        TypeError,
        lambda: via_seq.merge(b_chain, bytearray()),
        "merge rejects non-chain arguments",
    )

    # Both merged families still verify.
    _check(_checkpoint.verify_chain_memory(chain).ok, "the source still verifies")
    _check(_checkpoint.verify_chain_memory(branch).ok, "the merged target still verifies")


def _check_streaming_compaction_interleaves():
    # The streaming fold keeps working through concurrent appends and
    # reads: a compactor, an appender and readers run at once against one
    # in-memory chain and every observed state is a complete, loadable
    # chain.  The final chain compacts to one basis matching its head.
    seq, _ = _fresh_stack()
    chain = _checkpoint.MemoryChain()
    seq.save(chain)
    errors = []
    stop = False

    def appender():
        try:
            for _ in range(120):
                doc = _checkpoint.load_chain_memory(chain)
                _checkpoint.save_chain_memory(doc, chain)  # empty delta
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            while not stop:
                _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=appender)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for thread in threads:
        thread.start()
    for _ in range(30):
        _checkpoint.compact_chain_memory(chain)
    stop = True
    for thread in threads:
        thread.join()
    _check(errors == [], f"streaming interleave raised: {errors!r}")
    head_bytes = _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain))
    _checkpoint.compact_chain_memory(chain)
    _check(len(chain) == 1, "the interleaved chain finally folds to one basis")
    _check(
        _checkpoint.build_bytes(_checkpoint.load_chain_memory(chain)) == head_bytes,
        "the folded chain preserves the head state bit for bit",
    )


def _check_chain_verification():
    # A sound chain verifies and reports the head/segment count.
    seq, _ = _fresh_stack()
    chain = _checkpoint.MemoryChain()
    seq.save(chain)
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    seq.save(chain)
    seq.update(_LR)
    seq.save(chain)
    report = _checkpoint.verify_chain_memory(chain)
    _check(report.ok and report.head == 2 and report.segments == 3,
           "a sound chain verifies with a success report")
    probe, _ = _fresh_stack()
    _check(probe.verify(chain).ok, "Sequential.verify reads a sound chain")

    # A truncated delta is rejected at that segment, naming position/reason.
    seg1 = chain.read_segment(_checkpoint._segment_name(1))
    chain.write_segment(_checkpoint._segment_name(1), seg1[: len(seg1) // 2])
    try:
        _checkpoint.verify_chain_memory(chain)
    except ValueError as exc:
        message = str(exc)
        _check("segment 1" in message, f"verify names the first bad segment: {message}")
    else:
        raise _SelfTestFailure("a truncated segment must fail verification")
    chain.write_segment(_checkpoint._segment_name(1), seg1)
    _check(_checkpoint.verify_chain_memory(chain).ok,
           "the chain verifies once the segment is whole")

    # A truncated basis is located at segment 0.
    seg0 = chain.read_segment(_checkpoint._segment_name(0))
    chain.write_segment(_checkpoint._segment_name(0), seg0[:20])
    try:
        _checkpoint.verify_chain_memory(chain)
    except ValueError as exc:
        _check("segment 0" in str(exc), "a bad basis is located at segment 0")
    else:
        raise _SelfTestFailure("a truncated basis must fail verification")
    chain.write_segment(_checkpoint._segment_name(0), seg0)

    # A corrupt head pointer is rejected (not mistaken for an empty chain).
    chain.write_head(b"not-a-number")
    _expect(ValueError, lambda: _checkpoint.verify_chain_memory(chain),
            "a corrupt head pointer fails verification")
    chain.write_head(b"2")

    # An empty chain (no head) is rejected wholesale.
    _expect(
        ValueError,
        lambda: _checkpoint.verify_chain_memory(_checkpoint.MemoryChain()),
        "an empty chain fails verification",
    )
    _expect(TypeError, lambda: _checkpoint.verify_chain_memory(object()),
            "verify rejects a non-MemoryChain")


class _BoomBackwardAndReplayRNN(_RNNStep):
    """Layer whose first backward raises and whose cache-rebuild replay
    also raises, so the rebuild failure must be surfaced, not swallowed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._boom = True
        self._fail_replay = False

    def forward(self, x, hidden):
        if self._fail_replay:
            raise ValueError("replay forward failed")
        return super().forward(x, hidden)

    def backward(self, upstream):
        if self._boom:
            self._boom = False
            self._cache = None
            raise RuntimeError("layer backward failed")
        return super().backward(upstream)


def _check_backward_replay_failure_is_surfaced():
    weights = _base_weights()
    broken_layer = _BoomBackwardAndReplayRNN(
        _N_IN, _N_H1, weights["wxh1"], weights["whh1"], weights["b1"]
    )
    quiet = _RNNStep(_N_H1, _N_H2, weights["wxh2"], weights["whh2"], weights["b2"])
    seq = Sequential([broken_layer, quiet])
    seq.forward(Tensor(_SEG1))
    broken_layer._fail_replay = True
    try:
        seq.backward(1.0)
    except RuntimeError as exc:
        # The original layer exception is still what reaches the caller...
        _check("layer backward failed" in str(exc),
               "the original layer error is passed through")
        cause = exc.__cause__
        _check(isinstance(cause, ValueError),
               "a failed rebuild surfaces as a ValueError")
        message = str(cause).lower()
        _check("rebuild" in message and "replay" in message,
               f"the ValueError names the rebuild stage: {message}")
        _check("replay forward failed" in str(cause),
               "the ValueError carries the replay failure")
    else:
        raise _SelfTestFailure("backward must still raise the layer error")
    # Partial accumulation was rolled back even though the replay failed.
    _check(quiet.wxh.grad is None, "partial gradients are rolled back")
    # The failed rebuild surfaced (rather than being swallowed) and the
    # original error reached the caller.  With the layer healthy again a
    # fresh forward/backward trains normally; the bitwise retry of the
    # interrupted pass is covered separately for the rebuild-succeeds
    # case, since a failed rebuild leaves no caches to retry against.
    broken_layer._fail_replay = False
    fresh_out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(fresh_out))
    good, _ = _fresh_stack(weights)
    good_out, _ = good.forward(Tensor(_SEG1))
    good.backward(_total(good_out))
    _check(
        [p.grad.tolist() for p in seq.parameters()]
        == [p.grad.tolist() for p in good.parameters()],
        "the container trains normally after a surfaced rebuild failure",
    )


def _check_concurrent_adam_saves_loads():
    seq, _ = _fresh_stack()
    seq.forward(Tensor(_SEG1))
    seq.backward(1.0)
    base = bytearray()
    seq.save(base)
    errors = []

    def stepper():
        try:
            for _ in range(200):
                seq.adam_step(_ADAM_LR)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def probe_state(model):
        _check(model._adam_t >= 0, "a loaded checkpoint carries a valid step count")
        for param, m, v in zip(
            model.parameters(), model._adam_m, model._adam_v
        ):
            for tree in (param.tolist(), m, v, param.grad.tolist()):
                for value in _flatten(tree):
                    _check(
                        isinstance(value, (int, float))
                        and value == value
                        and value not in (float("inf"), float("-inf")),
                        "a concurrently saved checkpoint is fully finite",
                    )

    def buffer_roundtrip():
        try:
            for _ in range(200):
                buf = bytearray()
                seq.save(buf)
                probe, _ = _fresh_stack()
                probe.load(bytes(buf))
                probe_state(probe)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def chain_roundtrip():
        chain = _checkpoint.MemoryChain()
        try:
            for _ in range(200):
                seq.save(chain)
                probe, _ = _fresh_stack()
                probe.load(chain)
                probe_state(probe)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def loader_to_base():
        try:
            for _ in range(200):
                seq.load(bytes(base))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = (
        [threading.Thread(target=stepper) for _ in range(3)]
        + [
            threading.Thread(target=buffer_roundtrip),
            threading.Thread(target=chain_roundtrip),
            threading.Thread(target=loader_to_base),
        ]
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _check(errors == [], f"concurrent adam run raised: {errors!r}")


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
    ("adam step", _check_adam_step),
    ("tuple loss and backward retry", _check_tuple_loss_and_retry),
    ("checkpoint bitwise continuity", _check_checkpoint_continuity),
    ("checkpoint round-trip and rejection", _check_checkpoint_roundtrip),
    ("v3 format and v1/v2 migration", _check_v3_version_and_v1_v2_migration),
    ("incremental checkpoint chain", _check_incremental_chain_in_memory),
    ("v2 incremental chain migration", _check_v2_chain_migration),
    ("optimizer-state checkpoint guards", _check_optim_state_guards),
    ("adam checkpoint continuity", _check_adam_checkpoint_continuity),
    ("recompute mode equivalence", _check_recompute_equivalence),
    ("recompute bounded memory and guards", _check_recompute_bounded_memory_and_guards),
    ("backward retry restores layer caches", _check_backward_retry_restores_caches),
    ("in-memory chain compaction", _check_chain_compaction_memory),
    ("in-memory chain fork", _check_chain_fork_memory),
    ("in-memory chain delete", _check_chain_delete_memory),
    ("in-memory chain merge", _check_chain_merge_memory),
    ("streaming compaction interleave", _check_streaming_compaction_interleaves),
    ("chain verification", _check_chain_verification),
    ("backward replay failure surfaced", _check_backward_replay_failure_is_surfaced),
    ("concurrent update/save/load", _check_concurrent_updates_saves_loads),
    ("concurrent adam/save/load", _check_concurrent_adam_saves_loads),
    ("boundary save and eager hidden shapes", _check_boundary_save_and_eager_hidden_shapes),
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
