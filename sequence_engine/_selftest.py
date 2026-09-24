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


def _check_v2_version_and_v1_migration():
    # Native saves carry format version 2.
    seq, _ = _fresh_stack()
    out, h1 = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    raw = bytearray()
    seq.save(raw)
    _check(
        struct.unpack("<I", bytes(raw)[8:12])[0] == 2,
        "new full checkpoints are format version 2",
    )

    # A genuine version-1 file loads, migrates and records source 1.
    v1 = _v1_golden.trained_bytes()
    _check(struct.unpack("<I", v1[8:12])[0] == 1, "golden fixture really is v1")
    migrated, _ = _fresh_stack()
    restored_hidden = migrated.load(bytes(v1))
    _check(migrated.loaded_from_version == 1, "a v1 load records source version 1")
    _check(
        [s.tolist() for s in restored_hidden] == [s.tolist() for s in h1],
        "v1 migration restores slice-boundary hidden state",
    )
    _check(
        [p.tolist() for p in migrated.parameters()]
        == [p.tolist() for p in seq.parameters()],
        "v1 migration restores parameters exactly",
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

    # A migrated state re-saves as native v2, source recorded as 2.
    rebuf = bytearray()
    migrated.save(rebuf)
    _check(
        struct.unpack("<I", bytes(rebuf)[8:12])[0] == 2,
        "a migrated checkpoint re-saves natively as v2",
    )
    again, _ = _fresh_stack()
    again.load(bytes(rebuf))
    _check(again.loaded_from_version == 2, "a native v2 load records source version 2")

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

    # v2 header with a removed or added field is rejected wholesale.
    victim3, _ = _fresh_stack()
    _expect(
        ValueError,
        lambda: victim3.load(_repack_full(bytes(raw), lambda h: h.pop("layers"))),
        "v2 checkpoint missing a field is rejected",
    )
    _expect(
        ValueError,
        lambda: _fresh_stack()[0].load(
            _repack_full(bytes(raw), lambda h: h.update(surprise=1))
        ),
        "v2 checkpoint with an extra field is rejected",
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
    hidden_indices = {i for i, _s, _t in items2 if i >= 2 * len(seq.parameters())}
    _check(
        hidden_indices == set(range(2 * len(seq.parameters()), 2 * len(seq.parameters()) + 2)),
        "the delta that first fixes hidden state carries every hidden slot",
    )

    # Update: parameters change, hidden does not.
    seq.update(_LR)
    full_after = bytearray()
    seq.save(full_after)
    seq.save(chain)  # no-op state already captured? update changed params -> delta

    head_doc = _checkpoint.load_chain_memory(chain)
    _check(
        _checkpoint.build_bytes(head_doc) == bytes(full_after),
        "reassembling the chain reproduces the full snapshot bit for bit",
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
        == bytes(full_after),
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
    ("v2 format and v1 migration", _check_v2_version_and_v1_migration),
    ("incremental checkpoint chain", _check_incremental_chain_in_memory),
    ("concurrent update/save/load", _check_concurrent_updates_saves_loads),
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
