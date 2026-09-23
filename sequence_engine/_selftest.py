"""Built-in self-checks for the engine. Runs in memory; writes no files."""

from __future__ import annotations

import math
import struct
import sys

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
