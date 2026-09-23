"""Built-in self-checks for the engine.

All checks are built in; the checkpoint groups create files only inside
``tempfile.TemporaryDirectory()`` directories that are removed again, so a
run leaves no files behind in the working tree.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import tempfile

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


def _check_update_step():
    seq, _ = _build(_base_weights())
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    lr = 0.1
    before = [p.tolist() for p in seq.parameters()]
    grads = [p.grad.tolist() for p in seq.parameters()]
    seq.update(lr)
    for old, grad, param in zip(before, grads, seq.parameters()):
        for old_value, grad_value, new_value in zip(
            _flatten([old]), _flatten([grad]), _flatten([param.tolist()])
        ):
            _check(
                new_value == old_value - lr * grad_value,
                "update must apply theta <- theta - lr*g to every parameter",
            )
    # Non-finite and non-numeric learning rates are refused.
    for bad in (float("inf"), float("-inf"), float("nan"), "x", None, True):
        _expect(ValueError, lambda bad=bad: seq.update(bad), f"bad lr {bad!r}")
    # A parameter with no gradient is left untouched.
    fresh, fresh_layers = _build(_base_weights())
    untouched = fresh_layers[0].wxh.tolist()
    fresh.update(0.5)
    _check(
        fresh_layers[0].wxh.tolist() == untouched,
        "update must not move parameters that have no gradient yet",
    )


def _check_loss_and_retry_contract():
    # Tuple loss form is equivalent to the scalar form.
    seq, _ = _build(_base_weights())
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward((_total(out),))
    reference, _ = _build(_base_weights())
    out_ref, _ = reference.forward(Tensor(_SEG1))
    reference.backward(_total(out_ref))
    for a, b in zip(seq.parameters(), reference.parameters()):
        _check(
            a.grad.tolist() == b.grad.tolist(),
            "(loss,) must seed backward identically to loss",
        )
    _expect(ValueError, lambda: seq.backward((1.0, 2.0)), "multi-element tuple loss")

    # An exception mid-backward does not consume the forward: the retry
    # is allowed, while a third backward is a genuine second call.
    class _Flaky:
        def __init__(self):
            self.w = Tensor([1.0, 2.0])
            self.attempts = 0

        def parameters(self):
            return [self.w]

        def forward(self, x, hidden):
            y = Tensor([[0.5, 0.5]])
            return y, y

        def backward(self, upstream):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("simulated mid-pass failure")
            self.w.grad = Tensor([1.0, 1.0])
            return Tensor([[1.0, 1.0]])

    layer = _Flaky()
    chain = Sequential([layer])
    chain.forward(Tensor([[1.0, 1.0]]))
    _expect(RuntimeError, lambda: chain.backward(1.0), "first backward fails")
    chain.backward(1.0)
    _expect(RuntimeError, lambda: chain.backward(1.0), "retry consumed the forward")


def _model_snapshot(seq):
    return [
        (p.tolist(), p.grad.tolist() if p.grad is not None else None)
        for p in seq.parameters()
    ]


def _two_segment_reference():
    """Run both segments back-to-back, then one update; return full state."""
    seq, _ = _build(_base_weights())
    out1, hidden1 = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out1))
    out2, _ = seq.forward(Tensor(_SEG2), hidden1)
    seq.backward((_total(out2),))
    seq.update(0.13)
    return out2.tolist(), _model_snapshot(seq), [s.tolist() for s in hidden1]


def _resume_from_checkpoint(target):
    seq, _ = _build(_base_weights())
    hidden = seq.load(target)
    out, _ = seq.forward(Tensor(_SEG2), hidden)
    seq.backward(_total(out))
    seq.update(0.13)
    return out.tolist(), _model_snapshot(seq)


def _checkpoint_boundary_model():
    seq, _ = _build(_base_weights())
    out, hidden = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    return seq, hidden


def _check_checkpoint_round_trip():
    ref_out, ref_state, boundary_values = _two_segment_reference()

    boundary, hidden = _checkpoint_boundary_model()
    _check(
        [slot.tolist() for slot in hidden] == boundary_values,
        "fixture boundary hidden state mismatch",
    )

    blob = boundary.save()
    _check(isinstance(blob, (bytes, bytearray)), "save() returns bytes")
    mem_out, mem_state = _resume_from_checkpoint(blob)
    _check(mem_out == ref_out, "memory resume output must match uninterrupted run")
    _check(mem_state == ref_state, "memory resume params/grads must match bitwise")

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "checkpoint.bin")
        _check(boundary.save(path) is None, "save(path) returns None")
        _check(os.listdir(directory) == ["checkpoint.bin"], "no leftover temp files")
        disk_out, disk_state = _resume_from_checkpoint(path)
        # Overwrite: same single file, fresh bytes, no backups.
        boundary.save(path)
        _check(
            os.listdir(directory) == ["checkpoint.bin"],
            "overwrite leaves no backup or temp files",
        )
    _check(disk_out == ref_out, "disk resume output must match uninterrupted run")
    _check(disk_state == ref_state, "disk resume params/grads must match bitwise")
    _check(disk_state == mem_state, "disk and memory round trips must agree")

    # Repeated saves are pure: bytes identical, state untouched.
    before = _model_snapshot(boundary)
    again = boundary.save()
    _check(again == blob, "repeated saves produce identical bytes")
    _check(_model_snapshot(boundary) == before, "saving must not mutate state")

    # save -> load -> save is byte-identical.
    reloaded, _ = _build(_base_weights())
    reloaded.load(blob)
    _check(reloaded.save() == blob, "save/load/save must be byte-identical")

    # Layers without a gradient are checkpointed and restored as zeros.
    fresh, _ = _build(_base_weights())
    fresh.forward(Tensor(_SEG1))
    _check(
        all(p.grad is None for p in fresh.parameters()),
        "fixture: no gradients before backward",
    )
    restored, _ = _build(_base_weights())
    restored.load(fresh.save())
    for param in restored.parameters():
        _check(param.grad is not None, "missing gradient restored as a zero tensor")
        _check(
            all(value == 0.0 for value in _flatten(param.grad.tolist())),
            "missing gradient restored as zeros",
        )

    # After load, the model sits at a clean boundary: backward before a
    # fresh forward is still the documented RuntimeError.
    clean, _ = _build(_base_weights())
    restored_hidden = clean.load(blob)
    _expect(RuntimeError, lambda: clean.backward(1.0), "backward right after load")
    out, _ = clean.forward(Tensor(_SEG2), restored_hidden)
    clean.backward(_total(out))


def _check_checkpoint_precision():
    special = [-0.0, 0.0, 5e-324, -5e-324, 1.7976931348623157e308,
               -2.2250738585072014e-308, 3.141592653589793]
    seq, layers = _build(_base_weights())
    layers[0].wxh._replace_data(
        [special[:3], special[3:6]], [2, 3]
    )
    layers[0].whh._replace_data(
        [[special[6], 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], [3, 3]
    )
    seq.forward(Tensor([[1.0] * _N_IN] * _BATCH))
    restored, restored_layers = _build(_base_weights())
    restored.load(seq.save())
    values = restored_layers[0].wxh.tolist()
    flat = [values[0] + values[1]]
    for expected, actual in zip(special[:6], flat[0]):
        _check(
            struct.pack(">d", actual) == struct.pack(">d", expected),
            f"float64 must round-trip bitwise: {expected!r} -> {actual!r}",
        )
    _check(
        math.copysign(1.0, restored_layers[0].wxh.tolist()[0][0]) == -1.0,
        "negative zero must keep its sign through a checkpoint",
    )
    _check(
        struct.pack(">d", restored_layers[0].whh.tolist()[0][0])
        == struct.pack(">d", special[6]),
        "pi-like float must round-trip bitwise",
    )


def _check_checkpoint_rejection():
    boundary, _ = _checkpoint_boundary_model()
    blob = boundary.save()

    def rejects(target):
        fresh, _ = _build(_base_weights())
        try:
            fresh.load(target)
        except ValueError:
            return
        raise _SelfTestFailure("checkpoint should have been rejected")

    rejects(blob[: len(blob) // 2])
    rejects(b"")
    rejects(b"not a checkpoint")
    rejects(blob + b"\x00")
    flipped = bytearray(blob)
    flipped[60] ^= 0x01
    rejects(bytes(flipped))
    versioned = bytearray(blob)
    versioned[8], versioned[9] = 0, 2
    rejects(bytes(versioned))

    # Oversized integers and non-finite floats can never be saved.
    bad_int, bad_layers = _build(_base_weights())
    bad_layers[0].bias._replace_data([1 << 70, 0, 0], [3])
    _expect(ValueError, bad_int.save, "oversized integer")
    bad_float, bad_float_layers = _build(_base_weights())
    bad_float_layers[0].bias._replace_data([float("nan"), 0, 0], [3])
    _expect(ValueError, bad_float.save, "non-finite float")

    # Shape/layer-order mismatch against the live model rejects the whole
    # checkpoint without mutating current parameters.
    wider = Sequential(
        [
            _RNNStep(_N_IN, 4,
                     [[0.0] * 4 for _ in range(_N_IN)],
                     [[0.0] * 4 for _ in range(4)],
                     [0.0] * 4),
            _RNNStep(4, _N_H2,
                     [[0.0] * _N_H2 for _ in range(4)],
                     [[0.0] * _N_H2 for _ in range(_N_H2)],
                     [0.0] * _N_H2),
        ]
    )
    wider_params = [p.tolist() for p in wider.parameters()]
    _expect(ValueError, lambda: wider.load(blob), "shape mismatch rejected")
    for param, saved in zip(wider.parameters(), wider_params):
        _check(param.tolist() == saved, "rejected load must not mutate the model")

    with tempfile.TemporaryDirectory() as directory:
        _expect(
            FileNotFoundError,
            lambda: _build(_base_weights())[0].load(
                os.path.join(directory, "missing.bin")
            ),
            "missing checkpoint file",
        )
        _expect(
            OSError,
            lambda: boundary.save(os.path.join(directory, "no-dir", "c.bin")),
            "save into a missing directory",
        )
        # A half-written target left by a crashed process is refused.
        target = os.path.join(directory, "half.bin")
        with open(target, "wb") as handle:
            handle.write(blob[: len(blob) // 2])
        rejects(target)


_GROUPS = [
    ("tensor basics", _check_tensor_basics),
    ("tensor validation", _check_tensor_validation),
    ("container validation", _check_container_validation),
    ("forward and hidden state", _check_forward_and_hidden),
    ("backward contract", _check_backward_contract),
    ("numeric gradients", _check_gradients_numeric),
    ("truncated backpropagation", _check_truncation),
    ("zero_grad and parameters", _check_zero_grad_and_parameters),
    ("update step", _check_update_step),
    ("tuple loss and retry contract", _check_loss_and_retry_contract),
    ("checkpoint round-trip", _check_checkpoint_round_trip),
    ("checkpoint precision", _check_checkpoint_precision),
    ("checkpoint rejection", _check_checkpoint_rejection),
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
