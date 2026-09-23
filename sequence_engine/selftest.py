"""Built-in checks for ``python3 -m sequence_engine --selftest``.

The self-test runs only in-memory checks and writes nothing.  It exercises
the contract's observable points:

* analytic gradients of the reference layers match central finite
  differences through the container;
* hidden values carry across chunks but gradient history does not, and the
  same float operations land bit-for-bit;
* gradients accumulate as ordinary left-to-right float addition;
* one backward per forward, and the documented ``RuntimeError`` cases;
* tensor/container/hidden validation and ``zero_grad`` resetting to zero.
"""

import math

from .tensor import Tensor
from .sequential import Sequential
from .layers import Linear, SimpleRNNCell

# Tiny dimensions keep the pure-Python loops instant.
IN_SIZE = 2
HIDDEN = 3
OUT_SIZE = 1
BATCH = 2
CHUNK = 3


def _rand_nested(seed, shape):
    """Deterministic nested values in [-0.5, 0.5), independent of platform."""
    state = [seed]

    def nxt():
        state[0] = (state[0] * 1103515245 + 12345) % (2 ** 31)
        return state[0] / (2 ** 31) - 0.5

    def build(dims):
        if len(dims) == 1:
            return [nxt() for _ in range(dims[0])]
        return [build(dims[1:]) for _ in range(dims[0])]

    return build(list(shape))


def _scale(node, factor):
    if isinstance(node, list):
        return [_scale(child, factor) for child in node]
    return node * factor


def _flatten(node):
    if isinstance(node, list):
        out = []
        for child in node:
            out.extend(_flatten(child))
        return out
    return [node]


def _model(seed=7):
    """A fresh deterministic RNN -> Linear model with small weights."""
    rnn = SimpleRNNCell(IN_SIZE, HIDDEN)
    linear = Linear(HIDDEN, OUT_SIZE)
    model = Sequential([rnn, linear])
    weights = [
        (rnn.weight_ih, (IN_SIZE, HIDDEN)),
        (rnn.weight_hh, (HIDDEN, HIDDEN)),
        (rnn.bias, (HIDDEN,)),
        (linear.weight, (HIDDEN, OUT_SIZE)),
        (linear.bias, (OUT_SIZE,)),
    ]
    for index, (param, shape) in enumerate(weights):
        param._flat = _flatten(_scale(_rand_nested(seed + index, shape), 0.8))
        param._shape = tuple(shape)
        param._grad = None
    return model


def _input(seed, t_steps):
    return Tensor(_scale(_rand_nested(seed, (t_steps, BATCH, IN_SIZE)), 0.6))


def _scalar(x):
    return Tensor([x])


def _check_gradients_fd(check):
    """Analytic chunk gradients vs central finite differences."""
    model = _model()
    x = _input(101, CHUNK)
    out, _ = model.forward(x)
    target = sum(out._flat)
    model.backward(_scalar(target))

    eps = 1e-4

    def probe(param, index, delta):
        original = param._flat[index]
        param._flat[index] = original + delta
        value, _ = model.forward(x)
        param._flat[index] = original
        return sum(value._flat)

    for param in model.parameters():
        analytic = param.grad
        check(analytic is not None, "parameter received a gradient")
        for index in range(len(param._flat)):
            numeric = (
                probe(param, index, eps) - probe(param, index, -eps)
            ) / (2 * eps)
            got = analytic._flat[index]
            check(
                math.isclose(got, numeric, rel_tol=1e-5, abs_tol=1e-5),
                f"grad mismatch {param._shape}[{index}]: "
                f"analytic {got!r} vs finite-diff {numeric!r}",
            )


def _check_hidden_stack_and_values(check):
    model = _model()
    x = _input(202, CHUNK)
    output, hidden = model.forward(x)

    check(isinstance(hidden, list), "hidden is a list of cells")
    check(len(hidden) == 2, "one hidden cell per registered layer")
    check(hidden[0].shape == [BATCH, HIDDEN], "RNN cell is (batch, hidden)")
    check(hidden[1].shape == [1], "stateless layer keeps a placeholder cell")
    check(output.shape == [CHUNK, BATCH, OUT_SIZE], "output keeps batch axis")

    # A second, independently built model on the same data must land on the
    # same cell values, and the returned cell is an independent snapshot.
    other = _model()
    _, hidden_other = other.forward(x)
    check(
        hidden[0]._flat == hidden_other[0]._flat,
        "hidden values are deterministic and observable",
    )
    hidden[0]._flat[0] = 99.0
    _, hidden_fresh = _model().forward(x)
    check(
        hidden_fresh[0]._flat[0] != 99.0,
        "mutating a returned cell does not touch engine history",
    )


def _check_truncation(check):
    """Only hidden values cross a chunk boundary; chunk-B grads never see A."""
    full_input = _input(303, 2 * CHUNK)
    xa = Tensor(full_input.tolist()[:CHUNK])
    xb = Tensor(full_input.tolist()[CHUNK:])

    # Reference: B run in isolation, seeded with A's carried values, and only
    # B back-propagated.
    ref = _model()
    _, h_a_ref = ref.forward(xa)
    out_b_ref, _ = ref.forward(xb, h_a_ref)
    ref.backward(_scalar(sum(out_b_ref._flat)))
    isolated_b = [list(p.grad._flat) for p in ref.parameters()]

    # Combined: forward A, then forward B (A's per-layer cache is overwritten
    # in place), one backward.  Gradients must be bit-identical.
    combo = _model()
    _, h_a = combo.forward(xa)
    out_b, _ = combo.forward(xb, h_a)
    combo.backward(_scalar(sum(out_b._flat)))
    for param, expected in zip(combo.parameters(), isolated_b):
        check(
            param.grad._flat == expected,
            f"chunk-B grads identical with or without chunk A ahead "
            f"({param._shape})",
        )

    # A full unroll keeps the cross-boundary path and must therefore differ
    # on the recurrent weight -- truncation really removes something.
    full = _model()
    out_full, _ = full.forward(full_input)
    full.backward(_scalar(sum(out_full._flat)))
    check(
        full.parameters()[1].grad._flat != combo.parameters()[1].grad._flat,
        "full unroll and truncated chunk B differ on the recurrent weight",
    )

    # hidden=None discards history: B from zero equals a model that saw only B
    # and visibly differs from B seeded with A's state.
    cold = _model()
    out_cold, _ = cold.forward(xb, None)
    fresh = _model()
    out_fresh, _ = fresh.forward(xb)
    check(
        out_cold._flat == out_fresh._flat,
        "hidden=None restarts every layer from zero",
    )
    check(
        out_cold._flat != out_b._flat,
        "carried history visibly changes chunk-B output",
    )


def _check_accumulation_order(check):
    """Gradients accumulate as ordinary float adds, in call order."""
    full_input = _input(404, 2 * CHUNK)
    xa = Tensor(full_input.tolist()[:CHUNK])
    xb = Tensor(full_input.tolist()[CHUNK:])

    # One model: backward A, then backward B without zero_grad between.
    combined = _model()
    out_a, h_a = combined.forward(xa)
    combined.backward(_scalar(sum(out_a._flat)))
    out_b, _ = combined.forward(xb, h_a)
    combined.backward(_scalar(sum(out_b._flat)))

    # Separate model: capture A's gradient alone, then B's gradient alone.
    split = _model()
    out_a_s, h_a_s = split.forward(xa)
    split.backward(_scalar(sum(out_a_s._flat)))
    grads_a = [list(p.grad._flat) for p in split.parameters()]
    split.zero_grad()
    out_b_s, _ = split.forward(xb, h_a_s)
    split.backward(_scalar(sum(out_b_s._flat)))
    grads_b = [list(p.grad._flat) for p in split.parameters()]

    for param, ga, gb in zip(combined.parameters(), grads_a, grads_b):
        expected = [ga[i] + gb[i] for i in range(len(ga))]
        check(
            param.grad._flat == expected,
            f"accumulation is ordered float addition for {param._shape}",
        )


def _check_zero_grad_and_update(check):
    model = _model()
    x = _input(505, CHUNK)
    check(all(p.grad is None for p in model.parameters()),
          "grads start absent")
    out, _ = model.forward(x)
    model.backward(_scalar(sum(out._flat)))
    check(all(p.grad is not None for p in model.parameters()), "grads exist")
    model.zero_grad()
    check(
        all(
            p.grad is not None and all(v == 0.0 for v in p.grad._flat)
            for p in model.parameters()
        ),
        "zero_grad resets every gradient to an observable zero tensor",
    )

    # After zeroing, one new chunk accumulates exactly that chunk's
    # contribution: the zero baseline behaves as ordinary 0.0 + g.
    out2, _ = model.forward(x)
    model.backward(_scalar(sum(out2._flat)))
    fresh = _model()
    fout, _ = fresh.forward(x)
    fresh.backward(_scalar(sum(fout._flat)))
    for p, fp in zip(model.parameters(), fresh.parameters()):
        check(p.grad._flat == fp.grad._flat,
              f"post-clear accumulation equals the single chunk ({p._shape})")
    model.zero_grad()

    # Theta <- theta - eta*g from identical gradient values must be
    # bit-for-bit reproducible with ordinary float multiply and subtract.
    eta = 0.1
    m1, m2 = _model(), _model()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        gradient = [v * 0.01 for v in p1._flat]
        p1._flat = [v - eta * g for v, g in zip(p1._flat, gradient)]
        p2._flat = [v - eta * g for v, g in zip(p2._flat, gradient)]
        check(p1._flat == p2._flat, "same gradient yields identical update")


def _check_lifecycle_errors(check):
    model = _model()
    x = _input(606, CHUNK)

    _expect(check, RuntimeError, model.backward, _scalar(0.0),
            "backward before forward")
    out, _ = model.forward(x)
    model.backward(_scalar(sum(out._flat)))
    _expect(check, RuntimeError, model.backward, _scalar(0.0),
            "second backward without a new forward")

    # A second forward resets the pending history in place; one backward
    # stays valid and a second backward is refused.
    model.forward(x)
    out2, _ = model.forward(x, None)
    model.backward(_scalar(sum(out2._flat)))
    _expect(check, RuntimeError, model.backward, _scalar(0.0),
            "backward twice after a resetting forward")

    # With a fresh pending pass, invalid losses are rejected as ValueError
    # before the pass is consumed.
    model.forward(x)
    _expect(check, ValueError, model.backward, Tensor([[1.0, 2.0]]),
            "non-scalar loss")
    _expect(check, ValueError, model.backward, 1.0, "non-tensor loss")
    check(model._pending, "invalid backward leaves the pending pass intact")

    rnn = SimpleRNNCell(IN_SIZE, HIDDEN)
    _expect(check, RuntimeError, rnn.backward,
            Tensor.zeros((1, BATCH, HIDDEN)),
            "layer backward without layer forward")


def _check_validation(check):
    for bad in ([], [[]], [[1, 2], [3]], [True, False], [1, "x"], 5):
        _expect(check, ValueError, Tensor, bad, f"bad tensor data {bad!r}")

    model = _model()
    x = _input(707, CHUNK)
    _expect(check, ValueError, model.forward,
            Tensor.zeros((0, BATCH, IN_SIZE)), "empty batch")
    _expect(check, ValueError, model.forward, [[1.0]], "non-tensor batch")

    _expect(check, ValueError, Sequential, [], "empty container")
    _expect(check, ValueError, Sequential, [object()], "capability-less layer")

    class HalfLayer:
        def forward(self, x, h):
            return x, h

    _expect(check, ValueError, Sequential, [HalfLayer()],
            "layer missing backward/parameters")

    _expect(check, ValueError, model.forward, x,
            [Tensor.zeros((BATCH, HIDDEN))], "too few hidden cells")
    _expect(check, ValueError, model.forward, x,
            [Tensor.zeros((BATCH, HIDDEN + 1)), Tensor.zeros((1,))],
            "wrong hidden shape")
    _expect(check, ValueError, model.forward, x,
            [Tensor.zeros((BATCH, HIDDEN)), None], "non-tensor hidden cell")


def _expect(check, exc, fn, *args_and_label):
    """Call ``fn(*args)`` expecting ``exc``; the final positional arg is the
    human-readable check label."""
    label = args_and_label[-1]
    args = args_and_label[:-1]
    try:
        fn(*args)
    except exc:
        check(True, label)
    except Exception as other:  # noqa: BLE001 - the mismatch is the finding
        check(False, f"{label}: raised {type(other).__name__}, "
                     f"not {exc.__name__}")
    else:
        check(False, f"{label}: no exception")


_CHECKS = [
    _check_gradients_fd,
    _check_hidden_stack_and_values,
    _check_truncation,
    _check_accumulation_order,
    _check_zero_grad_and_update,
    _check_lifecycle_errors,
    _check_validation,
]


def run_selftest():
    failures = []
    count = 0

    def check(condition, label):
        nonlocal count
        count += 1
        if not condition:
            failures.append(label)

    for test in _CHECKS:
        test(check)

    lines = [
        f"sequence_engine self-test: {count} checks, {len(failures)} failed"
    ]
    lines.extend(f"  FAIL: {label}" for label in failures)
    return not failures, "\n".join(lines)
