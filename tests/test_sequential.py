"""Tests for :class:`sequence_engine.Sequential` and the layer protocol.

These tests use a caller-supplied layer (never only the shipped reference
layers) so that the three-capability protocol is proven independently.
"""

import unittest

from sequence_engine import Tensor, Sequential


class AccumLayer:
    """Minimal caller-supplied recurrent layer.

    ``z_t = w * x_t + u * h_{t-1}`` over input shape ``(T, B, 1)``; the
    per-batch hidden cell is shape ``(B, 1)`` and the output is every step's
    ``z``.  Linear activation keeps finite-difference expectations exact.
    """

    def __init__(self, w=0.3, u=0.2):
        self.w = Tensor([float(w)])
        self.u = Tensor([float(u)])
        self._cache = None

    def parameters(self):
        return [self.w, self.u]

    def forward(self, x, h):
        t_steps, batch = x.shape[0], x.shape[1]
        if h is None:
            cur = [0.0] * batch
        else:
            if h.shape != [batch, 1]:
                raise ValueError("bad hidden shape")
            cur = list(h._flat)
        inputs, states, outputs = [], [], []
        ww, uu = self.w._flat[0], self.u._flat[0]
        for t in range(t_steps):
            xt = [x._flat[t * batch + b] for b in range(batch)]
            inputs.append(xt)
            states.append(cur)
            nxt = [ww * xt[b] + uu * cur[b] for b in range(batch)]
            outputs.extend(nxt)
            cur = nxt
        self._cache = (inputs, states, t_steps, batch)
        y = Tensor._from_flat(outputs, (t_steps, batch, 1))
        h_new = Tensor._from_flat(cur, (batch, 1))
        return y, h_new

    def backward(self, dy):
        if self._cache is None:
            raise RuntimeError("backward without forward")
        inputs, states, t_steps, batch = self._cache
        self._cache = None
        ww, uu = self.w._flat[0], self.u._flat[0]
        dx = [0.0] * (t_steps * batch)
        dw = 0.0
        du = 0.0
        dh_in = [0.0] * batch
        for t in range(t_steps - 1, -1, -1):
            grad_z = [dy._flat[t * batch + b] + dh_in[b] for b in range(batch)]
            dh_out = [0.0] * batch
            for b in range(batch):
                gz = grad_z[b]
                dw += inputs[t][b] * gz
                du += states[t][b] * gz
                dx[t * batch + b] = ww * gz
                dh_out[b] = uu * gz
            dh_in = dh_out
        self.w._accumulate_grad(Tensor([dw]))
        self.u._accumulate_grad(Tensor([du]))
        return Tensor._from_flat(dx, (t_steps, batch, 1))


class ScaleLayer:
    """Stateless caller layer: ``y = s * x``; hidden cell is a zero marker."""

    def __init__(self, scale=2.0):
        self.scale = Tensor([scale])
        self._cache = None

    def parameters(self):
        return [self.scale]

    def forward(self, x, h):
        self._cache = list(x._flat), x._shape
        s = self.scale._flat[0]
        y = Tensor._from_flat([v * s for v in x._flat], x._shape)
        return y, Tensor.zeros((1,))

    def backward(self, dy):
        if self._cache is None:
            raise RuntimeError("backward without forward")
        x_flat, shape = self._cache
        self._cache = None
        s = self.scale._flat[0]
        self.scale._accumulate_grad(
            Tensor([sum(x * g for x, g in zip(x_flat, dy._flat))])
        )
        return Tensor._from_flat([g * s for g in dy._flat], shape)


def chunk(values, t_steps, batch=1):
    return Tensor(_reshape(values, (t_steps, batch, 1)))


def _reshape(flat, shape):
    out = []
    for t in range(shape[0]):
        time_row = []
        for b in range(shape[1]):
            time_row.append([float(flat[t * shape[1] + b])])
        out.append(time_row)
    return out


class ConstructionTest(unittest.TestCase):
    def test_empty_container_rejected(self):
        with self.assertRaises(ValueError):
            Sequential([])

    def test_non_iterable_rejected(self):
        with self.assertRaises(ValueError):
            Sequential(None)

    def test_capability_less_layer_rejected(self):
        with self.assertRaises(ValueError):
            Sequential([object()])

    def test_partial_capability_layer_rejected(self):
        class OnlyForward:
            def forward(self, x, h):
                return x, h

        with self.assertRaises(ValueError) as ctx:
            Sequential([OnlyForward()])
        self.assertIn("backward", str(ctx.exception))
        self.assertIn("parameters", str(ctx.exception))

    def test_parameters_in_registration_order(self):
        a, b, c = AccumLayer(), ScaleLayer(), AccumLayer()
        model = Sequential([a, b, c])
        self.assertEqual(
            model.parameters(),
            [a.w, a.u, b.scale, c.w, c.u],
        )


class ForwardTest(unittest.TestCase):
    def setUp(self):
        self.layer = AccumLayer()
        self.model = Sequential([self.layer])

    def test_cold_start_is_zero_state(self):
        x = chunk([1.0, 2.0, 3.0], 3)
        y, hidden = self.model.forward(x)
        # h evolves 0 -> w*1 -> w*2 + u*h1 -> w*3 + u*h2, using the same
        # float operation order as the engine.
        w, u = 0.3, 0.2
        h1 = w * 1.0 + u * 0.0
        h2 = w * 2.0 + u * h1
        h3 = w * 3.0 + u * h2
        self.assertEqual(hidden[0].tolist(), [[h3]])
        self.assertEqual(y.shape, [3, 1, 1])

    def test_hidden_stack_has_one_cell_per_layer(self):
        model = Sequential([AccumLayer(), ScaleLayer()])
        _, hidden = model.forward(chunk([1.0, 2.0], 2))
        self.assertEqual(len(hidden), 2)
        self.assertEqual(hidden[0].shape, [1, 1])
        self.assertEqual(hidden[1].shape, [1])

    def test_batch_axis_reflected_in_shapes(self):
        x = Tensor([[[1.0], [2.0]], [[3.0], [4.0]]])  # (T=2, B=2, 1)
        y, hidden = self.model.forward(x)
        self.assertEqual(y.shape, [2, 2, 1])
        self.assertEqual(hidden[0].shape, [2, 1])

    def test_empty_batch_rejected(self):
        with self.assertRaises(ValueError):
            self.model.forward(Tensor.zeros((0, 1, 1)))

    def test_non_tensor_batch_rejected(self):
        with self.assertRaises(ValueError):
            self.model.forward([[[1.0]]])

    def test_wrong_hidden_length_rejected(self):
        model = Sequential([AccumLayer(), ScaleLayer()])
        x = chunk([1.0], 1)
        with self.assertRaises(ValueError):
            model.forward(x, [Tensor.zeros((1, 1))])

    def test_wrong_hidden_shape_rejected(self):
        x = chunk([1.0], 1)
        with self.assertRaises(ValueError):
            self.model.forward(x, [Tensor.zeros((2, 1))])

    def test_non_tensor_hidden_cell_rejected(self):
        with self.assertRaises(ValueError):
            self.model.forward(chunk([1.0], 1), [None])

    def test_returned_hidden_is_a_detached_snapshot(self):
        _, hidden = self.model.forward(chunk([1.0, 2.0], 2))
        hidden[0]._flat[0] = 42.0
        _, again = self.model.forward(chunk([1.0, 2.0], 2))
        self.assertNotEqual(again[0]._flat[0], 42.0)


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.model = Sequential([AccumLayer()])
        self.x = chunk([1.0, 2.0], 2)

    def test_backward_before_forward(self):
        with self.assertRaises(RuntimeError):
            self.model.backward(Tensor([0.0]))

    def test_one_backward_per_forward(self):
        y, _ = self.model.forward(self.x)
        self.model.backward(Tensor([sum(y._flat)]))
        with self.assertRaises(RuntimeError):
            self.model.backward(Tensor([0.0]))

    def test_second_forward_resets_then_one_backward(self):
        self.model.forward(self.x)
        y, _ = self.model.forward(self.x)
        self.model.backward(Tensor([sum(y._flat)]))
        with self.assertRaises(RuntimeError):
            self.model.backward(Tensor([0.0]))

    def test_non_scalar_loss_is_value_error(self):
        self.model.forward(self.x)
        with self.assertRaises(ValueError):
            self.model.backward(Tensor([[1.0, 2.0]]))

    def test_non_tensor_loss_is_value_error(self):
        self.model.forward(self.x)
        with self.assertRaises(ValueError):
            self.model.backward(1.0)


class GradientTest(unittest.TestCase):
    def setUp(self):
        self.model = Sequential([AccumLayer(w=0.3, u=0.2)])
        self.x = chunk([0.5, -0.8, 1.1, 0.2], 4)

    def _loss(self):
        y, _ = self.model.forward(self.x)
        return sum(y._flat)

    def test_analytic_gradient_matches_finite_differences(self):
        target = self._loss()
        self.model.backward(Tensor([target]))
        eps = 1e-5
        for param in self.model.parameters():
            analytic = param.grad.tolist()[0]
            original = param._flat[0]
            param._flat[0] = original + eps
            plus = self._loss()
            param._flat[0] = original - eps
            minus = self._loss()
            param._flat[0] = original
            numeric = (plus - minus) / (2 * eps)
            self.assertAlmostEqual(analytic, numeric, places=5)

    def test_zero_grad_resets_to_zero(self):
        self.model.forward(self.x)
        self.model.backward(Tensor([self._loss()]))
        self.assertTrue(
            all(p.grad is not None for p in self.model.parameters())
        )
        self.model.zero_grad()
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(all(v == 0.0 for v in param.grad._flat))

    def test_accumulation_across_calls_is_plain_addition(self):
        x1 = chunk([0.5, -0.8], 2)
        x2 = chunk([1.1, 0.2], 2)

        # Combined run: backward on A, then backward on B, no reset.
        combined = Sequential([AccumLayer(w=0.3, u=0.2)])
        y1, h1 = combined.forward(x1)
        combined.backward(Tensor([sum(y1._flat)]))
        y2, _ = combined.forward(x2, h1)
        combined.backward(Tensor([sum(y2._flat)]))

        # Split run: capture A's contribution and B's contribution separately.
        split = Sequential([AccumLayer(w=0.3, u=0.2)])
        sy1, sh1 = split.forward(x1)
        split.backward(Tensor([sum(sy1._flat)]))
        g_a = [list(p.grad._flat) for p in split.parameters()]
        split.zero_grad()
        sy2, _ = split.forward(x2, sh1)
        split.backward(Tensor([sum(sy2._flat)]))
        g_b = [list(p.grad._flat) for p in split.parameters()]

        for param, ga, gb in zip(combined.parameters(), g_a, g_b):
            # One ordinary float add per element, A then B (call order).
            self.assertEqual(param.grad._flat,
                             [ga[i] + gb[i] for i in range(len(ga))])


class TruncationTest(unittest.TestCase):
    def setUp(self):
        self.values = [0.4, -0.6, 0.9, 0.1, -0.3, 0.7]
        self.x_a = chunk(self.values[:3], 3)
        self.x_b = chunk(self.values[3:], 3)
        self.x_full = chunk(self.values, 6)

    def _fresh(self):
        return Sequential([AccumLayer(w=0.35, u=0.25)])

    def _chunk_b_gradients_isolated(self):
        model = self._fresh()
        _, h_a = model.forward(self.x_a)
        model.zero_grad()  # discard A's gradients, keep carried values
        y_b, _ = model.forward(self.x_b, h_a)
        model.backward(Tensor([sum(y_b._flat)]))
        return [list(p.grad._flat) for p in model.parameters()]

    def test_gradient_does_not_leak_across_chunk_boundary(self):
        isolated = self._chunk_b_gradients_isolated()

        model = self._fresh()
        _, h_a = model.forward(self.x_a)          # A's cache is live...
        y_b, _ = model.forward(self.x_b, h_a)     # ...then overwritten here
        model.backward(Tensor([sum(y_b._flat)]))
        for param, expected in zip(model.parameters(), isolated):
            self.assertEqual(param.grad._flat, expected)

    def test_full_unroll_differs_from_truncated(self):
        truncated = self._chunk_b_gradients_isolated()

        full = self._fresh()
        y, _ = full.forward(self.x_full)
        full.backward(Tensor([sum(y._flat)]))
        full_grads = [p.grad._flat for p in full.parameters()]
        self.assertNotEqual(full_grads, truncated)

    def test_no_hidden_restarts_from_zero(self):
        seeded = self._fresh()
        _, h_a = seeded.forward(self.x_a)
        y_seeded, _ = seeded.forward(self.x_b, h_a)

        cold = self._fresh()
        y_cold, _ = cold.forward(self.x_b, None)

        fresh = self._fresh()
        y_fresh, _ = fresh.forward(self.x_b)

        self.assertEqual(y_cold._flat, y_fresh._flat)
        self.assertNotEqual(y_cold._flat, y_seeded._flat)

    def test_only_values_cross_boundary_bitwise(self):
        # The carried cell equals the observable final value of chunk A, and
        # chunk B's result is bitwise identical whether it consumes that cell
        # directly or an equivalent tensor rebuilt from its listed values.
        model = self._fresh()
        _, h_a = model.forward(self.x_a)
        y1, _ = model.forward(self.x_b, h_a)

        model2 = self._fresh()
        _, h_a2 = model2.forward(self.x_a)
        rebuilt = Tensor(h_a2[0].tolist())
        y2, _ = model2.forward(self.x_b, [rebuilt])
        self.assertEqual(y1._flat, y2._flat)


if __name__ == "__main__":
    unittest.main()
