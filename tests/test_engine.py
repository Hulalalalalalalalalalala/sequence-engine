import unittest

from sequence_engine import Sequential, Tensor


class RecurrentScale:
    """Test layer: y[b][j] = x[b][j] * w[j] + h[b][j]; the hidden slot is y."""

    def __init__(self, weights):
        self.w = Tensor(weights)
        self._cache = None

    def parameters(self):
        return [self.w]

    def forward(self, x, hidden):
        xs = x.tolist()
        ws = self.w.tolist()
        if hidden is None:
            hs = [[0.0] * len(ws) for _ in xs]
        else:
            hs = hidden.tolist()
            if len(hs) != len(xs) or any(len(row) != len(ws) for row in hs):
                raise ValueError("hidden state shape mismatch")
        y = [
            [xs[r][j] * ws[j] + hs[r][j] for j in range(len(ws))]
            for r in range(len(xs))
        ]
        self._cache = (xs, hs)
        return Tensor(y), Tensor(y)

    def backward(self, upstream):
        xs, _ = self._cache
        dy = upstream.tolist() if isinstance(upstream, Tensor) else upstream
        if isinstance(dy, (int, float)) and not isinstance(dy, bool):
            dy = [[float(dy)] * len(xs[0]) for _ in xs]
        ws = self.w.tolist()
        gw = [
            sum(dy[r][j] * xs[r][j] for r in range(len(xs)))
            for j in range(len(ws))
        ]
        if self.w.grad is None:
            self.w.grad = Tensor(gw)
        else:
            self.w.grad = Tensor(
                [a + b for a, b in zip(self.w.grad.tolist(), gw)]
            )
        return Tensor(
            [[dy[r][j] * ws[j] for j in range(len(ws))] for r in range(len(xs))]
        )


def make_stack():
    first = RecurrentScale([2.0, 3.0])
    second = RecurrentScale([4.0, 5.0])
    return Sequential([first, second]), first, second


class TensorTests(unittest.TestCase):
    def test_shape_and_tolist(self):
        t = Tensor([[1, 2], [3, 4]])
        self.assertEqual(t.shape, [2, 2])
        self.assertEqual(t.tolist(), [[1, 2], [3, 4]])

    def test_tolist_returns_a_copy(self):
        t = Tensor([[1.0]])
        t.tolist()[0][0] = 9.0
        self.assertEqual(t.tolist(), [[1.0]])

    def test_scalar(self):
        t = Tensor(2.5)
        self.assertEqual(t.shape, [])
        self.assertEqual(t.tolist(), 2.5)

    def test_grad_starts_empty(self):
        self.assertIsNone(Tensor([1.0]).grad)

    def test_rejects_empty_data(self):
        with self.assertRaises(ValueError):
            Tensor([])
        with self.assertRaises(ValueError):
            Tensor([[]])

    def test_rejects_ragged_data(self):
        with self.assertRaises(ValueError):
            Tensor([[1.0], [2.0, 3.0]])
        with self.assertRaises(ValueError):
            Tensor([1.0, [2.0]])

    def test_rejects_booleans(self):
        with self.assertRaises(ValueError):
            Tensor(True)
        with self.assertRaises(ValueError):
            Tensor([1.0, False])

    def test_rejects_non_numeric(self):
        with self.assertRaises(ValueError):
            Tensor("nope")


class ContainerValidationTests(unittest.TestCase):
    def test_rejects_empty_container(self):
        with self.assertRaises(ValueError):
            Sequential([])

    def test_rejects_layers_without_capabilities(self):
        with self.assertRaises(ValueError):
            Sequential([object()])

        class ForwardOnly:
            def forward(self, x, hidden):
                return x, hidden

        with self.assertRaises(ValueError):
            Sequential([ForwardOnly()])

    def test_rejects_bad_batch(self):
        seq, _, _ = make_stack()
        with self.assertRaises(ValueError):
            seq.forward("nope")
        with self.assertRaises(ValueError):
            seq.forward(Tensor(1.0))

    def test_rejects_bad_hidden(self):
        seq, _, _ = make_stack()
        with self.assertRaises(ValueError):
            seq.forward(Tensor([[1.0, 1.0]]), [Tensor([[0.0, 0.0]])])
        with self.assertRaises(ValueError):
            seq.forward(Tensor([[1.0, 1.0]]), [Tensor([[0.0, 0.0]]), "nope"])

    def test_rejects_mismatched_hidden_shape(self):
        seq, _, _ = make_stack()
        seq.forward(Tensor([[1.0, 1.0]]))
        bad = [Tensor([[0.0, 0.0]]), Tensor([[0.0, 0.0, 0.0]])]
        with self.assertRaises(ValueError):
            seq.forward(Tensor([[1.0, 1.0]]), bad)


class ForwardBackwardTests(unittest.TestCase):
    def test_forward_shapes_and_hidden_stack(self):
        seq, _, _ = make_stack()
        out, hidden = seq.forward(Tensor([[1.0, 1.0]]))
        self.assertEqual(out.tolist(), [[8.0, 15.0]])
        self.assertEqual(out.shape, [1, 2])
        self.assertEqual(len(hidden), 2)
        self.assertEqual(hidden[0].tolist(), [[2.0, 3.0]])
        self.assertEqual(hidden[1].tolist(), [[8.0, 15.0]])

    def test_backward_hand_computed(self):
        seq, first, second = make_stack()
        seq.forward(Tensor([[1.0, 1.0]]))
        seq.backward(1.0)
        self.assertEqual(first.w.grad.tolist(), [4.0, 5.0])
        self.assertEqual(second.w.grad.tolist(), [2.0, 3.0])

    def test_gradients_accumulate_in_call_order(self):
        seq, first, second = make_stack()
        seq.forward(Tensor([[1.0, 1.0]]))
        seq.backward(1.0)
        seq.forward(Tensor([[1.0, 1.0]]))
        seq.backward(1.0)
        self.assertEqual(first.w.grad.tolist(), [8.0, 10.0])
        self.assertEqual(second.w.grad.tolist(), [4.0, 6.0])

    def test_hidden_carries_values_not_history(self):
        seq, first, second = make_stack()
        _, hidden = seq.forward(Tensor([[1.0, 1.0]]))
        out, hidden2 = seq.forward(Tensor([[1.0, 1.0]]), hidden)
        # layer 1: 1*2+2=4, 1*3+3=6; layer 2: 4*4+8=24, 6*5+15=45
        self.assertEqual(out.tolist(), [[24.0, 45.0]])
        self.assertEqual(hidden2[0].tolist(), [[4.0, 6.0]])
        seq.backward(1.0)
        # Only the second segment's inputs contribute to the gradients.
        self.assertEqual(first.w.grad.tolist(), [4.0, 5.0])
        self.assertEqual(second.w.grad.tolist(), [4.0, 6.0])

    def test_truncated_segments_accumulate_per_segment_gradients(self):
        seq, first, second = make_stack()
        _, hidden = seq.forward(Tensor([[1.0, 1.0]]))
        seq.backward(1.0)
        seq.forward(Tensor([[1.0, 1.0]]), hidden)
        seq.backward(1.0)
        # seg1: first=[4,5], second=[2,3]; seg2: first=[4,5], second=[4,6]
        self.assertEqual(first.w.grad.tolist(), [8.0, 10.0])
        self.assertEqual(second.w.grad.tolist(), [6.0, 9.0])

    def test_backward_requires_forward(self):
        seq, _, _ = make_stack()
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)

    def test_one_backward_per_forward(self):
        seq, _, _ = make_stack()
        seq.forward(Tensor([[1.0, 1.0]]))
        seq.backward(1.0)
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)

    def test_zero_grad(self):
        seq, first, second = make_stack()
        seq.forward(Tensor([[1.0, 1.0]]))
        seq.backward(1.0)
        seq.zero_grad()
        self.assertEqual(first.w.grad.tolist(), [0.0, 0.0])
        self.assertEqual(second.w.grad.tolist(), [0.0, 0.0])

    def test_parameters_in_registration_order(self):
        seq, first, second = make_stack()
        self.assertEqual(seq.parameters(), [first.w, second.w])


if __name__ == "__main__":
    unittest.main()
