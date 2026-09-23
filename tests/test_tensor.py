"""Tests for :class:`sequence_engine.Tensor`."""

import unittest

from sequence_engine import Tensor


class TensorTest(unittest.TestCase):
    def test_shape_and_tolist_round_trip(self):
        data = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        t = Tensor(data)
        self.assertEqual(t.shape, [2, 2, 3])
        self.assertEqual(t.tolist(), data)

    def test_tolist_is_a_copy(self):
        t = Tensor([[1.0, 2.0]])
        nested = t.tolist()
        nested[0][0] = 99.0
        self.assertEqual(t.tolist(), [[1.0, 2.0]])

    def test_shape_is_a_copy(self):
        t = Tensor([1, 2, 3])
        shape = t.shape
        shape.append(4)
        self.assertEqual(t.shape, [3])

    def test_integers_widened_to_float(self):
        t = Tensor([1, 2, 3])
        self.assertEqual(t.tolist(), [1.0, 2.0, 3.0])
        self.assertIsInstance(t.tolist()[0], float)

    def test_grad_starts_absent(self):
        self.assertIsNone(Tensor([1.0]).grad)

    def test_grad_accumulates_in_call_order(self):
        p = Tensor([0.0, 0.0])
        p._accumulate_grad(Tensor([1.0, 2.0]))
        p._accumulate_grad(Tensor([0.5, 0.25]))
        self.assertEqual(p.grad.tolist(), [1.5, 2.25])

    def test_zero_grad_resets_to_observable_zero(self):
        p = Tensor([0.0, 0.0])
        p._accumulate_grad(Tensor([1.0, 2.0]))
        p._zero_grad()
        self.assertIsNotNone(p.grad)
        self.assertEqual(p.grad.tolist(), [0.0, 0.0])
        # A subsequent accumulation is exactly 0.0 + g == g.
        p._accumulate_grad(Tensor([3.0, -4.0]))
        self.assertEqual(p.grad.tolist(), [3.0, -4.0])

    def test_grad_accumulation_is_bitwise_ordinary_addition(self):
        # Repeated accumulation matches a left-to-right chain of float adds.
        contributions = [0.1, 0.2, 0.3, 0.4]
        p = Tensor([0.0])
        expected = 0.0
        for value in contributions:
            p._accumulate_grad(Tensor([value]))
            expected = expected + value
        self.assertEqual(p.grad.tolist()[0], expected)
        self.assertEqual(p.grad.tolist()[0], 0.1 + 0.2 + 0.3 + 0.4)

    def test_copy_is_independent_of_values_and_grad(self):
        p = Tensor([1.0])
        p._accumulate_grad(Tensor([9.0]))
        snapshot = p._copy()
        snapshot._flat[0] = 5.0
        self.assertEqual(p.tolist(), [1.0])
        self.assertIsNone(snapshot.grad)

    def test_invalid_data_raises_value_error(self):
        bad_inputs = [
            ([], "empty outer list"),
            ([[]], "empty nested list"),
            ([[1, 2], [3]], "ragged nesting"),
            ([[[1]], [2]], "depth-mismatched nesting"),
            ([True, False], "booleans"),
            ([1, "two"], "string leaf"),
            (5, "bare scalar"),
        ]
        for bad, label in bad_inputs:
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    Tensor(bad)

    def test_bool_deeply_nested_rejected(self):
        with self.assertRaises(ValueError):
            Tensor([[[False]]])


if __name__ == "__main__":
    unittest.main()
