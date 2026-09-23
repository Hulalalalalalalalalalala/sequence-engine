import os
import struct
import tempfile
import unittest

from sequence_engine import Sequential, Tensor
from sequence_engine import checkpoint as _checkpoint


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


class _MultiScale:
    """Layer with several independent weight vectors (distinct shapes)."""

    def __init__(self, widths):
        self.widths = list(widths)
        self.weights = [
            Tensor([float(j + 1) for j in range(width)]) for width in self.widths
        ]
        self._cache = None

    def parameters(self):
        return list(self.weights)

    def forward(self, x, hidden):
        xs = x.tolist()
        w0 = self.weights[0].tolist()
        if hidden is None:
            hs = [[0.0] * len(w0) for _ in xs]
        else:
            hs = hidden.tolist()
        y = [
            [xs[r][j] * w0[j] + hs[r][j] for j in range(len(w0))]
            for r in range(len(xs))
        ]
        self._cache = (xs, y)
        return Tensor(y), Tensor(y)

    def backward(self, upstream):
        xs, _ = self._cache
        if isinstance(upstream, Tensor):
            dy = upstream.tolist()
        else:
            dy = [[float(upstream)] * len(xs[0]) for _ in xs]
        w0 = self.weights[0].tolist()
        g0 = [
            sum(dy[r][j] * xs[r][j] for r in range(len(xs)))
            for j in range(len(w0))
        ]
        if self.weights[0].grad is None:
            self.weights[0].grad = Tensor(g0)
        for weight in self.weights[1:]:
            if weight.grad is None:
                weight.grad = Tensor([0.0] * len(weight.tolist()))
        return Tensor(
            [[dy[r][j] * w0[j] for j in range(len(w0))] for r in range(len(xs))]
        )


_SEG_A = [[1.0, 1.0]]
_SEG_B = [[2.0, 0.5]]
_LR = 0.25


def _snapshot(seq):
    return [
        (
            param.tolist(),
            param.grad.tolist() if param.grad is not None else None,
        )
        for param in seq.parameters()
    ]


def _run_reference():
    seq, _, _ = make_stack()
    out1, hidden = seq.forward(Tensor(_SEG_A))
    seq.backward(1.0)
    out2, _ = seq.forward(Tensor(_SEG_B), hidden)
    seq.backward((1.0,))
    seq.update(_LR)
    return out2.tolist(), _snapshot(seq)


def _run_resumed(checkpoint_target):
    seq, _, _ = make_stack()
    hidden = seq.load(checkpoint_target)
    out, _ = seq.forward(Tensor(_SEG_B), hidden)
    seq.backward(1.0)
    seq.update(_LR)
    return out.tolist(), _snapshot(seq)


def _checkpoint_at_segment_boundary():
    seq, _, _ = make_stack()
    seq.forward(Tensor(_SEG_A))
    seq.backward(1.0)
    return seq


class UpdateTests(unittest.TestCase):
    def test_update_rule_in_place(self):
        seq, first, second = make_stack()
        seq.forward(Tensor(_SEG_A))
        seq.backward(1.0)
        before = [p.tolist() for p in seq.parameters()]
        refs = [p for p in seq.parameters()]
        seq.update(0.5)
        # objects are preserved; values follow theta <- theta - lr * g
        self.assertEqual(seq.parameters(), refs)
        self.assertEqual(
            first.w.tolist(),
            [2.0 - 0.5 * 4.0, 3.0 - 0.5 * 5.0],
        )
        self.assertEqual(
            second.w.tolist(),
            [4.0 - 0.5 * 2.0, 5.0 - 0.5 * 3.0],
        )
        for after, param in zip(before, refs):
            self.assertEqual(len(after), len(param.tolist()))

    def test_update_with_zero_learning_rate(self):
        seq, first, _ = make_stack()
        seq.forward(Tensor(_SEG_A))
        seq.backward(1.0)
        seq.update(0)
        self.assertEqual(first.w.tolist(), [2.0, 3.0])

    def test_parameters_without_gradient_are_left_alone(self):
        seq, _, _ = make_stack()
        untouched, _ = seq.parameters()[0].tolist(), None
        seq.update(0.25)
        self.assertEqual(seq.parameters()[0].tolist(), untouched)

    def test_rejects_non_finite_or_non_numeric_lr(self):
        seq, _, _ = make_stack()
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.assertRaises(ValueError):
                seq.update(bad)
        for bad in ("0.1", None, True, [0.1]):
            with self.assertRaises(ValueError):
                seq.update(bad)


class BackwardContractTests(unittest.TestCase):
    def test_tuple_loss_is_accepted(self):
        seq, first, _ = make_stack()
        seq.forward(Tensor(_SEG_A))
        seq.backward((1.0,))
        self.assertEqual(first.w.grad.tolist(), [4.0, 5.0])

    def test_tuple_loss_must_have_one_element(self):
        seq, _, _ = make_stack()
        seq.forward(Tensor(_SEG_A))
        with self.assertRaises(ValueError):
            seq.backward(())
        with self.assertRaises(ValueError):
            seq.backward((1.0, 2.0))

    def test_malformed_argument_is_value_error_even_without_forward(self):
        seq, _, _ = make_stack()
        # A bad argument shape is a ValueError regardless of engine state;
        # only a valid seed with no forward in flight is a RuntimeError.
        with self.assertRaises(ValueError):
            seq.backward((1.0, 2.0))
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)
        with self.assertRaises(RuntimeError):
            seq.backward((1.0,))

    def test_retry_after_failed_backward_is_not_a_second_call(self):
        class _Flaky:
            calls = 0

            def __init__(self):
                self.w = Tensor([1.0, 2.0])

            def parameters(self):
                return [self.w]

            def forward(self, x, hidden):
                y = Tensor([[3.0, 4.0]])
                return y, y

            def backward(self, upstream):
                type(self).calls += 1
                if type(self).calls == 1:
                    raise RuntimeError("simulated mid-pass failure")
                self.w.grad = Tensor([9.0, 9.0])
                return Tensor([[1.0, 1.0]])

        layer = _Flaky()
        seq = Sequential([layer])
        seq.forward(Tensor(_SEG_A))
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)
        # The aborted pass did not consume the forward; the retry succeeds.
        seq.backward((1.0,))
        self.assertEqual(layer.w.grad.tolist(), [9.0, 9.0])
        # Once it completes, the forward is consumed.
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)


class CheckpointRoundTripTests(unittest.TestCase):
    def test_memory_and_disk_resume_match_uninterrupted_bitwise(self):
        ref_out, ref_snapshot = _run_reference()
        boundary = _checkpoint_at_segment_boundary()

        buffer = boundary.save()
        self.assertIsInstance(buffer, bytes)
        mem_out, mem_snapshot = _run_resumed(buffer)
        self.assertEqual(mem_out, ref_out)
        self.assertEqual(mem_snapshot, ref_snapshot)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.bin")
            self.assertIsNone(boundary.save(path))
            disk_out, disk_snapshot = _run_resumed(path)
        self.assertEqual(disk_out, ref_out)
        self.assertEqual(disk_snapshot, ref_snapshot)
        self.assertEqual(disk_snapshot, mem_snapshot)

    def test_load_returns_boundary_hidden_slots(self):
        boundary = _checkpoint_at_segment_boundary()
        seq, _, _ = make_stack()
        hidden = seq.load(boundary.save())
        self.assertEqual(len(hidden), 2)
        self.assertEqual(hidden[0].tolist(), [[2.0, 3.0]])
        self.assertEqual(hidden[1].tolist(), [[8.0, 15.0]])

    def test_repeated_saves_are_identical_and_do_not_mutate_state(self):
        seq = _checkpoint_at_segment_boundary()
        before = _snapshot(seq)
        first = seq.save()
        second = seq.save()
        third = seq.save()
        self.assertEqual(first, second)
        self.assertEqual(second, third)
        self.assertEqual(_snapshot(seq), before)

    def test_save_load_save_is_byte_identical(self):
        seq = _checkpoint_at_segment_boundary()
        blob = seq.save()
        restored, _, _ = make_stack()
        restored.load(blob)
        self.assertEqual(restored.save(), blob)

    def test_float64_round_trips_bitwise_including_negative_zero(self):
        special = [
            -0.0,
            0.0,
            1.5,
            -1.5,
            5e-324,
            -5e-324,
            1.7976931348623157e308,
            -2.2250738585072014e-308,
        ]
        source = Sequential([_MultiScale([len(special)])])
        source.parameters()[0]._replace_data(special, [len(special)])
        # Forward an all-ones batch so the saved hidden slots stay finite;
        # only the parameter values probe float64 round-tripping.
        source.forward(Tensor([[1.0] * len(special)]))
        restored = Sequential([_MultiScale([len(special)])])
        restored.load(source.save())
        values = restored.parameters()[0].tolist()
        for expected, actual in zip(special, values):
            self.assertEqual(struct.pack(">d", actual), struct.pack(">d", expected))
        self.assertEqual(
            struct.pack(">d", values[0]), struct.pack(">d", -0.0)
        )
        self.assertEqual(
            struct.pack(">d", values[1]), struct.pack(">d", 0.0)
        )

    def test_parameters_without_gradient_are_checkpointed_as_zeros(self):
        seq, _, _ = make_stack()
        seq.forward(Tensor(_SEG_A))
        self.assertTrue(all(param.grad is None for param in seq.parameters()))
        restored, _, _ = make_stack()
        restored.load(seq.save())
        for param in restored.parameters():
            self.assertIsNotNone(param.grad)
            values = param.grad.tolist()
            self.assertTrue(all(value == 0.0 for value in values))

    def test_save_overwrites_existing_file_without_backup_or_tmp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ck.bin")
            seq = _checkpoint_at_segment_boundary()
            seq.save(path)
            with open(path, "rb") as handle:
                first_bytes = handle.read()
            # A simulated crashed write leaves a sibling temp file.
            decoy = os.path.join(directory, ".ck.bin.tmp.999.1")
            with open(decoy, "wb") as handle:
                handle.write(b"partial")
            seq.forward(Tensor(_SEG_B), seq.load(first_bytes))
            seq.backward(1.0)
            seq.save(path)
            entries = sorted(os.listdir(directory))
            self.assertEqual(entries, ["ck.bin"])
            with open(path, "rb") as handle:
                second_bytes = handle.read()
            self.assertNotEqual(second_bytes, first_bytes)

    def test_chained_resume_over_three_segments(self):
        # Uninterrupted reference over three segments, accumulating grads.
        ref, _, _ = make_stack()
        _, hidden = ref.forward(Tensor(_SEG_A))
        ref.backward(1.0)
        _, hidden = ref.forward(Tensor(_SEG_B), hidden)
        ref.backward(1.0)
        out3, hidden3 = ref.forward(Tensor(_SEG_A), hidden)
        ref.backward(1.0)
        ref.update(_LR)
        ref_state = _snapshot(ref)

        # Process 1: segment 1, checkpoint; process 2: segment 2, checkpoint
        # again; process 3: segment 3, backward, update.
        p1, _, _ = make_stack()
        _, h1 = p1.forward(Tensor(_SEG_A))
        p1.backward(1.0)
        blob1 = p1.save()

        p2, _, _ = make_stack()
        carried = p2.load(blob1)
        _, h2 = p2.forward(Tensor(_SEG_B), carried)
        p2.backward(1.0)
        blob2 = p2.save()

        p3, _, _ = make_stack()
        carried = p3.load(blob2)
        out, h3 = p3.forward(Tensor(_SEG_A), carried)
        p3.backward(1.0)
        p3.update(_LR)
        self.assertEqual([slot.tolist() for slot in h3],
                         [slot.tolist() for slot in hidden3])
        self.assertEqual(out.tolist(), out3.tolist())
        self.assertEqual(_snapshot(p3), ref_state)

    def test_save_target_validation(self):
        seq, _, _ = make_stack()
        with self.assertRaises(ValueError):
            seq.save(123)
        with self.assertRaises(ValueError):
            seq.load(123)


class CheckpointRejectionTests(unittest.TestCase):
    def setUp(self):
        seq, _, _ = make_stack()
        seq.forward(Tensor(_SEG_A))
        seq.backward(1.0)
        self.blob = seq.save()

    def _reload(self, blob):
        fresh, _, _ = make_stack()
        fresh.load(blob)

    def test_truncated_files_rejected(self):
        for cut in (0, 1, 8, 10, 18, 49, 50, 51,
                    len(self.blob) - 9, len(self.blob) - 1):
            with self.assertRaises(ValueError):
                self._reload(self.blob[:cut])

    def test_trailing_garbage_rejected(self):
        with self.assertRaises(ValueError):
            self._reload(self.blob + b"\x00")
        with self.assertRaises(ValueError):
            self._reload(b"")
        with self.assertRaises(ValueError):
            self._reload(b"this is not a checkpoint")

    def test_corrupted_bytes_rejected(self):
        bad = bytearray(self.blob)
        bad[0] ^= 0xFF
        with self.assertRaises(ValueError):
            self._reload(bytes(bad))
        bad = bytearray(self.blob)
        bad[60] ^= 0x01
        with self.assertRaises(ValueError):
            self._reload(bytes(bad))
        bad = bytearray(self.blob)
        bad[-1] ^= 0x01
        with self.assertRaises(ValueError):
            self._reload(bytes(bad))

    def test_envelope_version_mismatch_rejected(self):
        bad = bytearray(self.blob)
        bad[8], bad[9] = 0, 2
        with self.assertRaises(ValueError):
            self._reload(bytes(bad))

    def test_half_written_target_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ck.bin")
            with open(path, "wb") as handle:
                handle.write(self.blob[: len(self.blob) // 2])
            with self.assertRaises(ValueError):
                self._reload(path)

    def test_document_field_mismatches_rejected(self):
        document = _checkpoint.decode(self.blob)

        def reencode(doc):
            return _checkpoint.encode(doc)

        wrong_version = dict(document)
        wrong_version["ver"] = 99
        with self.assertRaises(ValueError):
            self._reload(reencode(wrong_version))

        missing = {key: value for key, value in document.items() if key != "hshapes"}
        with self.assertRaises(ValueError):
            self._reload(reencode(missing))

        extra = dict(document)
        extra["surprise"] = 1
        with self.assertRaises(ValueError):
            self._reload(reencode(extra))

        ragged = {
            "ver": document["ver"],
            "hshapes": [list(shape) for shape in document["hshapes"]],
            "layers": [
                {
                    "params": [
                        {
                            "shape": list(param["shape"]),
                            "data": param["data"],
                            "grad": param["grad"],
                        }
                        for param in layer["params"]
                    ],
                    "hidden": (
                        {
                            "shape": list(layer["hidden"]["shape"]),
                            "data": layer["hidden"]["data"],
                        }
                        if layer["hidden"] is not None
                        else None
                    ),
                }
                for layer in document["layers"]
            ],
        }
        ragged["layers"][0]["params"][0]["data"].pop()
        with self.assertRaises(ValueError):
            self._reload(reencode(ragged))

    def test_model_shape_and_layer_order_mismatch_rejected(self):
        # Different layer width: first layer is 3-wide in the current model.
        other = Sequential(
            [RecurrentScale([2.0, 3.0, 4.0]), RecurrentScale([4.0, 5.0])]
        )
        with self.assertRaises(ValueError):
            other.load(self.blob)

        # Layer order swapped: the first layer has 2 parameters in the
        # checkpoint but 1 in the model it is loaded into.
        source = Sequential([_MultiScale([2, 4]), _MultiScale([3])])
        swapped = Sequential([_MultiScale([3]), _MultiScale([2, 4])])
        with self.assertRaises(ValueError):
            swapped.load(source.save())

        # Different number of layers.
        with self.assertRaises(ValueError):
            Sequential([RecurrentScale([2.0, 3.0])]).load(self.blob)

    def test_missing_path_raises_file_not_found(self):
        fresh, _, _ = make_stack()
        with self.assertRaises(FileNotFoundError):
            fresh.load("/nonexistent/path/to/checkpoint.bin")

    def test_rejected_load_does_not_mutate_current_model(self):
        target, _, _ = make_stack()
        target.forward(Tensor(_SEG_A))
        target.backward(1.0)
        blob = target.save()

        current, first, second = make_stack()
        # Mutate the current model so any partial restore would be visible.
        current.forward(Tensor(_SEG_B))
        current.backward(2.0)
        params_before = _snapshot(current)

        # Wrong shape -> wholesale rejection.
        widened = Sequential(
            [RecurrentScale([2.0, 3.0, 4.0]), RecurrentScale([4.0, 5.0])]
        )
        with self.assertRaises(ValueError):
            widened.load(blob)
        # Corrupt bytes -> wholesale rejection, current untouched.
        flipped = bytearray(blob)
        flipped[60] ^= 0x01
        with self.assertRaises(ValueError):
            current.load(bytes(flipped))
        self.assertEqual(_snapshot(current), params_before)
        self.assertEqual(first.w.tolist(), [2.0, 3.0])
        self.assertEqual(second.w.tolist(), [4.0, 5.0])

    def test_unwritable_target_raises_oserror(self):
        seq, _, _ = make_stack()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OSError):
                seq.save(os.path.join(directory, "missing-dir", "ck.bin"))

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root bypasses chmod")
    def test_read_only_directory_raises_oserror(self):
        seq, _, _ = make_stack()
        with tempfile.TemporaryDirectory() as directory:
            os.chmod(directory, 0o555)
            try:
                with self.assertRaises(OSError):
                    seq.save(os.path.join(directory, "ck.bin"))
            finally:
                os.chmod(directory, 0o755)

    def test_non_finite_and_oversized_state_rejected_at_save(self):
        seq, first, _ = make_stack()
        for bad_values, label in (
            ([float("nan"), 3.0], "nan parameter"),
            ([float("inf"), 3.0], "infinite parameter"),
            ([-(1 << 70), 3], "oversized integer parameter"),
        ):
            first.w._replace_data(bad_values, [2])
            with self.assertRaises(ValueError, msg=label):
                seq.save()
        first.w._replace_data([2.0, 3.0], [2])
        first.w.grad = Tensor([float("-inf"), 0.0])
        with self.assertRaises(ValueError):
            seq.save()


if __name__ == "__main__":
    unittest.main()
