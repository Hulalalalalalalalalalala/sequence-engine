import errno
import json
import os
import struct
import tempfile
import unittest
import zlib
from unittest import mock

from sequence_engine import Sequential, Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _RNNStep,
    _base_weights,
    _build,
    _SEG1,
    _SEG2,
    _total,
)


def _stack(weights=None):
    return _build(weights if weights is not None else _base_weights())


def _zeros_shape(shape):
    if not shape:
        return 0.0
    return [_zeros_shape(shape[1:]) for _ in range(shape[0])]


def _zero_optim(seq):
    return {
        "t": 0,
        "m": [
            {"s": p.shape, "v": _zeros_shape(p.shape)} for p in seq.parameters()
        ],
        "v": [
            {"s": p.shape, "v": _zeros_shape(p.shape)} for p in seq.parameters()
        ],
    }


class UpdateTests(unittest.TestCase):
    def test_step_is_theta_minus_lr_times_grad_in_place(self):
        seq, layers = _stack()
        seq.forward(Tensor(_SEG1))
        seq.backward(1.0)
        before = [p.tolist() for p in seq.parameters()]
        ids = {id(p) for p in seq.parameters()}
        lr = 0.375
        seq.update(lr)
        self.assertEqual({id(p) for p in seq.parameters()}, ids)  # in place

        def sub(values, grads):
            if isinstance(values, list):
                return [sub(a, b) for a, b in zip(values, grads)]
            return values - lr * grads

        for param, old in zip(seq.parameters(), before):
            self.assertEqual(param.tolist(), sub(old, param.grad.tolist()))

    def test_update_does_not_clear_gradients(self):
        seq, _ = _stack()
        seq.forward(Tensor(_SEG1))
        seq.backward(1.0)
        grads = [p.grad.tolist() for p in seq.parameters()]
        seq.update(0.1)
        self.assertEqual([p.grad.tolist() for p in seq.parameters()], grads)

    def test_parameters_without_gradients_are_left_untouched(self):
        seq, _ = _stack()
        before = [p.tolist() for p in seq.parameters()]
        seq.update(0.1)
        self.assertEqual([p.tolist() for p in seq.parameters()], before)

    def test_non_finite_or_non_numeric_lr_rejected(self):
        seq, _ = _stack()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                seq.update(bad)
        with self.assertRaises(ValueError):
            seq.update("0.1")
        with self.assertRaises(ValueError):
            seq.update(True)


class TupleLossAndRetryTests(unittest.TestCase):
    def test_tuple_loss_matches_scalar_seed(self):
        seq, _ = _stack()
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward((out, 2.0))
        ref, _ = _stack()
        ref.forward(Tensor(_SEG1))
        ref.backward(2.0)
        self.assertEqual(
            [p.grad.tolist() for p in seq.parameters()],
            [p.grad.tolist() for p in ref.parameters()],
        )

    def test_tuple_loss_requires_the_forward_output_tensor(self):
        seq, _ = _stack()
        seq.forward(Tensor(_SEG1))
        other, _ = _stack()
        other_out, _ = other.forward(Tensor(_SEG1))
        with self.assertRaises(ValueError):
            seq.backward((other_out, 1.0))
        with self.assertRaises(ValueError):
            seq.backward((other_out,))
        with self.assertRaises(ValueError):
            seq.backward((other_out, float("nan")))

    def test_backward_without_forward_is_runtime_error(self):
        seq, _ = _stack()
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)

    def test_double_backward_is_runtime_error(self):
        seq, _ = _stack()
        seq.forward(Tensor(_SEG1))
        seq.backward(1.0)
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)

    def test_retried_backward_after_failure_is_not_a_second_backward(self):
        class BoomOnce(_RNNStep):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._boom = True

            def backward(self, upstream):
                if self._boom:
                    self._boom = False
                    raise RuntimeError("transient")
                return super().backward(upstream)

        weights = _base_weights()
        boom = BoomOnce(2, 3, weights["wxh1"], weights["whh1"], weights["b1"])
        quiet = _RNNStep(3, 2, weights["wxh2"], weights["whh2"], weights["b2"])
        seq = Sequential([boom, quiet])
        seq.forward(Tensor(_SEG1))
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)
        # reverse-order walk touched the quiet layer first; partial grads
        # must have been rolled back exactly
        self.assertIsNone(quiet.wxh.grad)
        seq.backward(1.0)  # retry allowed
        ref, _ = _stack(weights)
        ref.forward(Tensor(_SEG1))
        ref.backward(1.0)
        self.assertEqual(
            [p.grad.tolist() for p in seq.parameters()],
            [p.grad.tolist() for p in ref.parameters()],
        )
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)  # the successful retry consumed the pass


class CheckpointRoundTripTests(unittest.TestCase):
    def _trained_state(self):
        seq, _ = _stack()
        out1, h1 = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out1))
        return seq, h1

    def test_memory_and_path_round_trip_bytes_are_identical(self):
        seq, h1 = self._trained_state()
        buf = bytearray()
        seq.save(buf)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.ckp")
            seq.save(path)
            with open(path, "rb") as fh:
                disk_bytes = fh.read()
        self.assertEqual(disk_bytes, bytes(buf))

        mem_model, _ = _stack()
        hidden_mem = mem_model.load(bytes(buf))
        disk_model, _ = _stack()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.ckp")
            seq.save(path)
            hidden_disk = disk_model.load(path)
        self.assertEqual(
            [s.tolist() for s in hidden_mem], [s.tolist() for s in h1]
        )
        self.assertEqual(
            [s.tolist() for s in hidden_disk], [s.tolist() for s in h1]
        )
        self.assertEqual(
            [p.tolist() for p in mem_model.parameters()],
            [p.tolist() for p in seq.parameters()],
        )
        self.assertEqual(
            [p.grad.tolist() for p in disk_model.parameters()],
            [p.grad.tolist() for p in seq.parameters()],
        )

    def test_bitwise_continuity_across_interruption(self):
        lr = 0.1
        seg3 = [[0.3, -0.7], [-0.2, 0.4]]

        def run(checkpoint_target=None):
            seq, _ = _stack()
            out1, hidden = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out1))
            if checkpoint_target is not None:
                seq.save(checkpoint_target)
                seq, _ = _stack()
                hidden = seq.load(checkpoint_target)
            out2, hidden2 = seq.forward(Tensor(_SEG2), hidden)
            seq.backward(_total(out2))
            seq.update(lr)
            carry = [Tensor(s.tolist()) for s in hidden2]
            out3, _ = seq.forward(Tensor(seg3), carry)
            seq.backward(_total(out3))
            return (
                [p.tolist() for p in seq.parameters()],
                [p.grad.tolist() for p in seq.parameters()],
            )

        expected = run()
        mem = bytearray()
        self.assertEqual(run(mem), expected)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.ckp")
            self.assertEqual(run(path), expected)

    def test_repeated_saves_are_identical_and_change_nothing(self):
        seq, _ = self._trained_state()
        buf = bytearray()
        seq.save(buf)
        first = bytes(buf)
        params = [p.tolist() for p in seq.parameters()]
        grads = [p.grad.tolist() for p in seq.parameters()]
        for _ in range(3):
            seq.save(buf)
        self.assertEqual(bytes(buf), first)
        self.assertEqual([p.tolist() for p in seq.parameters()], params)
        self.assertEqual([p.grad.tolist() for p in seq.parameters()], grads)

    def test_missing_gradients_round_trip_as_explicit_zeros(self):
        seq, _ = _stack()
        buf = bytearray()
        seq.save(buf)
        restored, _ = _stack()
        self.assertIsNone(restored.load(buf))
        for param in restored.parameters():
            self.assertIsNotNone(param.grad)
            leaves = []

            def flat(v):
                if isinstance(v, list):
                    for item in v:
                        flat(item)
                else:
                    leaves.append(v)

            flat(param.grad.tolist())
            self.assertTrue(leaves)
            self.assertTrue(all(v == 0 for v in leaves))

    def test_negative_zero_and_float_bits_round_trip(self):
        seq, _ = _stack()
        param = seq.parameters()[1]  # whh1 shape [3, 3]
        param._set_values(
            [[1.5, -0.0, 3.25], [1e15, 2.0 ** -50, 0.0], [-1e-300, 7.0, -2.5]]
        )
        seq.zero_grad()
        param.grad._set_values(
            [[-0.0, 0.0, 1.0], [2.0, -0.0, 3.0], [4.0, 5.0, -0.0]]
        )
        buf = bytearray()
        seq.save(buf)
        restored, _ = _stack()
        restored.load(bytes(buf))
        out_values = restored.parameters()[1].tolist()
        out_grads = restored.parameters()[1].grad.tolist()
        self.assertEqual(struct.pack("<d", out_values[0][1]), struct.pack("<d", -0.0))
        self.assertEqual(struct.pack("<d", out_grads[0][0]), struct.pack("<d", -0.0))
        self.assertEqual(struct.pack("<d", out_grads[1][1]), struct.pack("<d", -0.0))
        self.assertEqual(out_values, param.tolist())
        self.assertEqual(out_grads, param.grad.tolist())

    def test_overwrite_replaces_without_backup_or_temp_leftovers(self):
        seq, _ = self._trained_state()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.ckp")
            seq.save(path)
            with open(path, "rb") as fh:
                first = fh.read()
            seq.zero_grad()
            seq.save(path)  # different contents, direct overwrite
            with open(path, "rb") as fh:
                second = fh.read()
            self.assertNotEqual(first, second)
            self.assertEqual(cp.load_bytes(path)["grads"][0]["v"][0][0], 0.0)
            leftovers = [
                f
                for f in os.listdir(td)
                if f.startswith(".seqckp.tmp-") or f.endswith(".bak")
            ]
            self.assertEqual(leftovers, [])


class CheckpointRejectionTests(unittest.TestCase):
    def setUp(self):
        seq, _ = _stack()
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        self.buf = bytearray()
        seq.save(self.buf)

    def _load(self, raw):
        victim, _ = _stack()
        victim.load(raw)

    def test_path_not_found(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                self._load(os.path.join(td, "missing.ckp"))

    def test_missing_directory_on_save_is_oserror(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "no-such-dir", "state.ckp")
            with self.assertRaises(OSError):
                seq.save(path)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root bypasses perms")
    def test_unwritable_directory_is_oserror(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as td:
            os.chmod(td, 0o555)
            try:
                with self.assertRaises(OSError):
                    seq.save(os.path.join(td, "state.ckp"))
            finally:
                os.chmod(td, 0o755)

    def test_disk_full_propagates_as_oserror_and_leaves_nothing(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.ckp")
            seq.save(path)
            with open(path, "rb") as fh:
                good = fh.read()

            # ENOSPC while finishing the temp file (delayed allocation
            # surfaces space failures at fsync time on real filesystems)
            write_full = OSError(errno.ENOSPC, "simulated disk full on flush")

            def raise_enospc(fd):
                raise write_full

            with mock.patch("os.fsync", side_effect=raise_enospc):
                with self.assertRaises(OSError) as ctx:
                    seq.save(path)
                self.assertEqual(ctx.exception.errno, errno.ENOSPC)
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), good)
            self.assertEqual(
                [f for f in os.listdir(td) if f.startswith(".seqckp.tmp-")], []
            )

            # ENOSPC at the atomic commit
            replace_full = OSError(errno.ENOSPC, "simulated disk full on commit")
            with mock.patch("os.replace", side_effect=replace_full):
                with self.assertRaises(OSError) as ctx:
                    seq.save(path)
                self.assertEqual(ctx.exception.errno, errno.ENOSPC)
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), good)
            self.assertEqual(
                [f for f in os.listdir(td) if f.startswith(".seqckp.tmp-")], []
            )

    def test_torn_and_foreign_buffers_rejected(self):
        raw = bytes(self.buf)
        for bad in (
            b"",
            b"definitely not a checkpoint",
            raw[: len(raw) // 2],
            raw[:-1],
            raw + b" ",
        ):
            with self.subTest(bad=bad[:12]):
                with self.assertRaises(ValueError):
                    self._load(bad)

    def test_torn_file_at_final_path_rejected(self):
        raw = bytes(self.buf)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.ckp")
            with open(path, "wb") as fh:
                fh.write(raw[: len(raw) // 2])  # interrupted write simulation
            with self.assertRaises(ValueError):
                self._load(path)

    def test_crc_corruption_rejected(self):
        bad = bytearray(self.buf)
        bad[len(bad) // 2] ^= 0xFF
        with self.assertRaises(ValueError):
            self._load(bytes(bad))

    def test_version_mismatch_rejected(self):
        raw = bytes(self.buf)
        forged = raw[:8] + struct.pack("<I", 999) + raw[12:]
        with self.assertRaises(ValueError):
            self._load(forged)

    def test_parameter_shape_mismatch_rejected_wholesale(self):
        other_weights = _base_weights()
        other_weights["b1"] = [0.01, -0.02]  # shape [2] instead of [3]
        different, _ = _stack(other_weights)
        other_buf = bytearray()
        different.save(other_buf)
        victim, _ = _stack()
        params_before = [p.tolist() for p in victim.parameters()]
        with self.assertRaises(ValueError):
            victim.load(bytes(other_buf))
        # whole-document rejection: nothing applied
        self.assertEqual([p.tolist() for p in victim.parameters()], params_before)
        self.assertTrue(all(p.grad is None for p in victim.parameters()))

    def test_layer_order_mismatch_rejected(self):
        seq, _ = _stack()
        seq.zero_grad()
        doc = {
            "params": [{"s": p.shape, "v": p.tolist()} for p in seq.parameters()],
            "grads": [
                {"s": p.shape, "v": p.grad.tolist()} for p in seq.parameters()
            ],
            "optim": _zero_optim(seq),
            "hidden": None,
            "layers": [
                {"kind": "WrongLayer", "shapes": [[2, 3], [3, 3], [3]]},
                {"kind": "_RNNStep", "shapes": [[3, 2], [2, 2], [2]]},
            ],
            "pending": False,
        }
        with self.assertRaises(ValueError):
            self._load(cp.build_bytes(doc))

    def test_layer_count_mismatch_rejected(self):
        seq, _ = _stack()
        seq.zero_grad()
        shapes = [p.shape for p in seq.parameters()]
        doc = {
            "params": [{"s": s, "v": seq.parameters()[i].tolist()}
                       for i, s in enumerate(shapes)],
            "grads": [
                {"s": p.shape, "v": p.grad.tolist()} for p in seq.parameters()
            ],
            "optim": _zero_optim(seq),
            "hidden": None,
            "layers": [
                {"kind": "_RNNStep", "shapes": shapes[:3]},
                {"kind": "_RNNStep", "shapes": shapes[3:-1]},
                {"kind": "_RNNStep", "shapes": [shapes[-1]]},
            ],
            "pending": False,
        }
        with self.assertRaises(ValueError):
            self._load(cp.build_bytes(doc))

    def test_forged_header_with_missing_field_rejected(self):
        def repack(mutate):
            raw = bytes(self.buf)
            (hlen,) = struct.unpack("<Q", raw[12:20])
            header = json.loads(raw[20 : 20 + hlen])
            payload = raw[20 + hlen : raw.rfind(cp.END_MAGIC)]
            mutate(header)
            new_header = json.dumps(
                header, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            leaf_count = len(payload) // 9
            body = (
                raw[:8]
                + struct.pack("<I", cp.FORMAT_VERSION)
                + struct.pack("<Q", len(new_header))
                + new_header
                + payload
            )
            crc = zlib.crc32(body[20:])
            return body + cp.END_MAGIC + struct.pack("<QI", leaf_count, crc)

        with self.assertRaises(ValueError):
            self._load(repack(lambda h: h.pop("layers")))
        with self.assertRaises(ValueError):
            self._load(repack(lambda h: h.pop("pending")))
        with self.assertRaises(ValueError):
            self._load(repack(lambda h: h.update(pending="yes")))

    def test_forged_leaf_count_mismatch_rejected(self):
        raw = bytes(self.buf)
        crc = struct.unpack("<I", raw[-4:])[0]
        forged = raw[:-12] + struct.pack("<QI", 1, crc)
        with self.assertRaises(ValueError):
            self._load(forged)

    def test_save_after_forward_without_backward_fixes_the_boundary(self):
        # The post-forward state IS a slice boundary (the returned hidden
        # state lives there), so saving is judged by the boundary, not by
        # whether backward has run, and is now allowed.
        seq, _ = _stack()
        out, hidden = seq.forward(Tensor(_SEG1))  # no backward yet
        buf = bytearray()
        seq.save(buf)  # must not raise
        self.assertTrue(len(bytes(buf)) > 0)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.ckp")
            seq.save(path)
            victim, _ = _stack()
            restored = victim.load(path)
        self.assertEqual(
            [s.tolist() for s in restored], [s.tolist() for s in hidden]
        )
        # The still-pending backward of the saving model is unaffected.
        seq.backward(_total(out))
        self.assertTrue(all(p.grad is not None for p in seq.parameters()))


class CheckpointValueGuardsTests(unittest.TestCase):
    def _doc(self, value):
        return {
            "params": [{"s": [1], "v": [value]}],
            "grads": [{"s": [1], "v": [0.0]}],
            "optim": {
                "t": 0,
                "m": [{"s": [1], "v": [0.0]}],
                "v": [{"s": [1], "v": [0.0]}],
            },
            "hidden": None,
            "layers": [{"kind": "L", "shapes": [[1]]}],
            "pending": False,
        }

    def test_huge_integers_rejected(self):
        for value in (2**63, -(2**63) - 1, 10**40):
            with self.assertRaises(ValueError):
                cp.build_bytes(self._doc(value))

    def test_non_finite_floats_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                cp.build_bytes(self._doc(value))

    def test_non_finite_state_via_model_save_rejected(self):
        seq, _ = _stack()
        seq.parameters()[0]._set_values([[float("inf"), 0.0, 0.0], [0.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            seq.save(bytearray())


if __name__ == "__main__":
    unittest.main()
