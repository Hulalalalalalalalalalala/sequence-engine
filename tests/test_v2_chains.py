"""Tests for format v2: v1 read compatibility/migration and incremental chains."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import threading
import zlib
import unittest

from sequence_engine import Sequential, Tensor
from sequence_engine import checkpoint as cp
from sequence_engine import _v1_golden
from sequence_engine._selftest import (
    _base_weights,
    _build,
    _SEG1,
    _total,
)


def _v1_trained():
    return _v1_golden.trained_bytes()


def _v1_tricky():
    return _v1_golden.tricky_bytes()


def _stack(weights=None):
    return _build(weights if weights is not None else _base_weights())


class Version1CompatTests(unittest.TestCase):
    def test_v1_loads_and_reports_source_version(self):
        seq, _ = _stack()
        out, h1 = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        victim, _ = _stack()
        restored_hidden = victim.load(_v1_trained())
        self.assertEqual(victim.loaded_from_version, 1)
        self.assertEqual(
            [s.tolist() for s in restored_hidden], [s.tolist() for s in h1]
        )
        self.assertEqual(
            [p.tolist() for p in victim.parameters()],
            [p.tolist() for p in seq.parameters()],
        )
        self.assertEqual(
            [p.grad.tolist() for p in victim.parameters()],
            [p.grad.tolist() for p in seq.parameters()],
        )

    def test_v1_native_float_fidelity_including_negative_zero(self):
        victim, _ = _stack()
        victim.load(_v1_tricky())
        values = victim.parameters()[0].tolist()
        grad_values = victim.parameters()[0].grad.tolist()
        self.assertEqual(
            values, [[1.5, -0.0, 3.25], [100000000000000.0, -2.5, 0.0625]]
        )
        self.assertEqual(
            grad_values, [[-0.0, 2.0, -3.0], [0.0, 1.0, -1.0]]
        )
        self.assertEqual(
            struct.pack("<d", values[0][1]), struct.pack("<d", -0.0)
        )
        self.assertEqual(
            struct.pack("<d", grad_values[0][0]), struct.pack("<d", -0.0)
        )

    def test_v1_continue_training_is_bitwise_continuous(self):
        seq, _ = _stack()
        out1, h1 = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out1))
        migrated, _ = _stack()
        migrated.load(_v1_trained())
        # Re-save: migrated state is written out natively as version 3.
        rebuf = bytearray()
        migrated.save(rebuf)
        self.assertEqual(struct.unpack("<I", rebuf[8:12])[0], cp.FORMAT_VERSION)
        again, _ = _stack()
        hidden2 = again.load(bytes(rebuf))
        self.assertEqual(again.loaded_from_version, 3)
        self.assertEqual(
            [s.tolist() for s in hidden2], [s.tolist() for s in h1]
        )

    def test_corrupt_or_forged_v1_is_rejected_wholesale(self):
        raw = _v1_trained()
        for bad in (raw[: len(raw) // 2], raw[:-1], raw + b" "):
            victim, _ = _stack()
            with self.assertRaises(ValueError):
                victim.load(bad)
            self.assertIsNone(victim.loaded_from_version)
            self.assertTrue(all(p.grad is None for p in victim.parameters()))
        flipped = bytearray(raw)
        flipped[len(flipped) // 2] ^= 0xFF
        victim, _ = _stack()
        with self.assertRaises(ValueError):
            victim.load(bytes(flipped))
        self.assertTrue(all(p.grad is None for p in victim.parameters()))

    def test_v1_forged_header_missing_field_rejected_mid_migration(self):
        raw = _v1_trained()
        (hlen,) = struct.unpack("<Q", raw[12:20])
        header = json.loads(raw[20 : 20 + hlen])
        payload = raw[20 + hlen : raw.rfind(cp.END_MAGIC)]
        header.pop("pending")
        new_header = json.dumps(
            header, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        leaf_count = len(payload) // 9
        body = (
            raw[:8]
            + struct.pack("<I", 1)
            + struct.pack("<Q", len(new_header))
            + new_header
            + payload
        )
        crc = zlib.crc32(body[20:])
        forged = body + cp.END_MAGIC + struct.pack("<QI", leaf_count, crc)
        victim, _ = _stack()
        with self.assertRaises(ValueError):
            victim.load(forged)
        self.assertIsNone(victim.loaded_from_version)

    def test_v1_payload_non_finite_smuggled_rejected(self):
        # Rewrite one float leaf's bits to +inf with a valid CRC: migration
        # must refuse the item rather than accept a non-finite value.
        raw = bytearray(_v1_tricky())
        hlen = struct.unpack("<Q", bytes(raw[12:20]))[0]
        payload_start = 20 + hlen
        end = bytes(raw).rfind(cp.END_MAGIC)
        payload = bytes(raw)[payload_start:end]
        pos = None
        for i in range(0, len(payload), 9):
            if payload[i] == ord("f"):
                pos = i
                break
        self.assertIsNotNone(pos)
        raw[payload_start + pos + 1 : payload_start + pos + 9] = struct.pack(
            "<d", float("inf")
        )
        crc = zlib.crc32(bytes(raw)[20:end])
        raw[-4:] = struct.pack("<I", crc)
        victim, _ = _stack()
        with self.assertRaises(ValueError):
            victim.load(bytes(raw))


class Version2HeaderTests(unittest.TestCase):
    def _forged(self, mutate):
        seq, _ = _stack()
        seq.zero_grad()
        buf = bytearray()
        seq.save(buf)
        raw = bytes(buf)
        hlen = struct.unpack("<Q", raw[12:20])[0]
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

    def test_added_or_removed_header_field_rejected(self):
        victim, _ = _stack()
        with self.assertRaises(ValueError):
            victim.load(self._forged(lambda h: h.pop("layers")))
        victim, _ = _stack()
        with self.assertRaises(ValueError):
            victim.load(self._forged(lambda h: h.pop("pending")))
        victim, _ = _stack()
        with self.assertRaises(ValueError):
            victim.load(self._forged(lambda h: h.update(surprise=1)))
        victim, _ = _stack()
        with self.assertRaises(ValueError):
            victim.load(self._forged(lambda h: h.update(v=1)))


def _seg_name(i):
    return f"seg-{i:010d}.seqd"


class ChainRoundTripTests(unittest.TestCase):
    def _trained(self):
        seq, _ = _stack()
        out, h = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        return seq, h

    def test_chain_reassembles_bitwise_equal_to_full_snapshot(self):
        seq, h1 = self._trained()
        full0 = bytearray()
        seq.save(full0)
        with tempfile.TemporaryDirectory() as td:
            seq.save(td)  # basis
            names0 = sorted(os.listdir(td))
            self.assertEqual(names0, sorted(["head", _seg_name(0)]))

            # No changes: an empty delta, still a complete chain.
            seq.save(td)
            seq.save(td)

            # A real update changes parameters; then a second delta.
            seq.update(0.1)
            full1 = bytearray()
            seq.save(full1)
            seq.save(td)

            doc = cp.load_chain(td)
            # A reassembled chain is a normal current-format full document:
            # it rebuilds to exactly the bytes of the matching full save.
            self.assertEqual(cp.build_bytes(doc), bytes(full1))

            with open(os.path.join(td, _seg_name(3)), "rb") as fh:
                seg_raw = fh.read()
            _n, _hc, _p, items = cp._parse_delta(seg_raw, 3)
            self.assertTrue(items)  # the update delta carries leaves

            # The two no-op deltas carry no tensors at all.
            for i in (1, 2):
                with open(os.path.join(td, _seg_name(i)), "rb") as fh:
                    raw = fh.read()
                _n, _hc, _p, items = cp._parse_delta(raw, i)
                self.assertEqual(items, [])

            # Loading through the directory entry point on Sequential.
            victim, _ = _stack()
            restored = victim.load(td)
            self.assertEqual(
                [p.tolist() for p in victim.parameters()],
                [p.tolist() for p in seq.parameters()],
            )
            # update() leaves the slice-boundary hidden state untouched.
            self.assertEqual(
                [s.tolist() for s in restored], [s.tolist() for s in h1]
            )

            # Reassemble an earlier point of the chain.
            doc0 = cp.load_chain(td, up_to=0)
            self.assertEqual(cp.build_bytes(doc0), bytes(full0))
            doc_noop = cp.load_chain(td, up_to=2)
            self.assertEqual(cp.build_bytes(doc_noop), bytes(full0))
            doc_head = cp.load_chain(td)
            self.assertEqual(cp.build_bytes(doc_head), bytes(full1))

    def test_chain_hidden_state_introduction_and_carry(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as td:
            seq.save(td)  # basis: no hidden yet
            out, h = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out))
            seq.save(td)  # delta 1 introduces hidden state, all slots
            doc = cp.load_chain(td)
            self.assertEqual(
                [e["v"] for e in doc["hidden"]], [s.tolist() for s in h]
            )
            full = bytearray()
            seq.save(full)
            self.assertEqual(cp.build_bytes(doc), bytes(full))

            victim, _ = _stack()
            restored = victim.load(td)
            self.assertEqual(
                [s.tolist() for s in restored], [s.tolist() for s in h]
            )

    def test_empty_parameter_layer_is_deterministic_in_chain(self):
        class PassThrough:
            checkpoint_kind = "PassThrough"

            def parameters(self):
                return []

            def forward(self, x, hidden):
                return x, Tensor([[0.0]])

            def backward(self, upstream):
                return upstream

        class Scale:
            checkpoint_kind = "Scale"

            def __init__(self):
                self.w = Tensor([1.0, 2.0])

            def parameters(self):
                return [self.w]

            def forward(self, x, hidden):
                xs = x.tolist()
                y = [[v * self.w.tolist()[j] for j, v in enumerate(row)]
                     for row in xs]
                self._cache = xs
                return Tensor(y), Tensor(y)

            def backward(self, upstream):
                xs = self._cache
                if isinstance(upstream, Tensor):
                    dy = upstream.tolist()
                else:
                    dy = [[float(upstream)] * 2 for _ in xs]
                g = [sum(dy[r][j] * xs[r][j] for r in range(len(xs)))
                     for j in range(2)]
                self.w.grad = Tensor(g)
                return Tensor(dy)

        seq = Sequential([Scale(), PassThrough(), Scale()])
        x = Tensor([[1.0, 1.0]])
        out, h = seq.forward(x)
        seq.backward(1.0)
        with tempfile.TemporaryDirectory() as td:
            seq.save(td)
            seq.save(td)  # unchanged: delta must deterministically be empty
            with open(os.path.join(td, _seg_name(1)), "rb") as fh:
                raw = fh.read()
            _n, hc, _p, items = cp._parse_delta(raw, 1)
            self.assertEqual(items, [])
            doc = cp.load_chain(td)
            victim = Sequential([Scale(), PassThrough(), Scale()])
            restored = victim.load(td)
            self.assertEqual(
                [p.tolist() for p in victim.parameters()],
                [p.tolist() for p in seq.parameters()],
            )
            self.assertEqual(len(restored), 3)


class ChainRejectionTests(unittest.TestCase):
    def test_missing_directory_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                cp.load_chain(os.path.join(td, "no-chain"))
            victim, _ = _stack()
            with self.assertRaises(FileNotFoundError):
                victim.load(os.path.join(td, "no-chain"))

    def test_save_into_missing_chain_dir_is_oserror(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as td:
            # A full-file save inside a missing parent dir is an OSError.
            with self.assertRaises(OSError):
                seq.save(os.path.join(td, "no-such-dir", "state.ckp"))
            # An explicit chain append into a missing directory is too.
            with self.assertRaises(OSError):
                cp.save_chain(
                    cp.load_bytes(self._full_bytes(seq)),
                    os.path.join(td, "no-chain-dir"),
                )

    @staticmethod
    def _full_bytes(seq):
        buf = bytearray()
        seq.save(buf)
        return bytes(buf)

    def test_missing_referenced_segment_is_filenotfound(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as td:
            seq.save(td)
            seq.update(0.1)
            seq.save(td)
            os.unlink(os.path.join(td, _seg_name(0)))
            with self.assertRaises(FileNotFoundError):
                cp.load_chain(td)
            os.unlink(os.path.join(td, _seg_name(1)))
            with self.assertRaises(FileNotFoundError):
                cp.load_chain(td)

    def test_truncated_or_corrupt_segment_rejected(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as td:
            seq.save(td)
            seq.update(0.1)
            seq.save(td)
            path1 = os.path.join(td, _seg_name(1))
            with open(path1, "rb") as fh:
                good = fh.read()
            for bad in (good[: len(good) // 2], good[:-1]):
                with open(path1, "wb") as fh:
                    fh.write(bad)
                with self.assertRaises(ValueError):
                    cp.load_chain(td)
            with open(path1, "wb") as fh:
                fh.write(good)
            flipped = bytearray(good)
            flipped[len(flipped) // 2] ^= 0xFF
            with open(path1, "wb") as fh:
                fh.write(bytes(flipped))
            with self.assertRaises(ValueError):
                cp.load_chain(td)
            with open(path1, "wb") as fh:
                fh.write(good)
            # corrupt head pointer
            with open(os.path.join(td, "head"), "wb") as fh:
                fh.write(b"not-a-number")
            with self.assertRaises(ValueError):
                cp.load_chain(td)

    def test_delta_missing_field_rejected(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as td:
            seq.save(td)
            seq.update(0.1)
            seq.save(td)
            seg_path = os.path.join(td, _seg_name(1))
            with open(seg_path, "rb") as fh:
                raw = fh.read()
            hlen = struct.unpack("<Q", raw[12:20])[0]
            header = json.loads(raw[20 : 20 + hlen])
            payload = raw[20 + hlen : raw.rfind(cp.DELTA_END_MAGIC)]
            header.pop("changed")
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
            forged = body + cp.DELTA_END_MAGIC + struct.pack(
                "<QI", leaf_count, crc
            )
            with open(os.path.join(td, _seg_name(1)), "wb") as fh:
                fh.write(forged)
            with self.assertRaises(ValueError):
                cp.load_chain(td)

    def test_delta_with_huge_int_rejected_before_segment_write(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as td:
            seq.save(td)
            with open(os.path.join(td, "head"), "rb") as fh:
                head_before = fh.read()
            # A Python integer beyond int64 smuggled into a changed tensor
            # is refused while the delta leaves are frozen; no segment is
            # committed and the chain stays on its previous head.
            seq.parameters()[0]._set_values(
                [[10**40, 1.0, 1.0], [1.0, 1.0, 1.0]]
            )
            with self.assertRaises(ValueError):
                seq.save(td)
            with open(os.path.join(td, "head"), "rb") as fh:
                self.assertEqual(fh.read(), head_before)
            # Chain still reassembles to the pre-rejection state.
            victim, _ = _stack()
            victim.load(td)  # must not raise

    def test_delta_payload_nonfinite_or_bad_tag_rejected(self):
        seq, _ = _stack()
        seq.zero_grad()
        with tempfile.TemporaryDirectory() as td:
            seq.save(td)
            seq.parameters()[0].grad._set_values(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            )
            seq.save(td)
            path1 = os.path.join(td, _seg_name(1))
            with open(path1, "rb") as fh:
                good = fh.read()

            def load_forged(mutate, tag_value):
                raw = bytearray(good)
                hlen = struct.unpack("<Q", bytes(raw[12:20]))[0]
                end = bytes(raw).rfind(cp.DELTA_END_MAGIC)
                pstart = 20 + hlen
                mutate(raw, pstart)
                crc = zlib.crc32(bytes(raw)[20:end])
                raw[-4:] = struct.pack("<I", crc)
                with open(path1, "wb") as fh:
                    fh.write(bytes(raw))
                with self.assertRaises(ValueError):
                    cp.load_chain(td)

            # NaN bit pattern in an otherwise valid float leaf.
            load_forged(
                lambda raw, p: raw.__setitem__(
                    slice(p + 1, p + 9), struct.pack("<d", float("nan"))
                ),
                float("nan"),
            )
            # +inf bit pattern in a float leaf.
            load_forged(
                lambda raw, p: raw.__setitem__(
                    slice(p + 1, p + 9), struct.pack("<d", float("inf"))
                ),
                float("inf"),
            )
            # An unknown leaf tag byte.
            load_forged(lambda raw, p: raw.__setitem__(p, ord("z")), ord("z"))
            # Restore: the chain loads cleanly again.
            with open(path1, "wb") as fh:
                fh.write(good)
            cp.load_chain(td)

    def test_no_head_pointer_is_valueerror(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "junk"), "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(ValueError):
                cp.load_chain(td)


class ConcurrencyTests(unittest.TestCase):
    class _IntLayer:
        """Pure-integer arithmetic so update states sit on an exact lattice.

        After each forward/backward the gradient equals the constant g;
        update(1) moves every parameter by exactly -g.  A model state that
        is one serial checkpoint must satisfy value == theta0 - k*g for a
        single integer k shared by every parameter.
        """

        checkpoint_kind = "IntLayer"

        def __init__(self, w, g):
            self.w = Tensor(w)
            self._g = g
            self._cache = None

        def parameters(self):
            return [self.w]

        def forward(self, x, hidden):
            xs = x.tolist()
            y = [[xs[r][j] + self.w.tolist()[j] for j in range(len(xs[0]))]
                 for r in range(len(xs))]
            self._cache = xs
            return Tensor(y), Tensor(y)

        def backward(self, upstream):
            dy = upstream.tolist() if isinstance(upstream, Tensor) else [[upstream, upstream]]
            g = self._g
            self.w.grad = Tensor(list(g))
            return Tensor(dy)

    def _model(self):
        a = self._IntLayer([1000, 2000], [3, 7])
        b = self._IntLayer([5000, 9000], [1, 2])
        return Sequential([a, b]), a, b

    def _assert_lattice(self, param_values, base, grad):
        # Coherence invariant: every element equals base[j] - k * g[j] for
        # ONE shared, non-negative integer k. A torn update, or a load
        # applied halfway, would leave different elements at different k.
        # update() coerces lr to float but the magnitudes here stay exact
        # integers in float64.
        steps = None
        for values, w0, g in zip(param_values, base, grad):
            for j, value in enumerate(values):
                current = int(value)
                self.assertEqual(value, float(current))
                delta = w0[j] - current
                self.assertEqual(delta % g[j], 0, "weight moved off its step lattice")
                self.assertGreaterEqual(delta // g[j], 0)
                k = delta // g[j]
                if steps is None:
                    steps = k
                self.assertEqual(k, steps, "torn update/load: parameters disagree")

    def _primed_model(self):
        # One clean segment establishes a constant gradient for the race.
        seq, a, b = self._model()
        seq.forward(Tensor([[1, 1]]))
        seq.backward(1)
        return seq, a, b

    def test_concurrent_updates_and_snapshots_are_always_coherent(self):
        seq, a, b = self._primed_model()
        base = [p.tolist() for p in seq.parameters()]
        grad = [a._g, b._g]
        errors = []

        def updater():
            try:
                for _ in range(200):
                    seq.update(1)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def check_probe(probe):
            self._assert_lattice(
                [p.tolist() for p in probe.parameters()], base, grad
            )

        def buffer_saver():
            try:
                for _ in range(200):
                    buf = bytearray()
                    seq.save(buf)
                    probe, _, _ = self._model()
                    probe.load(bytes(buf))
                    check_probe(probe)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def chain_saver():
            chain = cp.MemoryChain()
            try:
                for _ in range(200):
                    seq.save(chain)
                    probe, _, _ = self._model()
                    probe.load(chain)
                    check_probe(probe)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.ckp")

            def path_saver():
                try:
                    for _ in range(200):
                        seq.save(path)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def path_reader():
                try:
                    for _ in range(200):
                        if os.path.exists(path):
                            probe, _, _ = self._model()
                            probe.load(path)  # must never observe a torn file
                            check_probe(probe)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = (
                [threading.Thread(target=updater) for _ in range(3)]
                + [
                    threading.Thread(target=buffer_saver),
                    threading.Thread(target=chain_saver),
                    threading.Thread(target=path_saver),
                    threading.Thread(target=path_reader),
                ]
            )
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(errors, [])
        self._assert_lattice(
            [p.tolist() for p in seq.parameters()], base, grad
        )

    def test_concurrent_update_and_load_never_half_applies(self):
        seq, a, b = self._primed_model()
        base = [p.tolist() for p in seq.parameters()]
        grad = [a._g, b._g]
        checkpoint_buf = bytearray()
        seq.save(checkpoint_buf)
        errors = []

        def updater():
            try:
                for _ in range(200):
                    seq.update(1)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def loader():
            try:
                for _ in range(200):
                    seq.load(bytes(checkpoint_buf))  # atomic reset to base
                    self._assert_lattice(
                        [p.tolist() for p in seq.parameters()], base, grad
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=updater) for _ in range(3)]
        threads += [threading.Thread(target=loader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self._assert_lattice(
            [p.tolist() for p in seq.parameters()], base, grad
        )

    def test_concurrent_distinct_models_same_path_lands_one_full_file(self):
        # Two independent models with identical architecture both hammer
        # one path. Atomic temp+replace means the file is ALWAYS one
        # complete checkpoint: it must parse into a fresh model and equal
        # exactly one of the two writers' current states.
        writer_a, _, _ = self._model()
        writer_b, _, _ = self._model()
        x = Tensor([[1, 1]])
        errors = []
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "shared.ckp")
            writer_a.save(path)

            def drive(model, lr):
                try:
                    for _ in range(120):
                        model.forward(x)
                        model.backward(1)
                        model.update(lr)
                        model.save(path)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            t1 = threading.Thread(target=drive, args=(writer_a, 1))
            t2 = threading.Thread(target=drive, args=(writer_b, 2))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # After the dust settles the file matches one writer exactly.
            final = bytearray()
            probe, _, _ = self._model()
            probe.load(path)  # must be a complete, loadable checkpoint
            a_state = [p.tolist() for p in writer_a.parameters()]
            b_state = [p.tolist() for p in writer_b.parameters()]
            got = [p.tolist() for p in probe.parameters()]
            self.assertIn(got, (a_state, b_state))

            # The committed bytes are exactly one writer's full snapshot.
            a_bytes = bytearray()
            b_bytes = bytearray()
            writer_a.save(a_bytes)
            writer_b.save(b_bytes)
            with open(path, "rb") as fh:
                disk = fh.read()
            self.assertIn(disk, (bytes(a_bytes), bytes(b_bytes)))
        self.assertEqual(errors, [])


class BoundaryAndEagerShapeTests(unittest.TestCase):
    def test_save_after_forward_without_backward_is_allowed(self):
        seq, _ = _stack()
        out, h = seq.forward(Tensor(_SEG1))  # no backward
        buf = bytearray()
        seq.save(buf)  # boundary hidden exists: allowed
        victim, _ = _stack()
        restored = victim.load(buf)
        self.assertEqual(
            [s.tolist() for s in restored], [s.tolist() for s in h]
        )
        self.assertEqual(
            [p.tolist() for p in victim.parameters()],
            [p.tolist() for p in seq.parameters()],
        )

    def test_fresh_model_eagerly_checks_hidden_shapes_on_load(self):
        seq, _ = _stack()
        out, h = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        buf = bytearray()
        seq.save(buf)
        raw = bytes(buf)
        hlen = struct.unpack("<Q", raw[12:20])[0]
        header = json.loads(raw[20 : 20 + hlen])
        # Mangle a hidden-slot shape (flip a dimension) while keeping the
        # declared leaf count inconsistent: fresh model must reject it.
        header["hidden"][0] = [7, 7]
        new_header = json.dumps(
            header, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        end = raw.rfind(cp.END_MAGIC)
        payload = raw[20 + hlen : end]
        body = (
            raw[:8]
            + struct.pack("<I", cp.FORMAT_VERSION)
            + struct.pack("<Q", len(new_header))
            + new_header
            + payload
        )
        crc = zlib.crc32(body[20:])
        leaf_count = len(payload) // 9
        forged = body + cp.END_MAGIC + struct.pack("<QI", leaf_count, crc)
        victim, _ = _stack()  # never forwarded: hidden shapes unknown
        with self.assertRaises(ValueError):
            victim.load(forged)


if __name__ == "__main__":
    unittest.main()
