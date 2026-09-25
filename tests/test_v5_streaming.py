"""Tests for the streaming online compaction, read-only chain
verification and the surfaced backward cache-rebuild failure.

These complement the black-box compaction/recovery tests in
``test_v4_compact.py`` by exercising the new guarantees directly:

* compaction streams: saves and loads interleave and the on-disk peak
  stays within the original chain plus one new basis segment;
* ``verify_chain`` / ``Sequential.verify`` locate the first bad segment
  and never write a byte;
* a backward whose cache-rebuild replay also fails surfaces that failure
  as a ``ValueError`` chained onto the original layer exception.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import threading
import zlib
import unittest

from sequence_engine import MemoryChain, Sequential, Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _base_weights,
    _build,
    _RNNStep,
    _SEG1,
    _total,
)

_LR = 0.1


def _stack(weights=None):
    return _build(weights if weights is not None else _base_weights())


def _seg(i):
    return f"seg-{i:010d}.seqd"


def _trained_dir(td, steps):
    seq, _ = _stack()
    seq.save(td)
    for _ in range(steps):
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        seq.update(_LR)
        seq.save(td)
    return seq


def _chain_hash(td):
    digest = hashlib.sha256()
    for name in sorted(os.listdir(td)):
        path = os.path.join(td, name)
        if os.path.isfile(path):
            digest.update(name.encode("ascii"))
            with open(path, "rb") as fh:
                digest.update(fh.read())
    return digest.hexdigest()


def _renumber_delta(path, new_number):
    with open(path, "rb") as fh:
        raw = fh.read()
    hlen = struct.unpack("<Q", raw[12:20])[0]
    header = json.loads(raw[20 : 20 + hlen])
    payload = raw[20 + hlen : raw.rfind(cp.DELTA_END_MAGIC)]
    header["n"] = new_number
    new_header = json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    body = (
        raw[:8]
        + struct.pack("<I", cp.FORMAT_VERSION)
        + struct.pack("<Q", len(new_header))
        + new_header
        + payload
    )
    crc = zlib.crc32(body[20:])
    return body + cp.DELTA_END_MAGIC + struct.pack(
        "<QI", len(payload) // 9, crc
    )


class StreamingCompactionTests(unittest.TestCase):
    def test_state_and_segment_count_after_streaming_compaction(self):
        with tempfile.TemporaryDirectory() as td:
            seq = _trained_dir(td, 5)
            full = cp.build_bytes(cp.load_chain(td))
            cp.compact_chain(td, up_to=2)
            with open(os.path.join(td, "head"), "rb") as fh:
                head = int(fh.read())
            self.assertEqual(head, 3)
            self.assertEqual(
                sorted(f for f in os.listdir(td) if f.startswith("seg-")),
                [_seg(i) for i in range(4)],
            )
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), full)
            cp.compact_chain(td)
            self.assertEqual(
                sorted(os.listdir(td)), ["head", _seg(0)]
            )
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), full)

    def test_peak_disk_usage_within_old_chain_plus_one_basis(self):
        with tempfile.TemporaryDirectory() as td:
            steps, fold = 30, 10
            _trained_dir(td, steps)
            old_total = sum(
                os.path.getsize(os.path.join(td, name))
                for name in os.listdir(td)
                if name.startswith("seg-")
            )
            new_basis = len(cp.build_bytes(cp.load_chain(td, up_to=fold)))
            samples = []
            original_write = cp._atomic_write

            def measuring_write(directory, final_name, raw):
                result = original_write(directory, final_name, raw)
                samples.append(
                    sum(
                        os.path.getsize(os.path.join(directory, name))
                        for name in os.listdir(directory)
                        if name.startswith(("seg-", ".seqc-"))
                    )
                )
                return result

            cp._atomic_write = measuring_write
            try:
                cp.compact_chain(td, up_to=fold)
            finally:
                cp._atomic_write = original_write
            # Every committed state during the fold fits in the original
            # chain plus the single new basis segment.
            self.assertLessEqual(max(samples), old_total + new_basis)

    def test_saves_and_loads_interleave_a_directory_compaction(self):
        with tempfile.TemporaryDirectory() as td:
            seq = _trained_dir(td, 30)
            errors = []
            stop = False

            def appender():
                try:
                    while not stop:
                        doc = cp.load_chain(td)
                        cp.save_chain(doc, td)  # an unchanged empty delta
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def reader():
                try:
                    while not stop:
                        cp.build_bytes(cp.load_chain(td))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=appender)]
            threads += [threading.Thread(target=reader) for _ in range(2)]
            for thread in threads:
                thread.start()
            for _ in range(20):
                cp.compact_chain(td)
            stop = True
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            head_state = cp.build_bytes(cp.load_chain(td))
            cp.compact_chain(td)
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), head_state)
            self.assertEqual(sorted(os.listdir(td)), ["head", _seg(0)])

    def test_interrupted_stream_leaves_no_debris_after_reopen(self):
        with tempfile.TemporaryDirectory() as td:
            seq = _trained_dir(td, 20)
            cp.compact_chain(td, up_to=8)
            self.assertFalse(
                any(
                    name.startswith((".seqc-", ".seqcompact"))
                    for name in os.listdir(td)
                )
            )
            cp.compact_chain(td)
            self.assertEqual(sorted(os.listdir(td)), ["head", _seg(0)])


class VerifyTests(unittest.TestCase):
    def test_sound_chain_returns_a_success_report(self):
        with tempfile.TemporaryDirectory() as td:
            _trained_dir(td, 4)
            report = cp.verify_chain(td)
            self.assertTrue(report.ok)
            self.assertEqual(report.head, 4)
            self.assertEqual(report.segments, 5)
            seq, _ = _stack()
            self.assertTrue(seq.verify(td).ok)

    def test_sound_memory_chain_verifies(self):
        chain = MemoryChain()
        seq, _ = _stack()
        seq.save(chain)
        for _ in range(5):
            out, _ = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out))
            seq.update(_LR)
            seq.save(chain)
        report = cp.verify_chain_memory(chain)
        self.assertTrue(report.ok)
        self.assertEqual((report.head, report.segments), (5, 6))
        self.assertTrue(seq.verify(chain).ok)

    def test_verify_locates_first_bad_segment_and_touches_nothing(self):
        cases = []
        with tempfile.TemporaryDirectory() as td:
            _trained_dir(td, 5)
            good = {}
            for index in (0, 2, 4):
                with open(os.path.join(td, _seg(index)), "rb") as fh:
                    good[index] = fh.read()

            # Truncated delta -> segment 2 named; bytes untouched.
            with open(os.path.join(td, _seg(2)), "wb") as fh:
                fh.write(good[2][: len(good[2]) // 2])
            before = _chain_hash(td)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain(td)
            self.assertIn("segment 2", str(ctx.exception))
            self.assertEqual(_chain_hash(td), before)

            # Restore; an out-of-order frame is located at that segment.
            with open(os.path.join(td, _seg(2)), "wb") as fh:
                fh.write(good[2])
            forged_4 = _renumber_delta(os.path.join(td, _seg(4)), 99)
            with open(os.path.join(td, _seg(4)), "wb") as fh:
                fh.write(forged_4)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain(td)
            self.assertIn("segment 4", str(ctx.exception))
            with open(os.path.join(td, _seg(4)), "wb") as fh:
                fh.write(good[4])

            # A corrupt basis is located at segment 0.
            with open(os.path.join(td, _seg(0)), "wb") as fh:
                fh.write(good[0][:24])
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain(td)
            self.assertIn("segment 0", str(ctx.exception))
            with open(os.path.join(td, _seg(0)), "wb") as fh:
                fh.write(good[0])
            self.assertTrue(cp.verify_chain(td).ok)

    def test_verify_locates_shape_drift_in_a_delta(self):
        with tempfile.TemporaryDirectory() as td:
            seq, _ = _stack()
            seq.save(td)
            out, _ = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out))
            seq.save(td)  # seg 1 carries gradients and hidden state
            path = os.path.join(td, _seg(1))
            with open(path, "rb") as fh:
                raw = fh.read()
            hlen = struct.unpack("<Q", raw[12:20])[0]
            header = json.loads(raw[20 : 20 + hlen])
            payload = raw[20 + hlen : raw.rfind(cp.DELTA_END_MAGIC)]
            header["changed"][0]["s"] = [9, 9]
            new_header = json.dumps(
                header, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            body = (
                raw[:8]
                + struct.pack("<I", cp.FORMAT_VERSION)
                + struct.pack("<Q", len(new_header))
                + new_header
                + payload
            )
            crc = zlib.crc32(body[20:])
            forged = body + cp.DELTA_END_MAGIC + struct.pack(
                "<QI", len(payload) // 9, crc
            )
            with open(path, "wb") as fh:
                fh.write(forged)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain(td)
            self.assertIn("segment 1", str(ctx.exception))

    def test_verify_read_only_directory_and_empty_chain(self):
        with tempfile.TemporaryDirectory() as td:
            _trained_dir(td, 3)
            before = _chain_hash(td)
            os.chmod(td, 0o555)
            try:
                self.assertTrue(cp.verify_chain(td).ok)
            finally:
                os.chmod(td, 0o755)
            self.assertEqual(_chain_hash(td), before)
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(ValueError):
                cp.verify_chain(empty)

    def test_verify_missing_directory_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                cp.verify_chain(os.path.join(td, "absent"))

    def test_verify_bad_head_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            _trained_dir(td, 2)
            with open(os.path.join(td, "head"), "wb") as fh:
                fh.write(b"garbage")
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain(td)
            self.assertIn("head", str(ctx.exception))

    def test_verify_type_checks(self):
        with self.assertRaises(TypeError):
            cp.verify_chain(123)
        with self.assertRaises(TypeError):
            cp.verify_chain_memory(object())
        seq, _ = _stack()
        with self.assertRaises(TypeError):
            seq.verify(b"not-a-chain")


class BackwardReplayFailureTests(unittest.TestCase):
    class _Broken(_RNNStep):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._boom = True
            self.fail_replay = False

        def forward(self, x, hidden):
            if self.fail_replay:
                raise ValueError("replay forward blew up")
            return super().forward(x, hidden)

        def backward(self, upstream):
            if self._boom:
                self._boom = False
                self._cache = None
                raise RuntimeError("layer backward blew up")
            return super().backward(upstream)

    def test_failed_rebuild_surfaces_as_chained_valueerror(self):
        weights = _base_weights()
        layer0 = self._Broken(2, 3, weights["wxh1"], weights["whh1"], weights["b1"])
        layer1 = _RNNStep(3, 2, weights["wxh2"], weights["whh2"], weights["b2"])
        seq = Sequential([layer0, layer1])
        seq.forward(Tensor(_SEG1))
        layer0.fail_replay = True
        with self.assertRaises(RuntimeError) as ctx:
            seq.backward(1.0)
        # The original layer exception still reaches the caller verbatim.
        self.assertIn("layer backward blew up", str(ctx.exception))
        cause = ctx.exception.__cause__
        self.assertIsInstance(cause, ValueError)
        message = str(cause)
        self.assertIn("rebuild", message)
        self.assertIn("replay forward blew up", message)
        # Partial gradients from the successor were rolled back.
        self.assertIsNone(layer1.wxh.grad)

    def test_successful_rebuild_still_allows_bitwise_retry(self):
        # Existing contract: rebuild succeeds -> retry is not a second
        # backward and the gradients match an uninterrupted pass.
        weights = _base_weights()

        class Wrecker(_RNNStep):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._boom = True

            def backward(self, upstream):
                if self._boom:
                    self._boom = False
                    self._cache = None
                    raise RuntimeError("boom")
                return super().backward(upstream)

        wrecker = Wrecker(2, 3, weights["wxh1"], weights["whh1"], weights["b1"])
        quiet = _RNNStep(3, 2, weights["wxh2"], weights["whh2"], weights["b2"])
        seq = Sequential([wrecker, quiet])
        out, _ = seq.forward(Tensor(_SEG1))
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)
        seq.backward(1.0)
        good, _ = _stack(weights)
        good_out, _ = good.forward(Tensor(_SEG1))
        good.backward(1.0)
        self.assertEqual(
            [p.grad.tolist() for p in seq.parameters()],
            [p.grad.tolist() for p in good.parameters()],
        )
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)


if __name__ == "__main__":
    unittest.main()
