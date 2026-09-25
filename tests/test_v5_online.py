"""Tests for the streaming online compaction, the read-only chain
verification entry point and the surfaced backward-replay failure.

* compaction folds a chain while walking it and writing the new basis as
  it goes: peak disk usage never exceeds the original chain plus one new
  basis segment, saves/loads keep working, and a process killed at any
  instant leaves one complete chain (old or new head);
* ``verify_chain`` / ``Sequential.verify`` audit a chain read-only and
  localise the first bad segment without changing a single byte;
* if rebuilding layer caches after a failed backward itself fails, the
  rebuild failure surfaces as ValueError with the original layer error
  attached as ``__cause__``.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import struct
import tempfile
import time
import unittest
import zlib

from sequence_engine import MemoryChain, Sequential, Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _base_weights,
    _build,
    _make_v2_chain,
    _SEG1,
    _SEG2,
    _SEG3,
    _total,
)

_LR = 0.1


def _seg_name(i):
    return f"seg-{i:010d}.seqd"


def _chain_bytes(td):
    return cp.build_bytes(cp.load_chain(td))


def _build_chain(td, steps=8):
    """Basis plus *steps* deltas; returns (seq, hidden)."""
    seq, _ = _build(_base_weights())
    seq.save(td)
    hidden = None
    for index in range(steps):
        out, hidden = seq.forward(
            Tensor((_SEG1, _SEG2, _SEG3)[index % 3]), hidden
        )
        seq.backward(_total(out))
        seq.update(_LR)
        seq.save(td)
    return seq, hidden


# ---------------------------------------------------------------------------
# Streaming compaction: basics
# ---------------------------------------------------------------------------


class StreamingCompactBasicsTests(unittest.TestCase):
    def test_full_and_partial_preserve_state_and_counts(self):
        for fold, steps in ((None, 8), (3, 8), (1, 8), (8, 8)):
            with self.subTest(fold=fold):
                with tempfile.TemporaryDirectory() as td:
                    seq, _ = _build_chain(td, steps)
                    expected = cp.build_bytes(cp.load_chain(td))
                    cp.compact_chain(td, fold)
                    new_head = 0 if fold is None else steps - fold
                    self.assertEqual(
                        sorted(os.listdir(td)),
                        ["head"]
                        + [_seg_name(i) for i in range(new_head + 1)],
                    )
                    with open(os.path.join(td, "head"), "rb") as fh:
                        self.assertEqual(fh.read(), str(new_head).encode())
                    self.assertEqual(_chain_bytes(td), expected)

    def test_memory_chain_streaming_compaction(self):
        chain = MemoryChain()
        seq, _ = _build(_base_weights())
        seq.save(chain)
        hidden = None
        full = []
        for index in range(6):
            out, hidden = seq.forward(
                Tensor((_SEG1, _SEG2, _SEG3)[index % 3]), hidden
            )
            seq.backward(_total(out))
            seq.adam_step(0.05)
            seq.save(chain)
        buf = bytearray()
        seq.save(buf)
        full = bytes(buf)
        cp.compact_chain_memory(chain, up_to=2)
        self.assertEqual(len(chain), 5)
        self.assertEqual(chain.read_head(), b"4")
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(chain)), full
        )
        cp.compact_chain_memory(chain)
        self.assertEqual(len(chain), 1)
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(chain)), full
        )

    def test_noop_compactions_touch_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            seq, _ = _build(_base_weights())
            seq.save(td)
            before = sorted(os.listdir(td))
            cp.compact_chain(td)
            cp.compact_chain(td, 0)
            self.assertEqual(sorted(os.listdir(td)), before)


# ---------------------------------------------------------------------------
# Streaming compaction: deterministic crash-point injection
# ---------------------------------------------------------------------------


class _BoomInjection(Exception):
    pass


def _directory_bytes(td):
    return sum(
        os.path.getsize(os.path.join(td, name))
        for name in os.listdir(td)
    )


def _segment_bytes(td):
    return sum(
        os.path.getsize(os.path.join(td, name))
        for name in os.listdir(td)
        if _seg_name_matches(name)
    )


def _seg_name_matches(name):
    return name.startswith("seg-") and name.endswith(".seqd")


class CompactCrashInjectionTests(unittest.TestCase):
    """Inject a crash (raise) at every atomic IO point of a compaction,
    then reopen like a fresh process and demand one complete chain."""

    steps = 7

    def _case(self, fold, crash_at, op):
        """Return None if the crash point was never hit, else the dir."""
        td = tempfile.mkdtemp()
        fired = {"hit": False}
        try:
            _build_chain(td, self.steps)
            expected = _chain_bytes(td)
            counter = [0]
            real_aw = cp._atomic_write
            real_unlink = os.unlink
            real_replace = os.replace

            def tick():
                counter[0] += 1
                if counter[0] == crash_at:
                    fired["hit"] = True
                    raise _BoomInjection()

            if op == "write":
                cp._atomic_write = lambda d, n, r: (tick(), real_aw(d, n, r))[1]
            elif op == "unlink":
                cp.os.unlink = lambda p: (tick(), real_unlink(p))[1]
            else:
                cp.os.replace = lambda a, b: (tick(), real_replace(a, b))[1]
            try:
                cp.compact_chain(td, fold)
            except _BoomInjection:
                pass
            finally:
                cp._atomic_write = real_aw
                cp.os.unlink = real_unlink
                cp.os.replace = real_replace
            if not fired["hit"]:
                return None
            # Fresh-process reopen: recovery must finish to a complete chain.
            got = _chain_bytes(td)
            self.assertEqual(
                got, expected,
                f"fold={fold} crash_at={crash_at} op={op}: state wrong",
            )
            with open(os.path.join(td, "head"), "rb") as fh:
                head = int(fh.read())
            self.assertIn(
                head, (self.steps, self.steps - fold),
                f"fold={fold} crash_at={crash_at}: head {head}",
            )
            segs = [n for n in os.listdir(td) if _seg_name_matches(n)]
            self.assertEqual(len(segs), head + 1)
            debris = [
                n for n in os.listdir(td)
                if n.startswith((".seqc-", ".seqcompact", ".seqckp.tmp-"))
            ]
            self.assertEqual(debris, [])
            # The recovered chain itself verifies clean.
            report = cp.verify_chain(td)
            self.assertEqual((report.head, report.segments), (head, head + 1))
            return td
        finally:
            # Keep only on assertion failure for inspection; otherwise drop.
            shutil.rmtree(td, ignore_errors=True)

    def test_every_crash_point_recovers(self):
        fired = 0
        for fold in (1, 2, 3, 5, 7):
            for op in ("write", "unlink", "replace"):
                for crash_at in range(1, 90):
                    if self._case(fold, crash_at, op) is not None:
                        fired += 1
        self.assertGreater(fired, 100)


# ---------------------------------------------------------------------------
# Streaming compaction: peak disk usage bound
# ---------------------------------------------------------------------------


class CompactDiskPeakTests(unittest.TestCase):
    def test_peak_disk_usage_is_old_chain_plus_one_basis(self):
        for fold in (1, 2, 5, 7, 12, 20, 24):
            with self.subTest(fold=fold):
                with tempfile.TemporaryDirectory() as td:
                    _build_chain(td, 24)
                    orig_segments = _segment_bytes(td)
                    folded_basis = len(
                        cp.build_bytes(cp.load_chain(td, up_to=fold))
                    )
                    bound = orig_segments + folded_basis
                    peak = [orig_segments]
                    real_aw = cp._atomic_write
                    real_unlink = os.unlink
                    real_replace = os.replace

                    def measure():
                        peak[0] = max(peak[0], _segment_bytes(td))

                    def aw(d, n, r):
                        result = real_aw(d, n, r)
                        measure()
                        return result

                    def un(p):
                        result = real_unlink(p)
                        measure()
                        return result

                    def rp(a, b):
                        result = real_replace(a, b)
                        measure()
                        return result

                    cp._atomic_write = aw
                    cp.os.unlink = un
                    cp.os.replace = rp
                    try:
                        cp.compact_chain(td, fold)
                    finally:
                        cp._atomic_write = real_aw
                        cp.os.unlink = real_unlink
                        cp.os.replace = real_replace
                    self.assertLessEqual(
                        peak[0], bound,
                        f"fold={fold}: peak {peak[0]} exceeds old chain + "
                        f"one basis {bound}",
                    )


# ---------------------------------------------------------------------------
# Real subprocess killed mid-roll-forward
# ---------------------------------------------------------------------------


def _compact_worker(td, fold, seed):
    import random as _random
    import time as _time

    _random.seed(seed)
    real_aw = cp._atomic_write

    def slow_aw(d, name, raw):
        if _random.random() < 0.8:
            _time.sleep(_random.random() * 0.03)
        return real_aw(d, name, raw)

    cp._atomic_write = slow_aw
    real_replace = cp.os.replace

    def slow_replace(a, b):
        if _random.random() < 0.8:
            _time.sleep(_random.random() * 0.03)
        return real_replace(a, b)

    cp.os.replace = slow_replace
    cp.compact_chain(td, fold)


class CompactKilledProcessTests(unittest.TestCase):
    def test_killed_during_tail_rollforward(self):
        finished = killed = 0
        for trial in range(12):
            with tempfile.TemporaryDirectory() as td:
                steps = 24
                fold = (1, 2, 3, 12, 24)[trial % 5]
                _build_chain(td, steps)
                expected = _chain_bytes(td)
                new_head = steps - fold
                proc = multiprocessing.get_context("fork").Process(
                    target=_compact_worker, args=(td, fold, trial)
                )
                proc.start()
                marker = os.path.join(td, ".seqcompact")
                deadline = time.time() + 20
                did_kill = False
                while proc.is_alive() and time.time() < deadline:
                    if os.path.exists(marker):
                        time.sleep(0.001 + 0.002 * (trial % 3))
                        proc.kill()
                        did_kill = True
                        killed += 1
                        break
                    time.sleep(0.0005)
                proc.join()
                self.assertEqual(_chain_bytes(td), expected)
                with open(os.path.join(td, "head"), "rb") as fh:
                    head = int(fh.read())
                self.assertIn(head, (steps, new_head))
                segs = [n for n in os.listdir(td) if _seg_name_matches(n)]
                self.assertEqual(len(segs), head + 1)
                self.assertFalse(
                    any(
                        n.startswith((".seqc-", ".seqcompact"))
                        for n in os.listdir(td)
                    )
                )
                if head == new_head:
                    finished += 1
                self.assertTrue(did_kill or proc.exitcode == 0)
        # The sweep must exercise both outcomes across trials.
        self.assertGreater(killed, 0)
        self.assertGreater(finished, 0)


# ---------------------------------------------------------------------------
# Read-only chain verification
# ---------------------------------------------------------------------------


def _snapshot(td):
    return {
        name: open(os.path.join(td, name), "rb").read()
        for name in os.listdir(td)
        if os.path.isfile(os.path.join(td, name))
    }


def _truncate(name):
    def mutate(td):
        path = os.path.join(td, name)
        with open(path, "rb") as fh:
            data = fh.read()
        with open(path, "wb") as fh:
            fh.write(data[: len(data) // 2])
    return mutate


def _delete(name):
    def mutate(td):
        os.unlink(os.path.join(td, name))
    return mutate


def _flip_byte(name):
    def mutate(td):
        path = os.path.join(td, name)
        with open(path, "rb") as fh:
            data = bytearray(fh.read())
        data[len(data) // 2] ^= 0xFF
        with open(path, "wb") as fh:
            fh.write(data)
    return mutate


class VerifyChainTests(unittest.TestCase):
    def _sound_chain_dir(self):
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td, ignore_errors=True)
        _build_chain(td, 3)
        return td

    def test_good_chain_report(self):
        td = self._sound_chain_dir()
        report = cp.verify_chain(td)
        self.assertEqual(report.head, 3)
        self.assertEqual(report.segments, 4)
        self.assertEqual(report.basis_version, cp.FORMAT_VERSION)
        self.assertTrue(report.has_hidden)

    def test_good_chain_directory_is_untouched(self):
        td = self._sound_chain_dir()
        before = _snapshot(td)
        cp.verify_chain(td)
        self.assertEqual(_snapshot(td), before)

    def test_verify_ignores_staged_debris_and_marker(self):
        # Verification audits the reachable chain only: a staged file and
        # an in-flight compaction marker are ignored, not completed, and
        # never written or removed.
        td = self._sound_chain_dir()
        cp._atomic_write(td, cp._staged_name(0), b"not-a-segment")
        cp._atomic_write(td, cp._COMPACT_MARKER, b'{"h":0,"u":2,"d":0}')
        before = _snapshot(td)
        report = cp.verify_chain(td)
        self.assertEqual(report.head, 3)
        self.assertEqual(_snapshot(td), before)

    def test_missing_directory_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                cp.verify_chain(os.path.join(td, "missing"))
            seq, _ = _build(_base_weights())
            with self.assertRaises(FileNotFoundError):
                seq.verify(os.path.join(td, "missing"))

    def test_empty_chain_is_valueerror(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain(td)
            self.assertIn("head", str(ctx.exception))

    def test_corrupt_head_pointer_is_valueerror(self):
        td = self._sound_chain_dir()
        with open(os.path.join(td, "head"), "wb") as fh:
            fh.write(b"not-a-number")
        with self.assertRaises(ValueError):
            cp.verify_chain(td)

    def test_damage_localises_first_bad_segment(self):
        cases = [
            ("truncated seg1", _truncate(_seg_name(1)), 1),
            ("missing seg1", _delete(_seg_name(1)), 1),
            ("crc flip seg1", _flip_byte(_seg_name(1)), 1),
            ("truncated seg3", _truncate(_seg_name(3)), 3),
            ("missing seg2", _delete(_seg_name(2)), 2),
            ("crc flip seg3", _flip_byte(_seg_name(3)), 3),
        ]
        for label, mutate, segment in cases:
            with self.subTest(label):
                td = self._sound_chain_dir()
                mutate(td)
                before = _snapshot(td)
                with self.assertRaises(ValueError) as ctx:
                    cp.verify_chain(td)
                message = str(ctx.exception)
                self.assertIn(f"segment {segment}", message, message)
                self.assertIn(_seg_name(segment), message)
                # Whole chain rejected; not one byte changed.
                self.assertEqual(_snapshot(td), before)

    def test_first_bad_segment_is_the_earliest_one(self):
        td = self._sound_chain_dir()
        _truncate(_seg_name(1))(td)
        _truncate(_seg_name(3))(td)
        with self.assertRaises(ValueError) as ctx:
            cp.verify_chain(td)
        self.assertIn("segment 1", str(ctx.exception))
        self.assertNotIn("segment 3", str(ctx.exception))

    def test_shape_drift_localises_segment(self):
        td = self._sound_chain_dir()
        path = os.path.join(td, _seg_name(1))
        raw = open(path, "rb").read()
        hlen = struct.unpack("<Q", raw[12:20])[0]
        header = json.loads(raw[20 : 20 + hlen])
        end = raw.rfind(cp.DELTA_END_MAGIC)
        payload = raw[20 + hlen : end]
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
        with open(path, "wb") as fh:
            fh.write(
                body
                + cp.DELTA_END_MAGIC
                + struct.pack("<QI", len(payload) // 9, crc)
            )
        with self.assertRaises(ValueError) as ctx:
            cp.verify_chain(td)
        self.assertIn("segment 1", str(ctx.exception))

    def test_missing_field_localises_segment(self):
        td = self._sound_chain_dir()
        path = os.path.join(td, _seg_name(2))
        raw = open(path, "rb").read()
        hlen = struct.unpack("<Q", raw[12:20])[0]
        header = json.loads(raw[20 : 20 + hlen])
        end = raw.rfind(cp.DELTA_END_MAGIC)
        payload = raw[20 + hlen : end]
        header.pop("changed")
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
        with open(path, "wb") as fh:
            fh.write(
                body
                + cp.DELTA_END_MAGIC
                + struct.pack("<QI", len(payload) // 9, crc)
            )
        with self.assertRaises(ValueError) as ctx:
            cp.verify_chain(td)
        self.assertIn("segment 2", str(ctx.exception))

    def test_basis_corruption_localises_segment_0(self):
        td = self._sound_chain_dir()
        path = os.path.join(td, _seg_name(0))
        data = bytearray(open(path, "rb").read())
        data[len(data) // 2] ^= 0xFF
        with open(path, "wb") as fh:
            fh.write(data)
        with self.assertRaises(ValueError) as ctx:
            cp.verify_chain(td)
        self.assertIn("segment 0", str(ctx.exception))

    def test_v2_chain_verifies_with_old_basis_version(self):
        chain, _hidden = _make_v2_chain()
        report = cp.verify_chain_memory(chain)
        self.assertEqual(report.head, 2)
        self.assertEqual(report.basis_version, 2)
        self.assertTrue(report.has_hidden)
        # Same chain on disk verifies through the directory entry point.
        with tempfile.TemporaryDirectory() as td:
            for name in (_seg_name(0), _seg_name(1), _seg_name(2)):
                with open(os.path.join(td, name), "wb") as fh:
                    fh.write(chain.read_segment(name))
            with open(os.path.join(td, "head"), "wb") as fh:
                fh.write(b"2")
            report = cp.verify_chain(td)
            self.assertEqual(report.basis_version, 2)
            before = _snapshot(td)
            cp.verify_chain(td)
            self.assertEqual(_snapshot(td), before)

    def test_verify_memory_chain_and_container_entry(self):
        chain = MemoryChain()
        seq_model, _ = _build(_base_weights())
        seq_model.save(chain)
        out, _ = seq_model.forward(Tensor(_SEG1))
        seq_model.backward(_total(out))
        seq_model.save(chain)
        report = cp.verify_chain_memory(chain)
        self.assertEqual((report.head, report.segments), (1, 2))
        self.assertEqual(seq_model.verify(chain), report)
        with self.assertRaises(TypeError):
            seq_model.verify(bytearray())
        with self.assertRaises(TypeError):
            seq_model.verify(42)

    def test_verify_runs_without_recovering_or_writing(self):
        # A valid marker naming a pending roll-forward must NOT be acted
        # upon by verify: the reachable chain is reported as-is.
        td = self._sound_chain_dir()
        # Mark a streaming compaction that folded through 2 -> new head 1,
        # without actually staging anything; load would reject this, but
        # verify must neither complete it nor fail on it.
        with open(os.path.join(td, ".seqcompact"), "wb") as fh:
            fh.write(b'{"h":1,"u":2,"d":0}')
        before = _snapshot(td)
        report = cp.verify_chain(td)
        self.assertEqual(report.head, 3)
        self.assertEqual(_snapshot(td), before)


# ---------------------------------------------------------------------------
# Backward replay failure is surfaced
# ---------------------------------------------------------------------------


from sequence_engine._selftest import _RNNStep  # noqa: E402


class _LayerBoom(_RNNStep):
    """Backward raises; the replay forward raises too on the second call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_forward = True

    def forward(self, x, hidden):
        if not self._first_forward:
            raise ValueError("replay forward failed")
        self._first_forward = False
        return super().forward(x, hidden)

    def backward(self, upstream):
        raise RuntimeError("layer backward failed")


class BackwardReplayFailureTests(unittest.TestCase):
    def _stack(self):
        weights = _base_weights()
        wrecker = _LayerBoom(
            2, 3, weights["wxh1"], weights["whh1"], weights["b1"]
        )
        quiet = _RNNStep(3, 2, weights["wxh2"], weights["whh2"], weights["b2"])
        return Sequential([wrecker, quiet]), wrecker, quiet, weights

    def test_replay_failure_surfaces_as_valueerror_with_cause(self):
        seq, wrecker, quiet, _weights = self._stack()
        seq.forward(Tensor(_SEG1))
        with self.assertRaises(ValueError) as ctx:
            seq.backward(1.0)
        error = ctx.exception
        # The rebuild failure is the surfaced error and names the stage.
        self.assertIn("rebuild", str(error))
        self.assertIn("replay forward failed", str(error))
        # The original layer error is still handed to the caller, attached.
        self.assertIsInstance(error.__cause__, RuntimeError)
        self.assertEqual(str(error.__cause__), "layer backward failed")
        self.assertIsInstance(error.__context__, ValueError)
        # Partial gradients were rolled back before the replay was tried.
        self.assertIsNone(quiet.wxh.grad)

    def test_successful_replay_keeps_original_error_and_allows_retry(self):
        weights = _base_weights()

        class BoomOnce(_RNNStep):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._boom = True

            def backward(self, upstream):
                if self._boom:
                    self._boom = False
                    self._cache = None
                    raise RuntimeError("transient layer failure")
                return super().backward(upstream)

        wrecker = BoomOnce(2, 3, weights["wxh1"], weights["whh1"], weights["b1"])
        quiet = _RNNStep(3, 2, weights["wxh2"], weights["whh2"], weights["b2"])
        seq = Sequential([wrecker, quiet])
        seq.forward(Tensor(_SEG1))
        with self.assertRaises(RuntimeError) as ctx:
            seq.backward(1.0)
        # Not wrapped: the original layer exception is exactly what the
        # caller sees when the replay succeeds.
        self.assertEqual(str(ctx.exception), "transient layer failure")
        self.assertIsNone(ctx.exception.__cause__)
        # The retry is not a second backward and matches a clean run.
        seq.backward(1.0)
        reference, _ = _build(weights)
        out, _ = reference.forward(Tensor(_SEG1))
        reference.backward(1.0)
        self.assertEqual(
            [p.grad.tolist() for p in seq.parameters()],
            [p.grad.tolist() for p in reference.parameters()],
        )
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)


if __name__ == "__main__":
    unittest.main()
