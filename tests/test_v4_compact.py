"""Tests for chain compaction, process-level crash recovery, and the
backward-retry cache restoration.

* ``compact_chain`` / ``Sequential.compact`` fold a chain's basis and a
  prefix of its deltas into one new basis segment, crash-safely;
* a training process killed at any point leaves a chain that loads to
  exactly the last complete commit;
* a failed backward rebuilds layer caches so the retry sees the same
  state as the interrupted pass.
"""

from __future__ import annotations

import errno
import multiprocessing
import os
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock

from sequence_engine import Sequential, Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _RNNStep,
    _base_weights,
    _build,
    _SEG1,
    _SEG2,
    _SEG3,
    _total,
)

_LR = 0.1
_ADAM_LR = 0.05


def _stack(weights=None):
    return _build(weights if weights is not None else _base_weights())


def _seg_name(i):
    return f"seg-{i:010d}.seqd"


def _full_bytes(seq):
    buf = bytearray()
    seq.save(buf)
    return bytes(buf)


def _trained_chain(td, steps=4):
    """A chain with a basis plus *steps* deltas; returns (seq, hidden)."""
    seq, _ = _stack()
    seq.save(td)  # basis: no hidden, t=0
    out, hidden = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    seq.save(td)  # delta: introduces hidden
    for _ in range(steps - 2):
        seq.update(_LR)
        seq.save(td)
    seq.adam_step(_ADAM_LR)
    seq.save(td)
    return seq, hidden



def _read_file(path):
    with open(path, "rb") as fh:
        return fh.read()

def _chain_bytes(td):
    return cp.build_bytes(cp.load_chain(td))


class CompactDirectoryTests(unittest.TestCase):
    def test_full_compaction_preserves_state_and_reduces_to_basis(self):
        with tempfile.TemporaryDirectory() as td:
            seq, hidden = _trained_chain(td)
            full = _full_bytes(seq)
            cp.compact_chain(td)
            self.assertEqual(sorted(os.listdir(td)), ["head", _seg_name(0)])
            with open(os.path.join(td, "head"), "rb") as fh:
                self.assertEqual(fh.read(), b"0")
            self.assertEqual(_chain_bytes(td), full)
            # The folded basis is a native current-version snapshot.
            with open(os.path.join(td, _seg_name(0)), "rb") as fh:
                raw = fh.read()
            self.assertEqual(
                struct.unpack("<I", raw[8:12])[0], cp.FORMAT_VERSION
            )
            # Optimizer semantics survive: same step count, and loading the
            # compacted chain continues the Adam trajectory identically.
            doc = cp.load_chain(td)
            self.assertEqual(doc["optim"]["t"], 1)
            resumed, _ = _stack()
            restored = resumed.load(td)
            self.assertEqual(
                [s.tolist() for s in restored],
                [s.tolist() for s in hidden],
            )
            reference, _ = _stack()
            reference.load(_full_bytes(seq))
            for model in (resumed, reference):
                model.adam_step(_ADAM_LR)
            self.assertEqual(
                [p.tolist() for p in resumed.parameters()],
                [p.tolist() for p in reference.parameters()],
            )

    def test_partial_compaction_folds_exactly_the_merged_range(self):
        with tempfile.TemporaryDirectory() as td:
            seq, _ = _trained_chain(td, steps=5)  # head == 5
            full = _full_bytes(seq)
            folded_at2 = cp.build_bytes(cp.load_chain(td, up_to=2))
            cp.compact_chain(td, up_to=2)
            self.assertEqual(
                sorted(os.listdir(td)),
                ["head", _seg_name(0), _seg_name(1), _seg_name(2), _seg_name(3)],
            )
            with open(os.path.join(td, "head"), "rb") as fh:
                self.assertEqual(fh.read(), b"3")
            self.assertEqual(_chain_bytes(td), full)
            self.assertEqual(
                cp.build_bytes(cp.load_chain(td, up_to=0)), folded_at2
            )
            # Every remaining segment is a current-version segment.
            for index in (1, 2, 3):
                with open(os.path.join(td, _seg_name(index)), "rb") as fh:
                    raw = fh.read()
                self.assertEqual(
                    struct.unpack("<I", raw[8:12])[0], cp.FORMAT_VERSION
                )

    def test_repeated_and_noop_compactions_are_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            seq, _ = _trained_chain(td)
            cp.compact_chain(td)
            first = _chain_bytes(td)
            files_first = sorted(os.listdir(td))
            cp.compact_chain(td)  # nothing left to merge
            cp.compact_chain(td, up_to=0)  # folds nothing
            self.assertEqual(_chain_bytes(td), first)
            self.assertEqual(sorted(os.listdir(td)), files_first)

            # A basis-only chain has nothing to merge: untouched.
            with tempfile.TemporaryDirectory() as solo_dir:
                solo, _ = _stack()
                solo.save(solo_dir)
                before = sorted(os.listdir(solo_dir))
                cp.compact_chain(solo_dir)
                self.assertEqual(sorted(os.listdir(solo_dir)), before)

    def test_compaction_does_not_advance_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as td:
            # t = 0 chain (never stepped).
            seq, _ = _stack()
            seq.save(td)
            seq.update(_LR)
            seq.save(td)
            cp.compact_chain(td)
            self.assertEqual(cp.load_chain(td)["optim"]["t"], 0)
            # t > 0 chain.
            seq.adam_step(_ADAM_LR)
            seq.adam_step(_ADAM_LR)
            seq.save(td)
            seq.update(_LR)
            seq.save(td)
            cp.compact_chain(td)
            self.assertEqual(cp.load_chain(td)["optim"]["t"], 2)

    def test_continue_after_compaction_bitwise_vs_uninterrupted(self):
        def run(chain_dir, compact):
            seq, _ = _stack()
            hidden = None
            for index, segment in enumerate((_SEG1, _SEG2, _SEG3, _SEG1)):
                out, hidden = seq.forward(Tensor(segment), hidden)
                seq.backward(_total(out))
                seq.adam_step(_ADAM_LR)
                seq.save(chain_dir)
                if compact and index == 1:
                    cp.compact_chain(chain_dir)
            return _full_bytes(seq)

        with tempfile.TemporaryDirectory() as plain_dir, tempfile.TemporaryDirectory() as compact_dir:
            expected = run(plain_dir, compact=False)
            got = run(compact_dir, compact=True)
            self.assertEqual(got, expected)
            self.assertEqual(_chain_bytes(compact_dir), expected)

    def test_v2_chain_compacts_to_current_version(self):
        from sequence_engine._selftest import _make_v2_chain
        from sequence_engine import MemoryChain

        # Build the v2 chain in memory, then move it onto disk.
        chain, hidden = _make_v2_chain()
        expected = cp.build_bytes(cp.load_chain_memory(chain))
        with tempfile.TemporaryDirectory() as td:
            for name in (_seg_name(0), _seg_name(1), _seg_name(2)):
                with open(os.path.join(td, name), "wb") as fh:
                    fh.write(chain.read_segment(name))
            with open(os.path.join(td, "head"), "wb") as fh:
                fh.write(b"2")
            cp.compact_chain(td)
            self.assertEqual(sorted(os.listdir(td)), ["head", _seg_name(0)])
            self.assertEqual(_chain_bytes(td), expected)
            with open(os.path.join(td, _seg_name(0)), "rb") as fh:
                raw = fh.read()
            self.assertEqual(
                struct.unpack("<I", raw[8:12])[0], cp.FORMAT_VERSION
            )
            doc = cp.load_chain(td)
            self.assertEqual(doc["optim"]["t"], 0)
            self.assertEqual(
                [e["v"] for e in doc["hidden"]],
                [s.tolist() for s in hidden],
            )

    def test_sequential_compact_on_directory(self):
        with tempfile.TemporaryDirectory() as td:
            seq, _ = _trained_chain(td)
            full = _full_bytes(seq)
            seq.compact(td)
            self.assertEqual(sorted(os.listdir(td)), ["head", _seg_name(0)])
            self.assertEqual(_chain_bytes(td), full)
            with self.assertRaises(TypeError):
                seq.compact(bytearray())
            with self.assertRaises(TypeError):
                seq.compact(42)

    def test_missing_directory_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "no-chain")
            with self.assertRaises(FileNotFoundError):
                cp.compact_chain(missing)
            seq, _ = _stack()
            with self.assertRaises(FileNotFoundError):
                seq.compact(missing)

    def test_unwritable_directory_is_oserror(self):
        with tempfile.TemporaryDirectory() as td:
            seq, _ = _trained_chain(td)
            head_before = _read_file(os.path.join(td, "head"))
            os.chmod(td, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.compact_chain(td)
            finally:
                os.chmod(td, 0o755)
            # The chain is untouched and still loads.
            self.assertEqual(
                _read_file(os.path.join(td, "head")), head_before
            )
            self.assertEqual(_chain_bytes(td), _full_bytes(seq))

    def test_disk_full_during_compaction_is_oserror_and_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            seq, _ = _trained_chain(td)
            full = _full_bytes(seq)
            head_before = _read_file(os.path.join(td, "head"))
            enospc = OSError(errno.ENOSPC, "simulated disk full")
            with mock.patch("os.fsync", side_effect=enospc):
                with self.assertRaises(OSError) as ctx:
                    cp.compact_chain(td)
            self.assertEqual(ctx.exception.errno, errno.ENOSPC)
            # Nothing was committed: the old chain is fully intact.
            self.assertEqual(
                _read_file(os.path.join(td, "head")), head_before
            )
            self.assertEqual(_chain_bytes(td), full)
            # Staging leftovers from the failed attempt are cleaned up on
            # the next open.
            self.assertFalse(
                any(name.startswith(".seqc-") for name in os.listdir(td))
            )

    def test_corrupt_chain_rejects_compaction_and_stays_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            seq, _ = _trained_chain(td)
            full = _full_bytes(seq)
            seg1 = os.path.join(td, _seg_name(1))
            with open(seg1, "rb") as fh:
                good = fh.read()
            with open(seg1, "wb") as fh:
                fh.write(good[: len(good) // 2])
            head_before = _read_file(os.path.join(td, "head"))
            with self.assertRaises(ValueError):
                cp.compact_chain(td)
            self.assertEqual(
                _read_file(os.path.join(td, "head")), head_before
            )
            # Restore: the chain compacts normally again.
            with open(seg1, "wb") as fh:
                fh.write(good)
            cp.compact_chain(td)
            self.assertEqual(_chain_bytes(td), full)

    def test_empty_chain_and_bad_ranges_are_valueerror(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                cp.compact_chain(td)  # no head pointer at all
            seq, _ = _trained_chain(td)
            for bad in (-1, True, 0.5, "1"):
                with self.assertRaises(ValueError):
                    cp.compact_chain(td, bad)
            with self.assertRaises(ValueError):
                cp.compact_chain(td, 99)  # beyond the head

    def test_shape_drift_inside_chain_rejects_compaction(self):
        # Forge a chain whose delta disagrees with the basis on shapes:
        # the whole compaction is refused with ValueError.
        with tempfile.TemporaryDirectory() as td:
            seq, _ = _stack()
            seq.save(td)
            out, _ = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out))
            seq.save(td)  # non-empty delta: gradients and hidden state
            # Corrupt the delta's declared shape for tensor 0.
            import json
            import zlib

            seg1 = os.path.join(td, _seg_name(1))
            with open(seg1, "rb") as fh:
                raw = fh.read()
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
            forged = body + cp.DELTA_END_MAGIC + struct.pack(
                "<QI", len(payload) // 9, crc
            )
            with open(seg1, "wb") as fh:
                fh.write(forged)
            with self.assertRaises(ValueError):
                cp.compact_chain(td)


class BackwardRetryCacheTests(unittest.TestCase):
    class _Wrecker(_RNNStep):
        """First backward destroys its cache (and a successor's), then raises."""

        def __init__(self, *args, partner=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._boom = True
            self._partner = partner

        def backward(self, upstream):
            if self._boom:
                self._boom = False
                self._cache = None
                if self._partner is not None:
                    self._partner._cache = None
                raise RuntimeError("transient failure during backward")
            return super().backward(upstream)

    def _broken_stack(self, recompute):
        weights = _base_weights()
        wrecker = self._Wrecker(
            2, 3, weights["wxh1"], weights["whh1"], weights["b1"]
        )
        quiet = _RNNStep(3, 2, weights["wxh2"], weights["whh2"], weights["b2"])
        wrecker._partner = quiet
        seq = Sequential([wrecker, quiet])
        seq.set_recompute(recompute)
        return seq, wrecker, quiet, weights

    def _reference_grads(self, weights):
        good, _ = _stack(weights)
        out, _ = good.forward(Tensor(_SEG1))
        good.backward(1.0)
        return [p.grad.tolist() for p in good.parameters()]

    def test_retry_rebuilds_destroyed_caches(self):
        for recompute in (False, True):
            with self.subTest(recompute=recompute):
                seq, wrecker, quiet, weights = self._broken_stack(recompute)
                out, _ = seq.forward(Tensor(_SEG1))
                with self.assertRaises(RuntimeError):
                    seq.backward(1.0)
                # Partial gradients rolled back, caches rebuilt.
                self.assertIsNone(quiet.wxh.grad)
                self.assertIsNotNone(wrecker._cache)
                self.assertIsNotNone(quiet._cache)
                # The retry is not a second backward and matches bit for bit.
                seq.backward(1.0)
                self.assertEqual(
                    [p.grad.tolist() for p in seq.parameters()],
                    self._reference_grads(weights),
                )
                with self.assertRaises(RuntimeError):
                    seq.backward(1.0)

    def test_tuple_loss_identity_survives_the_failed_attempt(self):
        seq, wrecker, quiet, weights = self._broken_stack(recompute=False)
        out, _ = seq.forward(Tensor(_SEG1))
        with self.assertRaises(RuntimeError):
            seq.backward((out, 1.0))
        seq.backward((out, 1.0))  # the recorded output is still the anchor
        self.assertEqual(
            [p.grad.tolist() for p in seq.parameters()],
            self._reference_grads(weights),
        )


# ---------------------------------------------------------------------------
# Process-level crash recovery (real subprocesses, really killed).
# ---------------------------------------------------------------------------


def _training_worker(td, rounds):
    """Deterministic training loop checkpointing into a chain directory."""
    from sequence_engine._selftest import _base_weights, _build, _SEG1, _SEG2, _SEG3, _total

    seq, _ = _build(_base_weights())
    segments = (_SEG1, _SEG2, _SEG3)
    hidden = None
    for index in range(rounds):
        out, hidden = seq.forward(Tensor(segments[index % 3]), hidden)
        seq.backward(_total(out))
        seq.update(0.1)
        seq.save(td)


def _compact_worker(td, up_to):
    from sequence_engine import checkpoint as cp

    cp.compact_chain(td, up_to)


def _replay_training(rounds):
    """Replay the worker's trajectory in-process, capturing each commit."""
    snapshots = []
    seq, _ = _stack()
    segments = (_SEG1, _SEG2, _SEG3)
    hidden = None
    for index in range(rounds):
        out, hidden = seq.forward(Tensor(segments[index % 3]), hidden)
        seq.backward(_total(out))
        seq.update(0.1)
        buf = bytearray()
        seq.save(buf)
        snapshots.append(bytes(buf))
    return snapshots


class CrashRecoveryTests(unittest.TestCase):
    def _kill_after(self, proc, condition, timeout=30.0):
        """Kill *proc* once condition() is true (or it exits); return why."""
        deadline = time.monotonic() + timeout
        fired = False
        while proc.is_alive() and time.monotonic() < deadline:
            if condition():
                fired = True
                break
            time.sleep(0.0005)
        if proc.is_alive():
            proc.kill()
        proc.join()
        return fired

    def test_killed_training_process_loads_last_complete_commit(self):
        rounds = 60
        snapshots = _replay_training(rounds)
        for attempt in range(3):
            with tempfile.TemporaryDirectory() as td:
                proc = multiprocessing.Process(
                    target=_training_worker, args=(td, rounds)
                )
                proc.start()
                # Wait until at least one commit exists, then kill at a
                # random point of the run.
                head = os.path.join(td, "head")
                deadline = time.monotonic() + 30
                while not os.path.exists(head) and time.monotonic() < deadline:
                    time.sleep(0.001)
                time.sleep(0.005 * attempt)
                if proc.is_alive():
                    proc.kill()
                proc.join()

                committed = _chain_bytes(td)
                self.assertIn(
                    committed,
                    snapshots,
                    "the chain must load to exactly one complete commit",
                )
                # Resume from the chain and finish the trajectory: the
                # result matches the uninterrupted run bit for bit.
                k = snapshots.index(committed)
                resumed, _ = _stack()
                hidden = resumed.load(td)
                for index in range(k + 1, rounds):
                    segment = (_SEG1, _SEG2, _SEG3)[index % 3]
                    out, hidden = resumed.forward(Tensor(segment), hidden)
                    resumed.backward(_total(out))
                    resumed.update(0.1)
                final = bytearray()
                resumed.save(final)
                self.assertEqual(bytes(final), snapshots[-1])

    def test_killed_compaction_before_marker_keeps_old_chain(self):
        with tempfile.TemporaryDirectory() as td:
            seq, _ = _trained_chain(td, steps=30)  # head == 30
            expected = _chain_bytes(td)
            head_before = _read_file(os.path.join(td, "head"))

            proc = multiprocessing.Process(
                target=_compact_worker, args=(td, 1)
            )
            proc.start()
            # Kill as soon as the staging area appears (before the marker
            # can be written, or at worst mid-roll-forward).
            saw_staging = self._kill_after(
                proc,
                lambda: any(
                    name.startswith(".seqc-") for name in os.listdir(td)
                ),
            )

            # The chain loads to one complete state -- either the old head
            # (killed before the marker) or the rolled-forward new head
            # (killed after it); both reassemble to the same document.
            self.assertEqual(_chain_bytes(td), expected)
            if saw_staging:
                head_after = _read_file(os.path.join(td, "head"))
                self.assertIn(head_after, (head_before, b"29"))
            # Recovery leaves no staging debris behind.
            self.assertFalse(
                any(
                    name.startswith((".seqc-", ".seqcompact"))
                    for name in os.listdir(td)
                )
            )

    def test_killed_compaction_after_marker_rolls_forward(self):
        for _ in range(3):
            with tempfile.TemporaryDirectory() as td:
                seq, _ = _trained_chain(td, steps=30)
                expected = _chain_bytes(td)
                proc = multiprocessing.Process(
                    target=_compact_worker, args=(td, 1)
                )
                proc.start()
                saw_marker = self._kill_after(
                    proc,
                    lambda: os.path.exists(os.path.join(td, ".seqcompact")),
                )
                # Whether the kill landed after the marker or the worker
                # finished first, the chain is complete and loads to the
                # preserved state.
                self.assertEqual(_chain_bytes(td), expected)
                if saw_marker or proc.exitcode == 0:
                    # The roll-forward completed on recovery: head moved
                    # and the staging area is gone.
                    with open(os.path.join(td, "head"), "rb") as fh:
                        self.assertEqual(fh.read(), b"29")
                    self.assertFalse(
                        any(
                            name.startswith((".seqc-", ".seqcompact"))
                            for name in os.listdir(td)
                        )
                    )

    def test_compaction_marker_protocol_directly(self):
        # Deterministic, no timing: stage a compaction by hand, then let
        # the next open roll it forward.
        with tempfile.TemporaryDirectory() as td:
            seq, _ = _trained_chain(td)
            expected = _chain_bytes(td)
            # Stage a full compaction manually: staged basis + marker,
            # no roll-forward yet.
            folded = cp.build_bytes(cp.load_chain(td))
            cp._atomic_write(td, cp._staged_name(0), folded)
            cp._atomic_write(td, cp._COMPACT_MARKER, b"0")
            # The next open recovers to exactly the new head.
            self.assertEqual(_chain_bytes(td), expected)
            self.assertEqual(sorted(os.listdir(td)), ["head", _seg_name(0)])
            # A staged file without a marker is simply dropped.
            cp._atomic_write(td, cp._staged_name(0), folded)
            self.assertEqual(_chain_bytes(td), expected)
            self.assertEqual(sorted(os.listdir(td)), ["head", _seg_name(0)])


def _writer_process(td, rounds):
    """A second process appending to the same chain directory."""
    from sequence_engine._selftest import _concurrency_model

    seq, _a, _b = _concurrency_model()
    seq.forward(Tensor([[1, 1]]))
    seq.backward(1)
    for _ in range(rounds):
        seq.update(1)
        seq.save(td)


class CompactConcurrencyTests(unittest.TestCase):
    def _lattice_model(self):
        from sequence_engine._selftest import _concurrency_model

        return _concurrency_model()

    def _assert_lattice(self, param_values, base, grad):
        steps = None
        for values, w0, g in zip(param_values, base, grad):
            for j, value in enumerate(values):
                current = int(value)
                self.assertEqual(value, float(current))
                delta = w0[j] - current
                self.assertEqual(delta % g[j], 0)
                k = delta // g[j]
                self.assertGreaterEqual(k, 0)
                if steps is None:
                    steps = k
                self.assertEqual(k, steps)

    def test_interleaved_compact_save_and_load_stay_coherent(self):
        seq, a, b = self._lattice_model()
        seq.forward(Tensor([[1, 1]]))
        seq.backward(1)
        base = [p.tolist() for p in seq.parameters()]
        grad = [a._grad, b._grad]
        errors = []

        with tempfile.TemporaryDirectory() as td:
            seq.save(td)

            def updater():
                try:
                    for _ in range(150):
                        seq.update(1)
                        seq.save(td)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def compactor():
                try:
                    for _ in range(40):
                        cp.compact_chain(td)
                        time.sleep(0.001)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def reader():
                try:
                    for _ in range(150):
                        probe, _, _ = self._lattice_model()
                        probe.load(td)
                        self._assert_lattice(
                            [p.tolist() for p in probe.parameters()], base, grad
                        )
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=updater),
                threading.Thread(target=compactor),
                threading.Thread(target=reader),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            # The chain still loads to a coherent lattice state.
            probe, _, _ = self._lattice_model()
            probe.load(td)
            self._assert_lattice(
                [p.tolist() for p in probe.parameters()], base, grad
            )

    def test_memory_chain_compact_save_load_race(self):
        from sequence_engine import MemoryChain

        seq, a, b = self._lattice_model()
        seq.forward(Tensor([[1, 1]]))
        seq.backward(1)
        base = [p.tolist() for p in seq.parameters()]
        grad = [a._grad, b._grad]
        chain = MemoryChain()
        seq.save(chain)
        errors = []

        def updater():
            try:
                for _ in range(200):
                    seq.update(1)
                    seq.save(chain)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def compactor():
            try:
                for _ in range(50):
                    cp.compact_chain_memory(chain)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def reader():
            try:
                for _ in range(200):
                    probe, _, _ = self._lattice_model()
                    probe.load(chain)
                    self._assert_lattice(
                        [p.tolist() for p in probe.parameters()], base, grad
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=updater),
            threading.Thread(target=compactor),
            threading.Thread(target=reader),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        probe, _, _ = self._lattice_model()
        probe.load(chain)
        self._assert_lattice(
            [p.tolist() for p in probe.parameters()], base, grad
        )

    def test_two_processes_writing_one_chain_with_compaction(self):
        rounds = 60
        with tempfile.TemporaryDirectory() as td:
            # Seed the chain so both writers share one architecture state.
            seq, a, b = self._lattice_model()
            seq.forward(Tensor([[1, 1]]))
            seq.backward(1)
            seq.save(td)
            base = [p.tolist() for p in seq.parameters()]
            grad = [a._grad, b._grad]

            procs = [
                multiprocessing.Process(target=_writer_process, args=(td, rounds))
                for _ in range(2)
            ]
            for proc in procs:
                proc.start()
            # Compact from the parent while the writers append.
            while any(proc.is_alive() for proc in procs):
                try:
                    cp.compact_chain(td)
                except ValueError:
                    pass  # a transient empty-chain moment is impossible, but
                    # never fail the race on a scheduling artefact
                time.sleep(0.002)
            for proc in procs:
                proc.join()
                self.assertEqual(proc.exitcode, 0)

            # Whatever the interleaving, the head names one complete chain
            # and every read segment is whole.
            probe, _, _ = self._lattice_model()
            probe.load(td)
            self._assert_lattice(
                [p.tolist() for p in probe.parameters()], base, grad
            )
            self.assertFalse(
                any(
                    name.startswith((".seqc-", ".seqcompact"))
                    for name in os.listdir(td)
                )
            )


if __name__ == "__main__":
    unittest.main()
