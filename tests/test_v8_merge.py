"""Tests for chain merges: landing one chain's current state on another.

A merge appends exactly one delta segment to the target chain -- the
delta carries only the tensors that differ between the target's head
state and the source's head state -- so loading the target afterwards
reassembles bit for bit to the source state at merge time.  The target
keeps every segment it had, the source chain is never modified, shared
prefix segments are never stored a second time, and an identical state
merges as an empty delta (a repeated merge is a no-op on the state).

These tests cover:

* target append semantics and bitwise state equality with the source;
* empty-delta idempotence and the source chain staying untouched;
* shared-prefix storage (hard links) with the appended segment owned by
  the target alone, plus independent evolution afterwards;
* a full save after the merge matching the merge result without
  advancing any optimizer step;
* the error taxonomy (FileNotFoundError / ValueError / OSError) and the
  guarantee that a rejected merge changes the target by no byte;
* crash-residue sweep and kill-at-any-point safety of the merge;
* concurrency with appends, streaming compactions and reverse merges;
* read-only family verification after merges;
* the in-memory (MemoryChain) merge and the Sequential entry point.
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
import threading
import time
import unittest

from sequence_engine import MemoryChain, Sequential, Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _base_weights,
    _build,
    _SEG1,
    _SEG2,
    _total,
)

_LR = 0.1
_ADAM_LR = 0.05


def _stack(weights=None):
    return _build(weights if weights is not None else _base_weights())


def _seg(i):
    return f"seg-{i:010d}.seqd"


def _train(seq, td, steps, hidden=None, start=0):
    """Run *steps* deterministic train/save cycles numbered from *start*."""
    for i in range(start, start + steps):
        out, hidden = seq.forward(
            Tensor(_SEG1 if i % 2 == 0 else _SEG2), hidden
        )
        seq.backward(_total(out))
        if i % 3 == 2:
            seq.adam_step(_ADAM_LR)
        else:
            seq.update(_LR)
        seq.save(td)
    return hidden


def _trained_dir(td, steps):
    """A fresh chain directory: basis save + *steps* cycles (head == steps)."""
    os.mkdir(td)
    seq, _ = _stack()
    seq.save(td)
    hidden = _train(seq, td, steps)
    return seq, hidden


def _chain_state(td):
    return cp.build_bytes(cp.load_chain(td))


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _snapshot_dir(td):
    return {
        name: _read(os.path.join(td, name))
        for name in os.listdir(td)
        if os.path.isfile(os.path.join(td, name))
    }


class MergeSemanticsTests(unittest.TestCase):
    def test_merge_appends_one_segment_and_matches_source_bitwise(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            other = os.path.join(root, "other")
            os.mkdir(other)
            oseq, _ = _stack()
            oseq.save(other)
            out, oh = oseq.forward(Tensor(_SEG2))
            oseq.backward(_total(out))
            oseq.adam_step(_ADAM_LR)
            oseq.save(other)

            source_state = _chain_state(main)
            before = {
                name: raw
                for name, raw in _snapshot_dir(other).items()
                if name != "head"
            }
            cp.merge_chain(main, other)

            # The target kept every segment it had and gained exactly one.
            self.assertEqual(
                sorted(os.listdir(other)),
                ["head", _seg(0), _seg(1), _seg(2)],
            )
            self.assertEqual(_read(os.path.join(other, "head")), b"2")
            for name, raw in before.items():
                self.assertEqual(_read(os.path.join(other, name)), raw)
            # It now reassembles to the source state bit for bit.
            self.assertEqual(_chain_state(other), source_state)
            # The source chain is untouched.
            self.assertEqual(_read(os.path.join(main, "head")), b"4")
            self.assertEqual(_chain_state(main), source_state)

    def test_merge_into_a_forked_branch_keeps_the_shared_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            # The branch diverges with its own segment 3.
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            out, bhidden = bseq.forward(Tensor(_SEG2), bhidden)
            bseq.backward(_total(out))
            bseq.adam_step(_ADAM_LR)
            bseq.save(branch)

            source_state = _chain_state(main)
            cp.merge_chain(main, branch)
            self.assertEqual(_read(os.path.join(branch, "head")), b"4")
            self.assertEqual(_chain_state(branch), source_state)
            # Prefix segments are still stored once (hard links); the
            # appended segment is owned by the target alone.
            for index in range(3):
                self.assertEqual(
                    os.stat(os.path.join(main, _seg(index))).st_ino,
                    os.stat(os.path.join(branch, _seg(index))).st_ino,
                )
            self.assertEqual(
                os.stat(os.path.join(branch, _seg(4))).st_nlink, 1
            )
            self.assertNotEqual(
                os.stat(os.path.join(main, _seg(4))).st_ino,
                os.stat(os.path.join(branch, _seg(4))).st_ino,
            )
            self.assertTrue(cp.verify_chain([main, branch]).ok)

    def test_merge_same_state_appends_an_empty_delta_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)  # identical head states
            state = _chain_state(main)

            cp.merge_chain(main, branch)
            self.assertEqual(_read(os.path.join(branch, "head")), b"4")
            number, _hc, _pending, items = cp._parse_delta(
                _read(os.path.join(branch, _seg(4))), 4
            )
            self.assertEqual(items, [])
            self.assertEqual(_chain_state(branch), state)

            # Merging the same state again appends another empty delta but
            # changes nothing about the reassembled state.
            cp.merge_chain(main, branch)
            self.assertEqual(_read(os.path.join(branch, "head")), b"5")
            _n, _h, _p, items2 = cp._parse_delta(
                _read(os.path.join(branch, _seg(5))), 5
            )
            self.assertEqual(items2, [])
            self.assertEqual(_chain_state(branch), state)
            # The source never moved.
            self.assertEqual(_read(os.path.join(main, "head")), b"3")

    def test_repeated_merges_always_add_exactly_one_segment(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            seq, hidden = _trained_dir(main, 2)
            other = os.path.join(root, "other")
            cp.fork_chain(main, other, up_to=0)
            for _ in range(3):
                cp.merge_chain(main, other)
                head = int(_read(os.path.join(other, "head")))
                self.assertEqual(_chain_state(other), _chain_state(main))
                self.assertEqual(len([n for n in os.listdir(other) if n.endswith(".seqd")]), head + 1)
            # Advance the source and merge: still exactly one new segment.
            _train(seq, main, 2, hidden=hidden, start=2)
            cp.merge_chain(main, other)
            self.assertEqual(_chain_state(other), _chain_state(main))

    def test_chains_evolve_independently_after_a_merge(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            mseq, mhidden = _trained_dir(main, 3)
            other = os.path.join(root, "other")
            os.mkdir(other)
            oseq, _ = _stack()
            oseq.save(other)

            cp.merge_chain(main, other)
            # Each chain then takes a different training step.
            _train(mseq, main, 1, hidden=mhidden, start=3)
            nseq, _ = _stack()
            nhidden = nseq.load(other)
            out, nhidden = nseq.forward(Tensor(_SEG2), nhidden)
            nseq.backward(_total(out))
            nseq.adam_step(_ADAM_LR)
            nseq.save(other)

            self.assertNotEqual(_chain_state(main), _chain_state(other))
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertTrue(cp.verify_chain(other).ok)

    def test_full_save_after_merge_matches_and_advances_no_step(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            other = os.path.join(root, "other")
            os.mkdir(other)
            nseq, _ = _stack()
            nseq.save(other)

            cp.merge_chain(main, other)
            merged_doc = cp.load_chain(other)
            t_at_merge = merged_doc["optim"]["t"]

            restored, _ = _stack()
            restored.load(other)
            self.assertEqual(restored._adam_t, t_at_merge)
            buf = bytearray()
            restored.save(buf)
            self.assertEqual(bytes(buf), cp.build_bytes(merged_doc))
            # Saving into the chain then still reassembles identically.
            restored.save(other)
            self.assertEqual(_chain_state(other), bytes(buf))
            self.assertEqual(restored._adam_t, t_at_merge)


class MergeErrorTests(unittest.TestCase):
    def test_merge_a_chain_into_itself_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            with self.assertRaises(ValueError):
                cp.merge_chain(main, main)

    def test_missing_directories_are_filenotfound_and_isolated(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            other = os.path.join(root, "other")
            os.mkdir(other)
            oseq, _ = _stack()
            oseq.save(other)
            before = _snapshot_dir(other)

            with self.assertRaises(FileNotFoundError):
                cp.merge_chain(os.path.join(root, "absent"), other)
            with self.assertRaises(FileNotFoundError):
                cp.merge_chain(main, os.path.join(root, "absent"))
            # The present chain was not touched.
            self.assertEqual(_snapshot_dir(other), before)
            self.assertTrue(cp.verify_chain(main).ok)

    def test_missing_referenced_segment_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            other = os.path.join(root, "other")
            os.mkdir(other)
            oseq, _ = _stack()
            oseq.save(other)
            before = _snapshot_dir(other)

            os.unlink(os.path.join(main, _seg(2)))
            with self.assertRaises(FileNotFoundError):
                cp.merge_chain(main, other)
            # The target changed by no byte.
            self.assertEqual(_snapshot_dir(other), before)
            self.assertEqual(_read(os.path.join(other, "head")), b"0")

            # A missing referenced segment on either side is a
            # FileNotFoundError; nothing is written into the other chain.
            os.unlink(os.path.join(main, _seg(0)))
            with self.assertRaises(FileNotFoundError):
                cp.merge_chain(other, main)
            self.assertEqual(_read(os.path.join(other, "head")), b"0")

    def test_shape_and_layer_order_mismatch_rejected_wholesale(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 1)
            other = os.path.join(root, "other")
            os.mkdir(other)
            different_weights = _base_weights()
            different_weights["b1"] = [0.01, -0.02]  # shape [2] not [3]
            dseq, _ = _stack(different_weights)
            dseq.save(other)
            before = _snapshot_dir(other)

            with self.assertRaises(ValueError):
                cp.merge_chain(main, other)
            self.assertEqual(_snapshot_dir(other), before)
            self.assertTrue(cp.verify_chain(main).ok)

    def test_corrupt_chain_is_rejected_before_the_target_changes(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            other = os.path.join(root, "other")
            os.mkdir(other)
            oseq, _ = _stack()
            oseq.save(other)
            before = _snapshot_dir(other)

            path = os.path.join(main, _seg(1))
            with open(path, "r+b") as fh:
                fh.truncate(os.path.getsize(path) // 2)
            with self.assertRaises(ValueError):
                cp.merge_chain(main, other)
            self.assertEqual(_snapshot_dir(other), before)

    def test_headless_chain_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 1)
            other = os.path.join(root, "other")
            os.mkdir(other)
            with self.assertRaises(ValueError):
                cp.merge_chain(main, other)
            with self.assertRaises(ValueError):
                cp.merge_chain(other, main)

    def test_type_checks(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 1)
            other = os.path.join(root, "other")
            os.mkdir(other)
            with self.assertRaises(TypeError):
                cp.merge_chain(123, other)
            with self.assertRaises(TypeError):
                cp.merge_chain(main, 123)
            seq, _ = _stack()
            with self.assertRaises(TypeError):
                seq.merge(main, 123)
            with self.assertRaises(TypeError):
                seq.merge(MemoryChain(), other)
            with self.assertRaises(TypeError):
                seq.merge(main, MemoryChain())

    def test_unwritable_target_is_oserror_without_half_segment(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write into read-only directories")
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 1)
            other = os.path.join(root, "other")
            os.mkdir(other)
            oseq, _ = _stack()
            oseq.save(other)
            os.chmod(other, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.merge_chain(main, other)
                # No temp file lingers in the read-only target.
                self.assertEqual(
                    [n for n in os.listdir(other) if n.startswith(".")], []
                )
            finally:
                os.chmod(other, 0o755)
            self.assertTrue(cp.verify_chain(other).ok)


class MergeCrashTests(unittest.TestCase):
    def test_killed_merge_leaves_old_or_new_head_only(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 30)
            other = os.path.join(root, "other")
            os.mkdir(other)
            oseq, _ = _stack()
            oseq.save(other)
            state = _chain_state(main)

            proc = multiprocessing.Process(
                target=_slow_merge_worker, args=(main, other)
            )
            proc.start()
            deadline = time.monotonic() + 30.0
            saw_temp = False
            while proc.is_alive() and time.monotonic() < deadline:
                if any(
                    name.startswith(cp._TMP_PREFIX)
                    for name in os.listdir(other)
                ) or _read(os.path.join(other, "head")) != b"0":
                    saw_temp = True
                    break
                time.sleep(0.001)
            if proc.is_alive():
                proc.kill()
            proc.join()
            self.assertTrue(saw_temp)

            # The head names exactly one complete chain: either the old
            # head 0 or the new complete head 31.
            head = _read(os.path.join(other, "head"))
            self.assertIn(head, (b"0", b"31"))
            # Reopening the target at either head already yields one
            # complete, verifiable state.
            self.assertTrue(cp.verify_chain(other).ok)

            # The next merge lands the source state exactly and sweeps
            # any residue deterministically (empty delta if it had
            # already landed).
            cp.merge_chain(main, other)
            self.assertEqual(_chain_state(other), state)
            self.assertTrue(cp.verify_chain(other).ok)
            self.assertEqual(
                [n for n in os.listdir(other) if n.startswith(".")], []
            )

    def test_orphan_and_temp_residue_is_swept_by_the_next_merge(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=1)
            # Kill residue: an orphan segment beyond the head and a
            # leftover atomic-write temp file.
            with open(os.path.join(branch, _seg(9)), "wb") as fh:
                fh.write(b"orphan")
            with open(
                os.path.join(branch, cp._TMP_PREFIX + "leftover"), "wb"
            ) as fh:
                fh.write(b"tmp")

            cp.merge_chain(main, branch)
            names = os.listdir(branch)
            self.assertNotIn(_seg(9), names)
            self.assertEqual(
                [n for n in names if n.startswith(cp._TMP_PREFIX)], []
            )
            self.assertEqual(_chain_state(branch), _chain_state(main))
            self.assertTrue(cp.verify_chain(branch).ok)

    def test_merge_residue_is_also_swept_by_compact_and_fork(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            cp.merge_chain(main, branch)  # branch head 3
            state = _chain_state(branch)

            # Simulate a kill between the segment write and the head
            # advance on a follow-up merge: the orphan lands beyond head.
            with open(os.path.join(branch, _seg(5)), "wb") as fh:
                fh.write(b"orphan")
            cp.compact_chain(branch)
            self.assertNotIn(_seg(5), os.listdir(branch))
            self.assertEqual(_chain_state(branch), state)

            # A fork off the compacted branch sweeps family staging and
            # the forked chain is complete; the orphan never traveled.
            child = os.path.join(root, "child")
            cp.fork_chain(branch, child)
            self.assertTrue(cp.verify_chain(child).ok)

    def test_family_verify_after_merge_attributes_a_bad_shared_segment(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            cp.merge_chain(main, branch)  # heads align, prefix shared

            # Damage a segment the merged family still shares (segment 1).
            shared = os.path.join(main, _seg(1))
            with open(shared, "r+b") as fh:
                fh.truncate(os.path.getsize(shared) // 2)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain([main, branch])
            message = str(ctx.exception)
            self.assertIn("segment 1", message)
            self.assertIn("chain 0", message)
            # The defect is reachable from both chains after the merge.
            self.assertIn("chain 1", message)


def _slow_merge_worker(source, target):
    from sequence_engine import checkpoint as cp

    original_replace = cp.os.replace

    def slow_replace(src, dst):
        time.sleep(0.02)
        return original_replace(src, dst)

    cp.os.replace = slow_replace
    try:
        cp.merge_chain(source, target)
    finally:
        cp.os.replace = original_replace


class MergeConcurrencyTests(unittest.TestCase):
    def test_merges_interleave_with_streaming_compactions(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 20)
            other = os.path.join(root, "other")
            os.mkdir(other)
            oseq, _ = _stack()
            oseq.save(other)
            errors = []
            stop = False

            def compactor(directory):
                try:
                    while not stop:
                        cp.compact_chain(directory)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=compactor, args=(other,)),
                threading.Thread(target=compactor, args=(main,)),
            ]
            for thread in threads:
                thread.start()
            try:
                for _ in range(8):
                    cp.merge_chain(main, other)
                    cp.merge_chain(other, main)
            finally:
                stop = True
                for thread in threads:
                    thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(_chain_state(main), _chain_state(other))
            self.assertTrue(cp.verify_chain([main, other]).ok)

    def test_reverse_direction_merges_cannot_deadlock(self):
        with tempfile.TemporaryDirectory() as root:
            a = os.path.join(root, "a")
            _trained_dir(a, 4)
            b = os.path.join(root, "b")
            cp.fork_chain(a, b, up_to=2)
            errors = []

            def merge(src, dst):
                try:
                    for _ in range(20):
                        cp.merge_chain(src, dst)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=merge, args=(a, b)),
                threading.Thread(target=merge, args=(b, a)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertTrue(cp.verify_chain([a, b]).ok)

    def test_appends_loads_and_merges_flow_together(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 6)
            other = os.path.join(root, "other")
            # A full cycle first so both chains already fix hidden state;
            # an appender saving a loaded document can then never race a
            # merge across the no-hidden/has-hidden boundary.
            _trained_dir(other, 1)
            errors = []
            stop = False

            def appender(directory):
                try:
                    while not stop:
                        doc = cp.load_chain(directory)
                        cp.save_chain(doc, directory)  # empty deltas
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def reader(directory):
                try:
                    while not stop:
                        cp.build_bytes(cp.load_chain(directory))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            # Only the merge source appends: a merge never changes the
            # source state, so its load/save cycles stay state-preserving
            # and the target converges to the source at the last merge.
            # Readers cover both chains throughout.
            threads = [threading.Thread(target=appender, args=(main,))]
            threads += [
                threading.Thread(target=reader, args=(main,)),
                threading.Thread(target=reader, args=(other,)),
            ]
            for thread in threads:
                thread.start()
            try:
                for _ in range(10):
                    cp.merge_chain(main, other)
            finally:
                stop = True
                for thread in threads:
                    thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(_chain_state(main), _chain_state(other))
            self.assertTrue(cp.verify_chain([main, other]).ok)

    def test_merge_completes_after_an_interrupted_compaction(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            other = os.path.join(root, "other")
            os.mkdir(other)
            oseq, _ = _stack()
            oseq.save(other)  # seg 0
            out, _oh = oseq.forward(Tensor(_SEG1))
            oseq.backward(_total(out))
            oseq.update(_LR)
            oseq.save(other)  # seg 1
            # A dead streaming fold on the target: the staged basis folds
            # segments 0..1 and the marker names the fold point.
            folded = cp.build_bytes(cp.load_chain(other, up_to=1))
            cp._atomic_write(other, cp._staged_name(0), folded)
            cp._atomic_write(other, cp._COMPACT_MARKER, b'{"u":1}')
            # The merge first rolls the dead fold forward, then appends.
            cp.merge_chain(main, other)
            self.assertEqual(_chain_state(other), _chain_state(main))
            self.assertTrue(cp.verify_chain(other).ok)


class MemoryMergeTests(unittest.TestCase):
    def _memory_chain(self, steps, seq=None):
        chain = MemoryChain()
        if seq is None:
            seq, _ = _stack()
        seq.save(chain)
        hidden = None
        for i in range(steps):
            out, hidden = seq.forward(
                Tensor(_SEG1 if i % 2 == 0 else _SEG2), hidden
            )
            seq.backward(_total(out))
            if i % 2 == 0:
                seq.update(_LR)
            else:
                seq.adam_step(_ADAM_LR)
            seq.save(chain)
        return chain, seq, hidden

    def test_memory_merge_appends_one_delta_and_source_is_untouched(self):
        source, _, _ = self._memory_chain(3)
        target, _, _ = self._memory_chain(1)
        source_state = cp.build_bytes(cp.load_chain_memory(source))
        source_head = source.read_head()

        cp.merge_chain_memory(source, target)
        self.assertEqual(target.read_head(), b"2")
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(target)), source_state
        )
        self.assertEqual(source.read_head(), source_head)
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(source)), source_state
        )

        # An identical follow-up merge appends an empty delta.
        cp.merge_chain_memory(source, target)
        self.assertEqual(target.read_head(), b"3")
        _n, _h, _p, items = cp._parse_delta(
            target.read_segment(_seg(3)), 3
        )
        self.assertEqual(items, [])
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(target)), source_state
        )

    def test_memory_merge_keeps_target_segments_and_independence(self):
        source, sseq, _ = self._memory_chain(3)
        target, tseq, thidden = self._memory_chain(1)
        originals = {
            name: target.read_segment(name)
            for name in list(target._objects)
            if name != "head"
        }
        cp.merge_chain_memory(source, target)
        for name, raw in originals.items():
            self.assertEqual(target.read_segment(name), raw)

        sseq.update(_LR)
        sseq.save(source)
        out, thidden = tseq.forward(Tensor(_SEG2), thidden)
        tseq.backward(_total(out))
        tseq.adam_step(_ADAM_LR)
        tseq.save(target)
        self.assertNotEqual(
            cp.build_bytes(cp.load_chain_memory(source)),
            cp.build_bytes(cp.load_chain_memory(target)),
        )
        self.assertTrue(cp.verify_chain_memory(source).ok)
        self.assertTrue(cp.verify_chain_memory(target).ok)

    def test_memory_merge_errors(self):
        source, _, _ = self._memory_chain(2)
        target, _, _ = self._memory_chain(1)
        with self.assertRaises(ValueError):
            cp.merge_chain_memory(source, source)
        with self.assertRaises(ValueError):
            cp.merge_chain_memory(MemoryChain(), target)
        with self.assertRaises(ValueError):
            cp.merge_chain_memory(source, MemoryChain())
        with self.assertRaises(TypeError):
            cp.merge_chain_memory(object(), target)
        with self.assertRaises(TypeError):
            cp.merge_chain_memory(source, object())

        different_weights = _base_weights()
        different_weights["b1"] = [0.01, -0.02]
        dseq, _ = _stack(different_weights)
        other = MemoryChain()
        dseq.save(other)
        head_before = target.read_head()
        with self.assertRaises(ValueError):
            cp.merge_chain_memory(other, target)
        self.assertEqual(target.read_head(), head_before)

    def test_sequential_merge_entry_points(self):
        source, _, _ = self._memory_chain(2)
        target, _, _ = self._memory_chain(1)
        seq, _ = _stack()
        self.assertIsNone(seq.merge(source, target))
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(target)),
            cp.build_bytes(cp.load_chain_memory(source)),
        )
        with self.assertRaises(TypeError):
            seq.merge(source, "a-directory")
        with self.assertRaises(TypeError):
            seq.merge("a-directory", target)


if __name__ == "__main__":
    unittest.main()
