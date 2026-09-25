"""Tests for branch merges: landing one chain's state onto another.

A merge appends exactly one new segment to the target chain -- the
delta between the target's head state and the source's head state --
keeping every segment the target already had.  Loading the target
afterwards is bit for bit identical to loading the source at merge
time; the source chain's head and segment files never move.  Shared
segments are never stored twice (the appended segment is the target's
alone and holds only genuinely differing tensors), an equal state
merges as an empty delta, and both chains keep evolving independently.

These tests cover:

* merge semantics: one target-owned segment appended, target loads to
  the source state, source untouched, shared prefix still hard-linked
  while the merge tail is exclusive;
* idempotence (empty delta), post-merge evolution, full saves and
  compaction after a merge, and that no optimizer step advances;
* the error taxonomy (FileNotFoundError / ValueError / OSError) and
  the guarantee that a rejected merge rewrites not one target byte;
* kill-at-any-point safety: the head is the old or the new one and the
  orphan residue is reclaimed deterministically by the next fork,
  compaction, deletion or merge;
* concurrency of merges (opposing directions included) with saves and
  compactions;
* read-only family verification still covering every member.
"""

from __future__ import annotations

import errno
import multiprocessing
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from sequence_engine import Sequential, Tensor
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


def _train(seq, td, steps, hidden=None, start=0, adam=()):
    """Run *steps* deterministic train/save cycles numbered from *start*."""
    for i in range(start, start + steps):
        out, hidden = seq.forward(Tensor(_SEG1 if i % 2 == 0 else _SEG2), hidden)
        seq.backward(_total(out))
        if i in adam or i % 3 == 2:
            seq.adam_step(_ADAM_LR)
        else:
            seq.update(_LR)
        seq.save(td)
    return hidden


def _trained_dir(td, steps):
    os.mkdir(td)
    seq, _ = _stack()
    seq.save(td)
    hidden = _train(seq, td, steps)
    return seq, hidden


def _chain_state(td):
    return cp.build_bytes(cp.load_chain(td))


def _dir_names(td):
    return sorted(os.listdir(td))


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


class MergeSemanticsTests(unittest.TestCase):
    def test_merge_appends_one_target_owned_segment_with_source_state(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            # The branch diverges from the main line.
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 2, hidden=bhidden, start=2)

            source_state = _chain_state(main)
            source_head = _read(os.path.join(main, "head"))
            source_listing = _dir_names(main)
            target_head_before = int(_read(os.path.join(branch, "head")))
            target_names_before = set(_dir_names(branch))

            cp.merge_chains(main, branch)

            # Exactly one segment was appended to the target; its head
            # moved by one and it now loads to the source state.
            self.assertEqual(
                _read(os.path.join(branch, "head")),
                str(target_head_before + 1).encode("ascii"),
            )
            new_names = set(_dir_names(branch)) - target_names_before
            self.assertEqual(new_names, {_seg(target_head_before + 1)})
            self.assertEqual(_chain_state(branch), source_state)

            # The source chain is byte for byte untouched.
            self.assertEqual(_read(os.path.join(main, "head")), source_head)
            self.assertEqual(_dir_names(main), source_listing)
            self.assertEqual(_chain_state(main), source_state)

            # The shared prefix keeps two links; the merge tail is the
            # target's alone.
            for index in range(3):
                self.assertEqual(
                    os.stat(os.path.join(branch, _seg(index))).st_nlink, 2
                )
            self.assertEqual(
                os.stat(
                    os.path.join(branch, _seg(target_head_before + 1))
                ).st_nlink,
                1,
            )
            self.assertTrue(cp.verify_chain([main, branch]).ok)

    def test_equal_states_merge_as_an_empty_delta_and_repeat_is_stable(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=3)  # identical head states
            state = _chain_state(main)
            head_before = int(_read(os.path.join(branch, "head")))

            cp.merge_chains(main, branch)
            # The appended delta carries no changed tensors at all.
            _n, _hc, _p, items = cp._parse_delta(
                _read(os.path.join(branch, _seg(head_before + 1))),
                head_before + 1,
            )
            self.assertEqual(items, [])
            self.assertEqual(_chain_state(branch), state)

            # Re-merging the same state appends another empty delta and
            # leaves the state unchanged; the segment count still grows.
            cp.merge_chains(main, branch)
            _n, _hc, _p, items = cp._parse_delta(
                _read(os.path.join(branch, _seg(head_before + 2))),
                head_before + 2,
            )
            self.assertEqual(items, [])
            self.assertEqual(_chain_state(branch), state)
            self.assertEqual(
                _read(os.path.join(branch, "head")),
                str(head_before + 2).encode("ascii"),
            )
            self.assertTrue(cp.verify_chain(branch).ok)

    def test_merge_carries_optimizer_state_without_advancing_a_step(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 5)  # cycles 2 and 5 adam-step
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=1)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=1)  # no adam here

            source_t = cp.load_chain(main)["optim"]["t"]
            target_t_before = cp.load_chain(branch)["optim"]["t"]
            self.assertNotEqual(source_t, target_t_before)

            cp.merge_chains(main, branch)
            merged = cp.load_chain(branch)
            self.assertEqual(merged["optim"]["t"], source_t)
            self.assertEqual(
                cp.build_bytes(merged),
                cp.build_bytes(cp.load_chain(main)),
            )

    def test_full_save_after_merge_lands_the_merged_state(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=1)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 2, hidden=bhidden, start=1)

            cp.merge_chains(main, branch)
            merged_state = _chain_state(branch)

            model, _ = _stack()
            model.load(branch)
            full_path = os.path.join(root, "full.seq")
            model.save(full_path)
            self.assertEqual(_read(full_path), merged_state)

            # Saving back into the chain appends an empty delta, after
            # which further training still tracks full snapshots.
            model.save(branch)
            self.assertEqual(_chain_state(branch), merged_state)
            out, hidden = model.forward(Tensor(_SEG2))
            model.backward(_total(out))
            model.update(_LR)
            full2 = os.path.join(root, "full2.seq")
            model.save(full2)
            model.save(branch)
            self.assertEqual(_chain_state(branch), _read(full2))

    def test_chains_evolve_independently_after_a_merge(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=1)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=1)

            cp.merge_chains(main, branch)
            main_state = _chain_state(main)

            # More training on the source cannot move the merged target:
            # it stays at the source's merge-time state (main head 3).
            mseq, mhidden = _stack()
            mhidden = mseq.load(main)
            _train(mseq, main, 2, hidden=mhidden, start=3)
            self.assertEqual(
                cp.build_bytes(cp.load_chain(branch)),
                cp.build_bytes(cp.load_chain(main, up_to=3)),
            )
            self.assertNotEqual(_chain_state(branch), _chain_state(main))

            # And the target's own next steps do not touch the source.
            source_state = _chain_state(main)
            tseq, thidden = _stack()
            thidden = tseq.load(branch)
            _train(tseq, branch, 2, hidden=thidden, start=3)
            self.assertEqual(_chain_state(main), source_state)
            self.assertTrue(cp.verify_chain([main, branch]).ok)

            # Compacting either chain after the merge preserves state and
            # leaves the other intact.
            branch_state = _chain_state(branch)
            cp.compact_chain(branch)
            self.assertEqual(_chain_state(branch), branch_state)
            self.assertEqual(_chain_state(main), source_state)
            cp.compact_chain(main)
            self.assertEqual(_chain_state(main), source_state)
            self.assertTrue(cp.verify_chain([main, branch]).ok)

    def test_merge_works_in_both_directions_between_forked_chains(self):
        with tempfile.TemporaryDirectory() as root:
            a = os.path.join(root, "a")
            _trained_dir(a, 3)
            b = os.path.join(root, "b")
            cp.fork_chain(a, b, up_to=1)
            bseq, _ = _stack()
            bhidden = bseq.load(b)
            _train(bseq, b, 3, hidden=bhidden, start=1)

            a_state = _chain_state(a)
            b_state = _chain_state(b)
            # a <- b then b <- a leaves each loading to the other's state.
            cp.merge_chains(b, a)
            self.assertEqual(_chain_state(a), b_state)
            cp.merge_chains(b, a)  # idempotent empty delta
            cp.merge_chains(a, b)  # b was still at its old state
            self.assertEqual(_chain_state(b), b_state)
            self.assertTrue(cp.verify_chain([a, b]).ok)
            self.assertNotEqual(a_state, b_state)

    def test_merge_between_unrelated_identical_shape_chains(self):
        # A merge is a state operation, not a fork operation: two chains
        # of identical shape and layer order merge whether or not they
        # share files.
        with tempfile.TemporaryDirectory() as root:
            a = os.path.join(root, "a")
            b = os.path.join(root, "b")
            _trained_dir(a, 3)
            _trained_dir(b, 1)
            # Independent inodes even for the prefix.
            self.assertNotEqual(
                os.stat(os.path.join(a, _seg(0))).st_ino,
                os.stat(os.path.join(b, _seg(0))).st_ino,
            )
            a_state = _chain_state(a)
            cp.merge_chains(a, b)
            self.assertEqual(_chain_state(b), a_state)
            # No sharing was created: every target segment stays
            # single-linked.
            for name in _dir_names(b):
                if name.startswith("seg-"):
                    self.assertEqual(
                        os.stat(os.path.join(b, name)).st_nlink, 1
                    )
            self.assertTrue(cp.verify_chain([a, b]).ok)


class MergeErrorTests(unittest.TestCase):
    def test_missing_directories_are_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            main_before = _dir_names(main)
            branch_before = _dir_names(branch)

            with self.assertRaises(FileNotFoundError):
                cp.merge_chains(os.path.join(root, "absent"), branch)
            with self.assertRaises(FileNotFoundError):
                cp.merge_chains(main, os.path.join(root, "absent2"))
            # A plain file is not a chain directory either.
            fake = os.path.join(root, "file")
            with open(fake, "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(FileNotFoundError):
                cp.merge_chains(fake, branch)
            with self.assertRaises(FileNotFoundError):
                cp.merge_chains(main, fake)
            self.assertEqual(_dir_names(main), main_before)
            self.assertEqual(_dir_names(branch), branch_before)

    def test_missing_referenced_segment_is_filenotfound_and_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=2)
            # A hole in the source's reachable chain.
            os.unlink(os.path.join(main, _seg(2)))
            source_head_before = _read(os.path.join(main, "head"))
            target_listing = _dir_names(branch)
            with self.assertRaises(FileNotFoundError):
                cp.merge_chains(main, branch)
            self.assertEqual(_read(os.path.join(main, "head")), source_head_before)
            self.assertEqual(_dir_names(branch), target_listing)

            # A hole in the target rejects the merge too.
            with self.assertRaises(FileNotFoundError):
                cp.merge_chains(branch, main)

    def test_self_merge_is_valueerror_and_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            before = _dir_names(main)
            with self.assertRaises(ValueError):
                cp.merge_chains(main, main + os.sep)  # lexical variants
            self.assertEqual(_dir_names(main), before)

    def test_shape_mismatch_is_valueerror_and_target_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            other_weights = _base_weights()
            other_weights["b1"] = [0.01, -0.02]  # shape [2] instead of [3]
            other = os.path.join(root, "other")
            os.mkdir(other)
            oseq, _ = _stack(other_weights)
            oseq.save(other)
            target_listing = _dir_names(main)
            target_head = _read(os.path.join(main, "head"))
            with self.assertRaises(ValueError):
                cp.merge_chains(other, main)
            self.assertEqual(_dir_names(main), target_listing)
            self.assertEqual(_read(os.path.join(main, "head")), target_head)

    def test_layer_order_mismatch_is_valueerror_and_target_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 1)
            # Same parameter shapes, different layer kind: forge a basis.
            seq, _ = _stack()
            doc = cp.load_chain(main)
            for layer in doc["layers"]:
                layer["kind"] = "SomethingElse"
            other = os.path.join(root, "other")
            os.mkdir(other)
            with open(os.path.join(other, _seg(0)), "wb") as fh:
                fh.write(cp.build_bytes(doc))
            with open(os.path.join(other, "head"), "wb") as fh:
                fh.write(b"0")
            before = _dir_names(main)
            with self.assertRaises(ValueError):
                cp.merge_chains(other, main)
            self.assertEqual(_dir_names(main), before)

    def test_corrupt_source_is_valueerror_and_target_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=2)
            tail = os.path.join(branch, _seg(3))
            with open(tail, "r+b") as fh:
                fh.truncate(os.path.getsize(tail) // 2)
            target_before = _dir_names(main)
            with self.assertRaises(ValueError):
                cp.merge_chains(branch, main)
            self.assertEqual(_dir_names(main), target_before)

    def test_empty_chains_are_valueerror(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 1)
            empty = os.path.join(root, "empty")
            os.mkdir(empty)
            with self.assertRaises(ValueError):
                cp.merge_chains(empty, main)
            with self.assertRaises(ValueError):
                cp.merge_chains(main, empty)

    def test_disk_full_writing_the_segment_is_oserror_old_head_kept(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=0)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=0)
            head_before = _read(os.path.join(branch, "head"))
            listing_before = set(_dir_names(branch))
            with mock.patch.object(
                cp.os,
                "replace",
                side_effect=OSError(errno.ENOSPC, "simulated disk full"),
            ):
                with self.assertRaises(OSError) as ctx:
                    cp.merge_chains(main, branch)
                self.assertEqual(ctx.exception.errno, errno.ENOSPC)
            # The head never moved and no temp file lingers; the target
            # is still its old, complete self.
            self.assertEqual(_read(os.path.join(branch, "head")), head_before)
            names = set(_dir_names(branch))
            self.assertEqual(names, listing_before)
            self.assertFalse(
                any(n.startswith(cp._TMP_PREFIX) for n in names)
            )
            self.assertTrue(cp.verify_chain(branch).ok)

    def test_readonly_target_is_oserror_and_target_untouched(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write into read-only directories")
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=0)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=0)
            head_before = _read(os.path.join(branch, "head"))
            listing_before = _dir_names(branch)
            os.chmod(branch, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.merge_chains(main, branch)
            finally:
                os.chmod(branch, 0o755)
            self.assertEqual(_read(os.path.join(branch, "head")), head_before)
            self.assertEqual(_dir_names(branch), listing_before)
            self.assertTrue(cp.verify_chain(branch).ok)

    def test_type_checks(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 1)
            with self.assertRaises(TypeError):
                cp.merge_chains(123, main)
            with self.assertRaises(TypeError):
                cp.merge_chains(main, 456)
            seq, _ = _stack()
            with self.assertRaises(TypeError):
                seq.merge(main, 123)


class MergeReclamationTests(unittest.TestCase):
    def test_killed_merge_residue_reclaimed_by_each_family_operation(self):
        for trigger in ("fork", "compact", "delete", "merge"):
            with self.subTest(trigger=trigger):
                with tempfile.TemporaryDirectory() as root:
                    main = os.path.join(root, "main")
                    _trained_dir(main, 3)
                    branch = os.path.join(root, "branch")
                    cp.fork_chain(main, branch, up_to=1)
                    bseq, _ = _stack()
                    bhidden = bseq.load(branch)
                    _train(bseq, branch, 1, hidden=bhidden, start=1)
                    # Simulate a kill after the merge segment was written
                    # but before the head advanced: an orphan one past the
                    # old head plus a commit temp file.
                    head = int(_read(os.path.join(branch, "head")))
                    orphan = os.path.join(branch, _seg(head + 1))
                    with open(orphan, "wb") as fh:
                        fh.write(b"garbage orphan segment bytes")
                    tmp = os.path.join(branch, ".seqckp.tmp-dead")
                    with open(tmp, "wb") as fh:
                        fh.write(b"half")
                    state_before = _chain_state(branch)

                    # The orphan lives in *branch*, so the triggering
                    # family operation must act on that chain.
                    if trigger == "fork":
                        cp.fork_chain(branch, os.path.join(root, "b2"))
                    elif trigger == "compact":
                        cp.compact_chain(branch, up_to=0)
                    elif trigger == "delete":
                        victim = os.path.join(root, "victim")
                        cp.fork_chain(branch, victim)
                        cp.delete_chain(victim)
                    else:
                        cp.merge_chains(main, branch)

                    self.assertFalse(os.path.exists(tmp))
                    self.assertTrue(cp.verify_chain(branch).ok)
                    if trigger != "merge":
                        # The non-merge triggers only reclaim: the orphan
                        # slot is gone and the state is exactly as before.
                        self.assertFalse(os.path.exists(orphan))
                        self.assertEqual(_chain_state(branch), state_before)
                    else:
                        # The merge trigger reclaims the garbage and then
                        # writes the genuine merge segment into that same
                        # slot: valid bytes, target now at source state.
                        self.assertTrue(os.path.exists(orphan))
                        head_after = int(_read(os.path.join(branch, "head")))
                        self.assertEqual(head_after, head + 1)
                        cp._parse_delta(_read(orphan), head + 1)
                        self.assertEqual(
                            _chain_state(branch), _chain_state(main)
                        )

    def test_merge_sweeps_orphan_segments_in_the_target_only(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            # Unreachable debris in both directories.
            with open(os.path.join(branch, _seg(9)), "wb") as fh:
                fh.write(b"target orphan")
            source_orphan = os.path.join(main, _seg(9))
            with open(source_orphan, "wb") as fh:
                fh.write(b"source orphan")
            cp.merge_chains(main, branch)
            self.assertFalse(os.path.exists(os.path.join(branch, _seg(9))))
            # A merge never touches the source directory's files.
            self.assertTrue(os.path.exists(source_orphan))
            # The source's own next family operation reclaims it.
            cp.compact_chain(main, up_to=0)
            self.assertFalse(os.path.exists(source_orphan))


def _slow_merge_worker(source, target):
    from sequence_engine import checkpoint as cp

    original_fsync = os.fsync

    def slow_fsync(fd):
        time.sleep(0.05)
        return original_fsync(fd)

    cp.os.fsync = slow_fsync
    try:
        cp.merge_chains(source, target)
    finally:
        cp.os.fsync = original_fsync


class MergeCrashTests(unittest.TestCase):
    def test_killed_merge_leaves_one_complete_chain(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 20)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=10)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 5, hidden=bhidden, start=10)
            old_head = int(_read(os.path.join(branch, "head")))
            source_state = _chain_state(main)

            proc = multiprocessing.Process(
                target=_slow_merge_worker, args=(main, branch)
            )
            proc.start()
            deadline = time.monotonic() + 30.0
            saw_activity = False
            while proc.is_alive() and time.monotonic() < deadline:
                names = os.listdir(branch)
                if any(
                    n.startswith(cp._TMP_PREFIX)
                    or n == _seg(old_head + 1)
                    for n in names
                ):
                    saw_activity = True
                    break
                time.sleep(0.001)
            if proc.is_alive():
                proc.kill()
            proc.join()
            self.assertTrue(saw_activity)

            # Whatever survived at the target path loads as one complete
            # chain, at either the old or the new head.
            head = int(_read(os.path.join(branch, "head")))
            self.assertIn(head, (old_head, old_head + 1))
            self.assertTrue(cp.verify_chain(branch).ok)
            if head == old_head:
                # The merge did not commit: the orphan residue is swept
                # and a retry lands the source state.
                self.assertEqual(
                    cp.build_bytes(cp.load_chain(branch, up_to=old_head)),
                    cp.build_bytes(cp.load_chain(branch)),
                )
                cp.merge_chains(main, branch)
                self.assertEqual(_chain_state(branch), source_state)
            else:
                self.assertEqual(_chain_state(branch), source_state)
            # The source survived untouched and complete.
            self.assertEqual(_chain_state(main), source_state)
            self.assertTrue(cp.verify_chain([main, branch]).ok)
            cp.compact_chain(branch)
            self.assertTrue(cp.verify_chain(branch).ok)


class MergeConcurrencyTests(unittest.TestCase):
    def test_opposing_merges_do_not_deadlock_and_states_stay_complete(self):
        with tempfile.TemporaryDirectory() as root:
            a = os.path.join(root, "a")
            b = os.path.join(root, "b")
            _trained_dir(a, 10)
            cp.fork_chain(a, b, up_to=5)
            bseq, _ = _stack()
            bhidden = bseq.load(b)
            _train(bseq, b, 4, hidden=bhidden, start=5)
            errors = []
            state_a0 = _chain_state(a)
            state_b0 = _chain_state(b)

            def merge_ab():
                try:
                    for _ in range(50):
                        cp.merge_chains(a, b)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def merge_ba():
                try:
                    for _ in range(50):
                        cp.merge_chains(b, a)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=merge_ab),
                threading.Thread(target=merge_ba),
            ]
            for thread in threads:
                thread.start()
                thread.join(timeout=30)
                self.assertFalse(thread.is_alive(), "opposing merges deadlocked")
            self.assertEqual(errors, [])
            self.assertTrue(cp.verify_chain([a, b]).ok)
            # Only state copies moved between the chains, so each final
            # state is one of the two states the runs started from.
            self.assertIn(_chain_state(a), (state_a0, state_b0))
            self.assertIn(_chain_state(b), (state_a0, state_b0))

    def test_merges_racing_saves_and_compactions(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 12)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=6)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 3, hidden=bhidden, start=6)
            errors = []
            stop = False

            def appender(td):
                try:
                    while not stop:
                        doc = cp.load_chain(td)
                        cp.save_chain(doc, td)  # empty deltas only
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def compactor(td):
                try:
                    while not stop:
                        cp.compact_chain(td)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def merger():
                try:
                    for _ in range(40):
                        cp.merge_chains(main, branch)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def verifier():
                try:
                    while not stop:
                        cp.verify_chain([main, branch])
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=appender, args=(main,)),
                threading.Thread(target=appender, args=(branch,)),
                threading.Thread(target=compactor, args=(main,)),
                threading.Thread(target=compactor, args=(branch,)),
                threading.Thread(target=merger),
                threading.Thread(target=verifier),
            ]
            for thread in threads:
                thread.start()
            # Let the racy operations churn for a bounded time, then ask
            # every worker to stop and wait for a clean shutdown.
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not errors:
                time.sleep(0.02)
            stop = True
            for thread in threads:
                thread.join(timeout=30)
                self.assertFalse(thread.is_alive(), "a merge/compact race deadlocked")
            self.assertEqual(errors, [])
            self.assertTrue(cp.verify_chain([main, branch]).ok)
            # Appenders only ever wrote empty deltas, so the final merge
            # makes the target load exactly to the source's state.
            cp.merge_chains(main, branch)
            self.assertEqual(_chain_state(branch), _chain_state(main))


class MergeFamilyVerificationTests(unittest.TestCase):
    def test_verify_after_merge_covers_all_members_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=1)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=1)
            cp.merge_chains(main, branch)
            # One call covers every member after the merge, read-only.
            def family_hash():
                digest_parts = []
                for td in (main, branch):
                    h = []
                    for name in sorted(os.listdir(td)):
                        p = os.path.join(td, name)
                        if os.path.isfile(p):
                            h.append((name, _read(p)))
                    digest_parts.append(tuple(h))
                return tuple(digest_parts)

            before = family_hash()
            report = cp.verify_chain([main, branch])
            self.assertTrue(report.ok)
            self.assertEqual(report.members, (main, branch))
            self.assertEqual(family_hash(), before)

    def test_corrupt_merge_tail_names_only_the_target_chain(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=1)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=1)
            cp.merge_chains(main, branch)
            merge_seg = int(_read(os.path.join(branch, "head")))
            # The merge segment belongs to the target alone (nlink 1).
            self.assertEqual(
                os.stat(os.path.join(branch, _seg(merge_seg))).st_nlink, 1
            )
            tail = os.path.join(branch, _seg(merge_seg))
            with open(tail, "r+b") as fh:
                fh.truncate(os.path.getsize(tail) // 2)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain([main, branch])
            message = str(ctx.exception)
            self.assertIn(f"segment {merge_seg}", message)
            self.assertIn("chain 1", message)
            self.assertIn(branch, message)
            # The source chain is not implicated and verifies on its own.
            self.assertNotIn("chain 0", message)
            self.assertTrue(cp.verify_chain(main).ok)


class SequentialMergeTests(unittest.TestCase):
    def test_sequential_merge_directories(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            seq, _ = _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            seq.fork(main, branch, up_to=1)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=1)
            self.assertIsNone(seq.merge(main, branch))
            self.assertEqual(_chain_state(branch), _chain_state(main))
            self.assertTrue(seq.verify([main, branch]).ok)


if __name__ == "__main__":
    unittest.main()
