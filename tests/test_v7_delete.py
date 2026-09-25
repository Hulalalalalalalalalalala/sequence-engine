"""Tests for branch deletion and deterministic shared-segment reclamation.

Deleting a chain removes its directory, its ``head`` pointer and the
segments no other chain's head can still reach; shared segments are kept
exactly as they are while any chain references them.  These tests cover:

* the deletion itself (directory, head and exclusive segments gone;
  shared segments surviving through the remaining chains);
* the error taxonomy (FileNotFoundError / ValueError / OSError) and the
  guarantee that a rejected deletion touches nothing;
* deterministic, reachability-based reclamation of staging debris and
  orphan segments on the next fork, compaction or deletion;
* kill-at-any-point safety of a deletion interleaved with forks and
  compactions on the same family;
* family verification attributing a corrupt shared segment to every
  chain whose head reaches it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
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
        out, hidden = seq.forward(Tensor(_SEG1 if i % 2 == 0 else _SEG2), hidden)
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


def _dir_hash(td):
    digest = hashlib.sha256()
    for name in sorted(os.listdir(td)):
        path = os.path.join(td, name)
        if os.path.isfile(path):
            digest.update(name.encode("ascii"))
            with open(path, "rb") as fh:
                digest.update(fh.read())
    return digest.hexdigest()


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _nlink(path):
    return os.stat(path).st_nlink


class DeleteBasicsTests(unittest.TestCase):
    def test_delete_removes_directory_head_and_exclusive_segments(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 2, hidden=bhidden, start=2)  # segs 3, 4
            main_state = _chain_state(main)
            main_hash = _dir_hash(main)

            cp.delete_chain(branch)

            # The branch directory, its head and its exclusive deltas are
            # gone; the main chain is byte for byte untouched and still
            # reassembles its complete state.
            self.assertFalse(os.path.exists(branch))
            self.assertEqual(_dir_hash(main), main_hash)
            self.assertEqual(_chain_state(main), main_state)
            for index in range(3):
                self.assertEqual(_nlink(os.path.join(main, _seg(index))), 1)
            self.assertTrue(cp.verify_chain(main).ok)

    def test_delete_source_chain_keeps_branch_complete(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            branch_state = _chain_state(branch)

            cp.delete_chain(main)

            self.assertFalse(os.path.exists(main))
            # The shared prefix stays reachable through the branch: its
            # files are still there and the state is bit for bit intact.
            self.assertEqual(_chain_state(branch), branch_state)
            for index in range(3):
                path = os.path.join(branch, _seg(index))
                self.assertTrue(os.path.exists(path))
                self.assertEqual(_nlink(path), 1)
            self.assertTrue(cp.verify_chain(branch).ok)

    def test_shared_segments_survive_until_the_last_reference(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            b1 = os.path.join(root, "b1")
            b2 = os.path.join(root, "b2")
            cp.fork_chain(main, b1, up_to=2)
            cp.fork_chain(main, b2, up_to=2)
            b1_state = _chain_state(b1)
            b2_state = _chain_state(b2)

            cp.delete_chain(main)
            self.assertEqual(_nlink(os.path.join(b1, _seg(0))), 2)
            self.assertEqual(_chain_state(b1), b1_state)
            self.assertEqual(_chain_state(b2), b2_state)

            cp.delete_chain(b1)
            self.assertEqual(_nlink(os.path.join(b2, _seg(0))), 1)
            self.assertEqual(_chain_state(b2), b2_state)
            self.assertTrue(cp.verify_chain(b2).ok)

            cp.delete_chain(b2)
            self.assertEqual(os.listdir(root), [])

    def test_delete_allows_refork_to_the_same_name(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            cp.delete_chain(branch)
            self.assertFalse(os.path.exists(branch))
            # The name is free again: a fresh fork to it works.
            cp.fork_chain(main, branch, up_to=1)
            self.assertEqual(_read(os.path.join(branch, "head")), b"1")
            self.assertTrue(cp.verify_chain(branch).ok)

    def test_delete_and_reclaim_advance_no_optimizer_step(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 5)
            t_before = cp.load_chain(main)["optim"]["t"]
            self.assertGreater(t_before, 0)
            main_state = _chain_state(main)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=3)
            cp.delete_chain(branch)
            cp.compact_chain(main)
            doc = cp.load_chain(main)
            self.assertEqual(doc["optim"]["t"], t_before)
            self.assertEqual(cp.build_bytes(doc), main_state)

    def test_delete_rolls_forward_an_interrupted_compaction_first(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=3)
            main_state = _chain_state(main)
            # A streaming compaction on the branch killed right after its
            # marker: staged basis and marker on disk, owner dead.
            folded = cp.build_bytes(cp.load_chain(branch, up_to=2))
            cp._atomic_write(branch, cp._staged_name(0), folded)
            cp._atomic_write(branch, cp._COMPACT_MARKER, b'{"u":2}')
            cp.delete_chain(branch)
            # The branch is gone entirely; the main chain is untouched.
            self.assertFalse(os.path.exists(branch))
            self.assertEqual(_chain_state(main), main_state)
            self.assertTrue(cp.verify_chain(main).ok)


class DeleteErrorTests(unittest.TestCase):
    def test_missing_directory_is_filenotfound_and_changes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            before = (_dir_hash(main), _dir_hash(branch))
            with self.assertRaises(FileNotFoundError):
                cp.delete_chain(os.path.join(root, "absent"))
            self.assertEqual((_dir_hash(main), _dir_hash(branch)), before)

    def test_already_deleted_directory_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            cp.delete_chain(branch)
            with self.assertRaises(FileNotFoundError):
                cp.delete_chain(branch)
            with self.assertRaises(FileNotFoundError):
                cp.delete_chain(os.path.join(root, "never", "existed"))
            self.assertTrue(cp.verify_chain(main).ok)

    def test_missing_head_is_valueerror_and_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=1)
            os.unlink(os.path.join(branch, "head"))
            main_hash = _dir_hash(main)
            with self.assertRaises(ValueError):
                cp.delete_chain(branch)
            # The directory is still there and no shared segment's
            # reference was released.
            self.assertTrue(os.path.isdir(branch))
            self.assertEqual(
                sorted(os.listdir(branch)), [_seg(0), _seg(1)]
            )
            self.assertEqual(_nlink(os.path.join(main, _seg(0))), 2)
            self.assertEqual(_dir_hash(main), main_hash)

    def test_corrupt_head_is_valueerror_and_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=1)
            with open(os.path.join(branch, "head"), "wb") as fh:
                fh.write(b"not-a-number")
            with self.assertRaises(ValueError):
                cp.delete_chain(branch)
            self.assertTrue(os.path.isdir(branch))
            self.assertEqual(_nlink(os.path.join(main, _seg(0))), 2)

    def test_truncated_segment_is_valueerror_and_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            # Give the branch an exclusive tail segment, then corrupt it.
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=2)
            victim = os.path.join(branch, _seg(3))
            with open(victim, "r+b") as fh:
                fh.truncate(os.path.getsize(victim) // 2)
            branch_listing = sorted(os.listdir(branch))
            with self.assertRaises(ValueError):
                cp.delete_chain(branch)
            self.assertEqual(sorted(os.listdir(branch)), branch_listing)
            self.assertEqual(_nlink(os.path.join(main, _seg(0))), 2)
            self.assertTrue(cp.verify_chain(main).ok)

    def test_missing_segment_is_valueerror_and_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)  # shares segments 0..3
            os.unlink(os.path.join(branch, _seg(2)))
            with self.assertRaises(ValueError):
                cp.delete_chain(branch)
            # The remaining shared references were not released either.
            self.assertTrue(os.path.isdir(branch))
            self.assertEqual(_nlink(os.path.join(main, _seg(0))), 2)
            self.assertEqual(_nlink(os.path.join(main, _seg(1))), 2)
            self.assertEqual(_nlink(os.path.join(main, _seg(3))), 2)

    def test_delete_type_checks(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 1)
            with self.assertRaises(TypeError):
                cp.delete_chain(123)
            with self.assertRaises(TypeError):
                cp.delete_chain_memory(object())
            seq, _ = _stack()
            with self.assertRaises(TypeError):
                seq.delete_branch(bytearray())
            with self.assertRaises(TypeError):
                seq.delete(b"not-a-chain")

    def test_file_at_chain_path_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "file")
            with open(path, "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(FileNotFoundError):
                cp.delete_chain(path)

    def test_unwritable_parent_is_oserror_and_deletes_nothing(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write into read-only directories")
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            os.chmod(root, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.delete_chain(branch)
            finally:
                os.chmod(root, 0o755)
            self.assertTrue(cp.verify_chain(branch).ok)
            self.assertEqual(_nlink(os.path.join(main, _seg(0))), 2)


class ReclamationTests(unittest.TestCase):
    def _plant_orphans(self, td):
        """Debris of a kill: an uncommitted segment and a temp file."""
        orphan = os.path.join(td, _seg(99))
        with open(orphan, "wb") as fh:
            fh.write(b"orphaned segment bytes")
        tmp = os.path.join(td, ".seqckp.tmp-dead")
        with open(tmp, "wb") as fh:
            fh.write(b"torn temp file")
        return orphan, tmp

    def test_orphan_segments_reclaimed_on_compact(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            state = _chain_state(main)
            orphan, tmp = self._plant_orphans(main)
            cp.compact_chain(main, up_to=1)
            self.assertFalse(os.path.exists(orphan))
            self.assertFalse(os.path.exists(tmp))
            self.assertEqual(_chain_state(main), state)

    def test_orphan_segments_reclaimed_on_fork(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            orphan, tmp = self._plant_orphans(main)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            self.assertFalse(os.path.exists(orphan))
            self.assertFalse(os.path.exists(tmp))
            self.assertTrue(cp.verify_chain([main, branch]).ok)

    def test_verify_never_reclaims_orphans(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            orphan, tmp = self._plant_orphans(main)
            # Verification is strictly read-only: the chain verifies and
            # the debris is left exactly where it was.
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertTrue(os.path.exists(orphan))
            self.assertTrue(os.path.exists(tmp))

    def test_killed_delete_staging_reclaimed_by_next_delete(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            # A deletion killed after the rename-aside: the chain name is
            # gone, the staging debris holds the shared references.
            debris = os.path.join(root, ".seqdelete.tmp-branch-dead")
            os.rename(branch, debris)
            self.assertEqual(_nlink(os.path.join(main, _seg(0))), 2)
            with self.assertRaises(FileNotFoundError):
                cp.delete_chain(branch)
            # The failed delete still swept the debris deterministically.
            self.assertFalse(os.path.exists(debris))
            self.assertEqual(_nlink(os.path.join(main, _seg(0))), 1)
            self.assertTrue(cp.verify_chain(main).ok)

    def test_killed_delete_staging_reclaimed_by_next_fork(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            debris = os.path.join(root, ".seqdelete.tmp-branch-dead")
            os.rename(branch, debris)
            cp.fork_chain(main, branch, up_to=1)
            self.assertFalse(os.path.exists(debris))
            self.assertEqual(_read(os.path.join(branch, "head")), b"1")
            self.assertTrue(cp.verify_chain([main, branch]).ok)

    def test_killed_delete_staging_reclaimed_by_next_compact(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            debris = os.path.join(root, ".seqdelete.tmp-branch-dead")
            os.rename(branch, debris)
            with self.assertRaises(FileNotFoundError):
                cp.compact_chain(branch)
            self.assertFalse(os.path.exists(debris))
            self.assertEqual(_nlink(os.path.join(main, _seg(0))), 1)

    def test_killed_fork_staging_reclaimed_by_delete(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            # Debris of a killed fork to the same target name.
            fork_debris = os.path.join(root, ".seqfork.tmp-branch-dead")
            os.mkdir(fork_debris)
            os.link(
                os.path.join(main, _seg(0)),
                os.path.join(fork_debris, _seg(0)),
            )
            self.assertEqual(_nlink(os.path.join(main, _seg(0))), 3)
            cp.delete_chain(branch)
            self.assertFalse(os.path.exists(branch))
            self.assertFalse(os.path.exists(fork_debris))
            self.assertEqual(_nlink(os.path.join(main, _seg(0))), 1)

    def test_repeated_cleanup_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            state = _chain_state(main)
            cp.delete_chain(branch)
            snapshot = _dir_hash(main)
            # Repeating the reclamation changes nothing at all.
            with self.assertRaises(FileNotFoundError):
                cp.delete_chain(branch)
            cp.compact_chain(main, up_to=0)  # nothing to merge: no-op
            self.assertEqual(_dir_hash(main), snapshot)
            self.assertEqual(_chain_state(main), state)


class DeleteConcurrencyTests(unittest.TestCase):
    def test_delete_branch_while_main_compacts_appends_and_reads(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 20)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=10)
            errors = []
            stop = False

            def compactor():
                try:
                    while not stop:
                        cp.compact_chain(main)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def appender():
                try:
                    while not stop:
                        doc = cp.load_chain(main)
                        cp.save_chain(doc, main)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=compactor),
                threading.Thread(target=appender),
            ]
            for thread in threads:
                thread.start()
            cp.delete_chain(branch)
            stop = True
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertFalse(os.path.exists(branch))
            # The surviving chain is one complete, verifiable state.
            self.assertTrue(cp.verify_chain(main).ok)

    def test_forks_and_deletes_interleave_on_one_family(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 10)
            errors = []

            def fork_and_delete(offset):
                try:
                    for i in range(6):
                        target = os.path.join(root, f"b{i + offset}")
                        cp.fork_chain(main, target)
                        cp.delete_chain(target)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            # Distinct target names per thread to keep the race honest.
            threads = [
                threading.Thread(target=fork_and_delete, args=(0,)),
                threading.Thread(target=fork_and_delete, args=(100,)),
            ]
            for thread in threads:
                thread.start()
            for _ in range(3):
                cp.compact_chain(main)
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertEqual(
                sorted(os.listdir(root)), ["main"]
            )


class FamilyAttributionTests(unittest.TestCase):
    def _family(self, root, fork_at=2, main_steps=4):
        main = os.path.join(root, "main")
        _trained_dir(main, main_steps)
        branch = os.path.join(root, "branch")
        cp.fork_chain(main, branch, up_to=fork_at)
        bseq, _ = _stack()
        bhidden = bseq.load(branch)
        _train(bseq, branch, 1, hidden=bhidden, start=fork_at)
        return main, branch

    def test_shared_segment_damage_names_every_referencing_chain(self):
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._family(root)
            shared = os.path.join(main, _seg(1))
            with open(shared, "r+b") as fh:
                fh.truncate(os.path.getsize(shared) // 2)
            before = (_dir_hash(main), _dir_hash(branch))
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain([main, branch])
            message = str(ctx.exception)
            # The corrupt shared segment is attributed to both chains.
            self.assertIn("chain 0", message)
            self.assertIn("chain 1", message)
            self.assertIn(main, message)
            self.assertIn(branch, message)
            self.assertIn("segment 1", message)
            # Verification wrote nothing anywhere.
            self.assertEqual((_dir_hash(main), _dir_hash(branch)), before)

    def test_shared_attribution_regardless_of_member_order(self):
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._family(root)
            shared = os.path.join(branch, _seg(2))
            with open(shared, "r+b") as fh:
                fh.truncate(os.path.getsize(shared) // 2)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain([branch, main])
            message = str(ctx.exception)
            self.assertIn("chain 0", message)
            self.assertIn("chain 1", message)
            self.assertIn(main, message)
            self.assertIn(branch, message)
            self.assertIn("segment 2", message)

    def test_branch_only_damage_attributes_only_the_branch(self):
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._family(root)
            tail = os.path.join(branch, _seg(3))
            with open(tail, "r+b") as fh:
                fh.truncate(os.path.getsize(tail) // 2)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain([main, branch])
            message = str(ctx.exception)
            self.assertIn("chain 1", message)
            self.assertIn(branch, message)
            self.assertNotIn("shared with", message)


class MemoryDeleteTests(unittest.TestCase):
    def _memory_chain(self, steps):
        chain = MemoryChain()
        seq, _ = _stack()
        seq.save(chain)
        hidden = None
        for i in range(steps):
            out, hidden = seq.forward(
                Tensor(_SEG1 if i % 2 == 0 else _SEG2), hidden
            )
            seq.backward(_total(out))
            seq.update(_LR)
            seq.save(chain)
        return chain

    def test_memory_delete_releases_only_own_references(self):
        chain = self._memory_chain(3)
        main_state = cp.build_bytes(cp.load_chain_memory(chain))
        branch = cp.fork_chain_memory(chain, up_to=2)
        cp.delete_chain_memory(branch)
        self.assertEqual(len(branch), 0)
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(chain)), main_state
        )
        # Deleting the source keeps the branch's shared prefix reachable.
        chain2 = self._memory_chain(2)
        branch2 = cp.fork_chain_memory(chain2)
        branch2_state = cp.build_bytes(cp.load_chain_memory(branch2))
        cp.delete_chain_memory(chain2)
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(branch2)), branch2_state
        )
        self.assertTrue(cp.verify_chain_memory(branch2).ok)

    def test_memory_delete_errors(self):
        chain = self._memory_chain(2)
        cp.delete_chain_memory(chain)
        with self.assertRaises(ValueError):
            cp.delete_chain_memory(chain)  # already deleted
        with self.assertRaises(ValueError):
            cp.delete_chain_memory(MemoryChain())  # never committed
        corrupt = self._memory_chain(2)
        raw = corrupt.read_segment(cp._segment_name(1))
        corrupt.write_segment(cp._segment_name(1), raw[: len(raw) // 2])
        with self.assertRaises(ValueError):
            cp.delete_chain_memory(corrupt)
        self.assertEqual(len(corrupt), 3)  # nothing dropped
        with self.assertRaises(TypeError):
            cp.delete_chain_memory(object())


class SequentialDeleteTests(unittest.TestCase):
    def test_sequential_delete_branch_directory(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            seq, _ = _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            seq.fork(main, branch, up_to=2)
            self.assertIsNone(seq.delete_branch(branch))
            self.assertFalse(os.path.exists(branch))
            self.assertTrue(seq.verify(main).ok)

    def test_sequential_delete_alias_and_memory_dispatch(self):
        seq, _ = _stack()
        chain = MemoryChain()
        seq.save(chain)
        seq.update(_LR)
        seq.save(chain)
        branch = seq.fork(chain)
        seq.delete(branch)
        self.assertEqual(len(branch), 0)
        seq.delete_branch(chain)
        self.assertEqual(len(chain), 0)


if __name__ == "__main__":
    unittest.main()
