"""Tests for branch deletion and deterministic reachability reclamation.

Deleting a branch chain removes its directory, its head and the delta
segments it alone owned; shared segments are reclaimed by reachability
only -- exactly when no chain's head can reach them any more.  Crash
residue (staging directories of killed forks/deletes, orphan segments no
head can reach, temp-file leftovers) is swept deterministically on the
next fork, compaction or deletion.  These tests cover:

* deletion semantics: exclusive segments reclaimed, shared segments
  kept while any chain references them, surviving chains always one
  complete state;
* the delete error taxonomy (FileNotFoundError / ValueError / OSError)
  and the guarantee that a rejected delete removes nothing;
* deterministic, idempotent reclamation of kill residue by fork,
  compaction and deletion (live staging is never disturbed);
* kill-at-any-point safety of the deletion itself and delete/compact
  interleavings on one family;
* family verification attributing a shared bad segment to every chain
  that reaches it.
"""

from __future__ import annotations

import errno
import hashlib
import multiprocessing
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from sequence_engine import MemoryChain, Sequential, Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _base_weights,
    _build,
    _SEG1,
    _SEG2,
    _total,
)

try:
    import fcntl
except ImportError:  # non-POSIX platforms
    fcntl = None

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


class DeleteBranchTests(unittest.TestCase):
    def test_delete_releases_only_the_branchs_references(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            # The branch grows two exclusive tail segments of its own.
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 2, hidden=bhidden, start=2)
            self.assertEqual(_read(os.path.join(branch, "head")), b"4")
            main_state = _chain_state(main)
            for index in range(3):
                self.assertEqual(
                    os.stat(os.path.join(main, _seg(index))).st_nlink, 2
                )

            cp.delete_chain(branch)

            self.assertFalse(os.path.exists(branch))
            # The main chain is bit for bit intact and now holds the only
            # references to the previously shared prefix.
            self.assertEqual(_chain_state(main), main_state)
            self.assertEqual(_read(os.path.join(main, "head")), b"4")
            for index in range(3):
                self.assertEqual(
                    os.stat(os.path.join(main, _seg(index))).st_nlink, 1
                )
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
            # The branch still reaches the whole shared prefix: it loads
            # to exactly the same complete state.
            self.assertEqual(_chain_state(branch), branch_state)
            for index in range(3):
                stat = os.stat(os.path.join(branch, _seg(index)))
                self.assertEqual(stat.st_nlink, 1)
            self.assertTrue(cp.verify_chain(branch).ok)
            # And it keeps evolving on its own.
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=3)
            self.assertEqual(_read(os.path.join(branch, "head")), b"3")
            self.assertTrue(cp.verify_chain(branch).ok)

    def test_delete_both_chains_reclaims_everything(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            cp.delete_chain(main)
            cp.delete_chain(branch)
            self.assertEqual(os.listdir(root), [])

    def test_delete_then_refork_to_the_same_name(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            cp.delete_chain(branch)
            cp.fork_chain(main, branch, up_to=2)
            self.assertEqual(
                _chain_state(branch),
                cp.build_bytes(cp.load_chain(main, up_to=2)),
            )
            self.assertTrue(cp.verify_chain([main, branch]).ok)

    def test_delete_waits_out_and_rolls_forward_compaction_markers(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            main_state = _chain_state(main)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=3)
            # A streaming compaction on the branch killed right after its
            # marker: the delete rolls it forward first, then removes the
            # chain.
            folded = cp.build_bytes(cp.load_chain(branch, up_to=2))
            cp._atomic_write(branch, cp._staged_name(0), folded)
            cp._atomic_write(branch, cp._COMPACT_MARKER, b'{"u":2}')
            cp.delete_chain(branch)
            self.assertFalse(os.path.exists(branch))
            # The surviving chain is untouched and complete.
            self.assertEqual(os.listdir(root), ["main"])
            self.assertEqual(_chain_state(main), main_state)
            self.assertTrue(cp.verify_chain(main).ok)


class DeleteErrorTests(unittest.TestCase):
    def test_missing_directory_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            before = _dir_hash(main)
            with self.assertRaises(FileNotFoundError):
                cp.delete_chain(os.path.join(root, "absent"))
            # A plain file at the path is not a chain directory either.
            fake = os.path.join(root, "file")
            with open(fake, "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(FileNotFoundError):
                cp.delete_chain(fake)
            self.assertEqual(_dir_hash(main), before)

    def test_already_deleted_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            cp.delete_chain(branch)
            with self.assertRaises(FileNotFoundError):
                cp.delete_chain(branch)
            # The failed second delete changed nothing in the family.
            self.assertTrue(cp.verify_chain(main).ok)

    def test_missing_head_is_valueerror_and_nothing_is_removed(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            os.unlink(os.path.join(branch, "head"))
            listing_before = sorted(os.listdir(branch))
            main_links_before = [
                os.stat(os.path.join(main, _seg(i))).st_nlink for i in range(3)
            ]
            with self.assertRaises(ValueError):
                cp.delete_chain(branch)
            # Not a single segment -- shared or not -- was removed.
            self.assertEqual(sorted(os.listdir(branch)), listing_before)
            self.assertEqual(
                [os.stat(os.path.join(main, _seg(i))).st_nlink for i in range(3)],
                main_links_before,
            )

    def test_corrupt_chain_is_valueerror_and_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=3)
            # Corrupt the branch's exclusive tail segment.
            tail = os.path.join(branch, _seg(3))
            with open(tail, "r+b") as fh:
                fh.truncate(os.path.getsize(tail) // 2)
            before = _dir_hash(branch)
            with self.assertRaises(ValueError):
                cp.delete_chain(branch)
            self.assertEqual(_dir_hash(branch), before)
            # The shared prefix still has both references.
            for index in range(3):
                self.assertEqual(
                    os.stat(os.path.join(main, _seg(index))).st_nlink, 2
                )

    def test_missing_segment_is_valueerror_and_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=3)
            # A hole in the branch's own tail: unparseable structure.
            os.unlink(os.path.join(branch, _seg(2)))
            listing_before = sorted(os.listdir(branch))
            with self.assertRaises(ValueError):
                cp.delete_chain(branch)
            self.assertEqual(sorted(os.listdir(branch)), listing_before)

    def test_unwritable_parent_is_oserror_and_chain_intact(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write into read-only directories")
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            before = _dir_hash(branch)
            os.chmod(root, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.delete_chain(branch)
            finally:
                os.chmod(root, 0o755)
            # The rename never happened: the chain is exactly as it was.
            self.assertEqual(_dir_hash(branch), before)
            self.assertEqual(
                [n for n in os.listdir(root) if n.startswith(".seqdel.tmp-")],
                [],
            )

    def test_interrupted_teardown_leaves_only_whole_files(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            main_state = _chain_state(main)
            shared_inode = os.stat(os.path.join(main, _seg(1))).st_ino

            # A full disk surfaces mid-teardown: the deletion committed
            # (the rename) but the clean-up fails with OSError.
            with mock.patch.object(
                cp.os, "unlink",
                side_effect=OSError(errno.ENOSPC, "simulated disk full"),
            ):
                with self.assertRaises(OSError) as ctx:
                    cp.delete_chain(branch)
                self.assertEqual(ctx.exception.errno, errno.ENOSPC)
            self.assertFalse(os.path.exists(branch))
            # The detached staging directory holds only whole files, and
            # the shared segment's bytes are still reachable through it.
            debris = [
                n for n in os.listdir(root) if n.startswith(".seqdel.tmp-")
            ]
            self.assertEqual(len(debris), 1)
            staging = os.path.join(root, debris[0])
            self.assertEqual(
                os.stat(os.path.join(staging, _seg(1))).st_ino, shared_inode
            )
            self.assertEqual(
                _read(os.path.join(staging, _seg(1))),
                _read(os.path.join(main, _seg(1))),
            )
            # The next family operation sweeps the residue
            # deterministically; the surviving chain never noticed.
            cp.compact_chain(main, up_to=0)  # a no-op fold still sweeps
            self.assertEqual(
                [n for n in os.listdir(root) if n.startswith(".seqdel.tmp-")],
                [],
            )
            self.assertEqual(_chain_state(main), main_state)
            self.assertEqual(os.stat(os.path.join(main, _seg(1))).st_nlink, 1)

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
                seq.delete(b"not-a-chain")
            with self.assertRaises(TypeError):
                seq.delete(123)


class DeterministicReclamationTests(unittest.TestCase):
    def test_orphan_segments_and_temp_residue_swept_on_compact(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            state_before = _chain_state(main)
            t_before = cp.load_chain(main)["optim"]["t"]
            # Crash residue: an orphan segment beyond the head and a
            # half-written commit temp file.
            orphan = os.path.join(main, _seg(9))
            with open(orphan, "wb") as fh:
                fh.write(_read(os.path.join(main, _seg(1))))
            tmp = os.path.join(main, ".seqckp.tmp-dead")
            with open(tmp, "wb") as fh:
                fh.write(b"half a segment")

            cp.compact_chain(main, up_to=0)  # a no-op fold still sweeps

            self.assertFalse(os.path.exists(orphan))
            self.assertFalse(os.path.exists(tmp))
            self.assertEqual(
                sorted(os.listdir(main)), ["head"] + [_seg(i) for i in range(4)]
            )
            # Reclamation changed no reachable state and no step count.
            self.assertEqual(_chain_state(main), state_before)
            self.assertEqual(cp.load_chain(main)["optim"]["t"], t_before)
            # Repeating the sweep is a deterministic no-op.
            cp.compact_chain(main, up_to=0)
            self.assertEqual(
                sorted(os.listdir(main)), ["head"] + [_seg(i) for i in range(4)]
            )

    def test_orphan_segments_swept_on_fork(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            state_before = _chain_state(main)
            orphan = os.path.join(main, _seg(7))
            with open(orphan, "wb") as fh:
                fh.write(b"orphan")
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            self.assertFalse(os.path.exists(orphan))
            self.assertEqual(_chain_state(main), state_before)
            self.assertTrue(cp.verify_chain([main, branch]).ok)

    def test_killed_staging_directories_swept_by_all_three_operations(self):
        for trigger in ("compact", "fork", "delete"):
            with self.subTest(trigger=trigger):
                with tempfile.TemporaryDirectory() as root:
                    main = os.path.join(root, "main")
                    _trained_dir(main, 3)
                    branch = os.path.join(root, "branch")
                    cp.fork_chain(main, branch, up_to=2)
                    # Debris of a killed fork and a killed delete, each
                    # holding a shared segment reference hostage.
                    fork_debris = os.path.join(root, ".seqfork.tmp-x-dead")
                    os.mkdir(fork_debris)
                    os.link(
                        os.path.join(main, _seg(0)),
                        os.path.join(fork_debris, _seg(0)),
                    )
                    del_debris = os.path.join(root, ".seqdel.tmp-y-dead")
                    os.mkdir(del_debris)
                    os.link(
                        os.path.join(main, _seg(1)),
                        os.path.join(del_debris, _seg(1)),
                    )
                    self.assertEqual(
                        os.stat(os.path.join(main, _seg(0))).st_nlink, 3
                    )
                    self.assertEqual(
                        os.stat(os.path.join(main, _seg(1))).st_nlink, 3
                    )

                    if trigger == "compact":
                        cp.compact_chain(main, up_to=0)
                    elif trigger == "fork":
                        cp.fork_chain(main, os.path.join(root, "b2"), up_to=1)
                    else:
                        cp.delete_chain(branch)

                    self.assertFalse(os.path.exists(fork_debris))
                    self.assertFalse(os.path.exists(del_debris))
                    # The hostage references were released deterministically:
                    # segment 0 is left referenced by main + branch (2), plus
                    # the new fork's own link (3), or by main alone once the
                    # branch is deleted (1).
                    expected = {"compact": 2, "fork": 3, "delete": 1}[trigger]
                    self.assertEqual(
                        os.stat(os.path.join(main, _seg(0))).st_nlink, expected
                    )
                    self.assertTrue(cp.verify_chain(main).ok)

    def test_live_fork_staging_is_never_swept(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            staging = os.path.join(root, ".seqfork.tmp-live-x")
            os.mkdir(staging)
            os.link(
                os.path.join(main, _seg(0)), os.path.join(staging, _seg(0))
            )
            # A staging registered as owned by a live fork is skipped.
            cp._fork_staging_register(staging)
            try:
                cp.compact_chain(main, up_to=0)
                self.assertTrue(os.path.isdir(staging))
            finally:
                cp._fork_staging_release(staging)
            # Once the owner is gone the same sweep reclaims it.
            cp.compact_chain(main, up_to=0)
            self.assertFalse(os.path.exists(staging))

    @unittest.skipIf(fcntl is None, "cross-process sentinel needs fcntl")
    def test_live_fork_staging_sentinel_is_respected_across_processes(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            staging = os.path.join(root, ".seqfork.tmp-live-y")
            os.mkdir(staging)
            sentinel = os.path.join(staging, ".seqstag.lock")
            fd = os.open(sentinel, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                # The flock marks a live owner even with no in-process
                # registration: the sweep leaves the staging alone.
                cp.compact_chain(main, up_to=0)
                self.assertTrue(os.path.isdir(staging))
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            cp.compact_chain(main, up_to=0)
            self.assertFalse(os.path.exists(staging))

    def test_delete_advances_no_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 5)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=3)
            doc_before = cp.load_chain(main)
            cp.delete_chain(branch)
            doc_after = cp.load_chain(main)
            self.assertEqual(doc_before["optim"]["t"], doc_after["optim"]["t"])
            self.assertEqual(
                cp.build_bytes(doc_before), cp.build_bytes(doc_after)
            )


class DeleteCrashTests(unittest.TestCase):
    def test_killed_delete_leaves_family_complete_and_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 30)
            main_state = _chain_state(main)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=20)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 10, hidden=bhidden, start=20)

            proc = multiprocessing.Process(
                target=_slow_delete_worker, args=(branch,)
            )
            proc.start()
            # Kill once the chain directory has been renamed aside, i.e.
            # mid-teardown.
            deadline = time.monotonic() + 30.0
            saw_staging = False
            while proc.is_alive() and time.monotonic() < deadline:
                if any(
                    name.startswith(".seqdel.tmp-") for name in os.listdir(root)
                ):
                    saw_staging = True
                    break
                time.sleep(0.001)
            if proc.is_alive():
                proc.kill()
            proc.join()
            self.assertTrue(saw_staging)

            # The surviving chain is one complete state, bit for bit.
            self.assertEqual(_chain_state(main), main_state)
            self.assertTrue(cp.verify_chain(main).ok)
            # The branch either never committed the deletion (complete
            # chain) or is gone; never a half chain at its path.
            if os.path.exists(branch):
                self.assertTrue(cp.verify_chain(branch).ok)
                cp.delete_chain(branch)
            # Whatever teardown residue the kill left is reclaimed
            # deterministically by the next family operation.
            cp.compact_chain(main)
            self.assertEqual(
                [n for n in os.listdir(root) if n.startswith(".seqdel.tmp-")],
                [],
            )
            self.assertEqual(sorted(os.listdir(root)), ["main"])
            self.assertEqual(_chain_state(main), main_state)


def _slow_delete_worker(directory):
    from sequence_engine import checkpoint as cp

    original_unlink = os.unlink

    def slow_unlink(path):
        time.sleep(0.02)
        return original_unlink(path)

    cp.os.unlink = slow_unlink
    try:
        cp.delete_chain(directory)
    finally:
        cp.os.unlink = original_unlink


class DeleteConcurrencyTests(unittest.TestCase):
    def test_delete_and_fork_while_compactions_and_saves_run(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 20)
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
            try:
                for i in range(6):
                    target = os.path.join(root, f"b{i}")
                    cp.fork_chain(main, target)
                    self.assertTrue(cp.verify_chain(target).ok)
                    cp.delete_chain(target)
                    self.assertFalse(os.path.exists(target))
            finally:
                stop = True
                for thread in threads:
                    thread.join()
            self.assertEqual(errors, [])
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertEqual(sorted(os.listdir(root)), ["main"])

    def test_loads_racing_delete_see_complete_state_or_missing_chain(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 10)
            target = os.path.join(root, "ephemeral")
            errors = []

            def churn():
                for i in range(20):
                    cp.fork_chain(main, target)
                    self.assertTrue(cp.verify_chain(target).ok)
                    cp.delete_chain(target)

            def reader():
                for _ in range(500):
                    try:
                        # A raced reader sees either one complete chain or
                        # a missing directory -- never a half/headless one.
                        cp.verify_chain(target)
                    except FileNotFoundError:
                        pass
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc)

            threads = [threading.Thread(target=churn)]
            threads += [threading.Thread(target=reader) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertTrue(cp.verify_chain(main).ok)

    def test_delete_interleaved_with_compaction_of_the_same_chain(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 20)
            main_state = _chain_state(main)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=10)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 5, hidden=bhidden, start=10)
            errors = []
            stop = False

            def compactor():
                try:
                    while not stop:
                        try:
                            cp.compact_chain(branch)
                        except FileNotFoundError:
                            return  # the deletion committed; expected
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            thread = threading.Thread(target=compactor)
            thread.start()
            cp.delete_chain(branch)
            stop = True
            thread.join()
            self.assertEqual(errors, [])
            self.assertFalse(os.path.exists(branch))
            # The shared segments the branch referenced were neither
            # wrongly deleted nor leaked: the survivor holds them all.
            self.assertEqual(_chain_state(main), main_state)
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertEqual(
                sorted(os.listdir(root)),
                ["main"],
            )


class FamilyAttributionTests(unittest.TestCase):
    def test_shared_segment_defect_names_every_referencing_chain(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            b1 = os.path.join(root, "b1")
            b2 = os.path.join(root, "b2")
            cp.fork_chain(main, b1, up_to=2)
            cp.fork_chain(main, b2, up_to=2)

            # Corrupt a segment all three chains share (one inode).
            shared = os.path.join(main, _seg(1))
            with open(shared, "r+b") as fh:
                fh.truncate(os.path.getsize(shared) // 2)
            before = (
                _dir_hash(main),
                _dir_hash(b1),
                _dir_hash(b2),
            )
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain([main, b1, b2])
            message = str(ctx.exception)
            self.assertIn("segment 1", message)
            for position, path in ((0, main), (1, b1), (2, b2)):
                self.assertIn(f"chain {position}", message)
                self.assertIn(path, message)
            # Verification wrote nothing anywhere.
            self.assertEqual((_dir_hash(main), _dir_hash(b1), _dir_hash(b2)), before)

    def test_attribution_covers_exactly_the_referencing_chains(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            b1 = os.path.join(root, "b1")
            cp.fork_chain(main, b1, up_to=2)  # reaches only segments 0..2
            b2 = os.path.join(root, "b2")
            cp.fork_chain(main, b2, up_to=4)  # shares the whole prefix

            # Corrupt a segment shared by main and b2 but beyond b1's head.
            shared = os.path.join(main, _seg(3))
            with open(shared, "r+b") as fh:
                fh.truncate(os.path.getsize(shared) // 2)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain([main, b1, b2])
            message = str(ctx.exception)
            self.assertIn("segment 3", message)
            self.assertIn("chain 0", message)
            self.assertIn(main, message)
            self.assertIn("chain 2", message)
            self.assertIn(b2, message)
            # b1 never reaches segment 3 and is not named.
            self.assertNotIn("chain 1", message)
            self.assertNotIn(b1, message)

    def test_branch_only_defect_names_only_the_branch(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=bhidden, start=2)

            tail = os.path.join(branch, _seg(3))
            with open(tail, "r+b") as fh:
                fh.truncate(os.path.getsize(tail) // 2)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain([main, branch])
            message = str(ctx.exception)
            self.assertIn("chain 1", message)
            self.assertIn(branch, message)
            self.assertIn("segment 3", message)
            # The main chain is not implicated and still verifies.
            self.assertNotIn("chain 0", message)
            self.assertTrue(cp.verify_chain(main).ok)

    def test_family_verify_after_delete(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            cp.delete_chain(branch)
            self.assertTrue(cp.verify_chain([main]).ok)
            with self.assertRaises(FileNotFoundError):
                cp.verify_chain([main, branch])


class SequentialDeleteTests(unittest.TestCase):
    def test_sequential_delete_directory(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            seq, _ = _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            seq.fork(main, branch, up_to=2)
            self.assertIsNone(seq.delete(branch))
            self.assertFalse(os.path.exists(branch))
            self.assertTrue(seq.verify(main).ok)
            with self.assertRaises(FileNotFoundError):
                seq.delete(branch)

    def test_sequential_delete_memory_chain(self):
        chain = MemoryChain()
        seq, _ = _stack()
        seq.save(chain)
        seq.update(_LR)
        seq.save(chain)
        branch = seq.fork(chain, up_to=1)
        self.assertIsNone(seq.delete(branch))
        self.assertEqual(len(branch), 0)
        with self.assertRaises(ValueError):
            seq.delete(branch)
        # The source chain is intact and the emptied chain is reusable.
        self.assertEqual(chain.read_head(), b"1")
        seq.save(branch)
        self.assertEqual(branch.read_head(), b"0")


if __name__ == "__main__":
    unittest.main()
