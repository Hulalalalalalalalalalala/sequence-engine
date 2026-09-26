"""Tests for family-level compaction.

``compact_family`` folds the prefix every member of a forked chain
family shares into one new basis segment in a single call.  These tests
cover:

* the fold mechanics: every member's head state is bit for bit
  preserved, each member's segment count drops by exactly the common
  folded range and each member's exclusive tail follows the new basis
  unchanged;
* storage: the folded basis is physically stored once (one inode hard
  linked across every member directory) and the old shared prefix is
  reclaimed by reachability;
* determinism: repeating the fold is a no-op and a family without a
  foldable common delta is left untouched;
* no optimizer step advances, for both ``t = 0`` and stepped chains;
* crash safety: a family fold killed at any point leaves every member
  loadable as one complete state and the next open/family operation
  rolls every member forward;
* the error taxonomy (FileNotFoundError / ValueError / OSError) with
  the guarantee that a rejected fold moves not one member byte;
* concurrency with saves, loads, forks, deletes, merges and
  single-chain compactions;
* read-only family verification after the fold.
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


def _train(seq, td, steps, hidden=None, start=0, flip=False):
    """Run *steps* deterministic train/save cycles numbered from *start*.

    *flip* swaps the input pattern, so a branch trained with it diverges
    from the main chain's trajectory on its very first own segment.
    """
    for i in range(start, start + steps):
        use_seg1 = (i % 2 != 0) if flip else (i % 2 == 0)
        out, hidden = seq.forward(Tensor(_SEG1 if use_seg1 else _SEG2), hidden)
        seq.backward(_total(out))
        if i % 3 == 2:
            seq.adam_step(_ADAM_LR)
        else:
            seq.update(_LR)
        seq.save(td)
    return hidden


def _chain_state(td):
    return cp.build_bytes(cp.load_chain(td))


def _head(td):
    with open(os.path.join(td, "head"), "rb") as fh:
        return int(fh.read())


def _seg_count(td):
    return len([n for n in os.listdir(td) if n.endswith(".seqd")])


def _listing(td):
    listing = {}
    for name in os.listdir(td):
        path = os.path.join(td, name)
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                listing[name] = fh.read()
    return listing


def _build_main(root, steps, name="main"):
    main = os.path.join(root, name)
    os.mkdir(main)
    seq, _ = _stack()
    seq.save(main)
    _train(seq, main, steps)
    return main, seq


def _build_branch(root, main, point, steps, name=None, flip=True):
    branch = os.path.join(root, name if name is not None else f"b{point}")
    cp.fork_chain(main, branch, up_to=point)
    bseq, _ = _stack()
    hidden = bseq.load(branch)
    _train(bseq, branch, steps, hidden=hidden, start=point, flip=flip)
    return branch


def _common_fold(members):
    """The longest byte-identical segment prefix across *members*."""
    heads = [_head(td) for td in members]
    fold = 0
    for index in range(1, min(heads) + 1):
        raws = []
        for td in members:
            with open(os.path.join(td, _seg(index)), "rb") as fh:
                raws.append(fh.read())
        if any(raw != raws[0] for raw in raws[1:]):
            break
        fold = index
    return fold


def _flatten(tree):
    if isinstance(tree, list):
        for item in tree:
            yield from _flatten(item)
    else:
        yield tree


class FamilyCompactionSemanticsTests(unittest.TestCase):
    def test_folds_common_prefix_and_preserves_every_state(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 8)
            b4 = _build_branch(root, main, 4, 2)
            b2 = _build_branch(root, main, 2, 2)
            members = [main, b4, b2]
            states = {td: _chain_state(td) for td in members}
            heads_before = {td: _head(td) for td in members}
            fold = _common_fold(members)
            self.assertEqual(fold, 2)  # b2 diverges at its seg 3
            cp.compact_family(members)
            for td in members:
                self.assertEqual(_chain_state(td), states[td])
                self.assertEqual(_head(td), heads_before[td] - fold)
                self.assertTrue(cp.verify_chain(td).ok)

    def test_segment_counts_drop_by_exactly_the_fold(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 8)
            # All branches fork at >= 3 and diverge on their first own
            # segment, so the common delta prefix ends at segment 3.
            branches = [
                _build_branch(root, main, 5, 2, name="b5"),
                _build_branch(root, main, 3, 2, name="b3"),
                _build_branch(root, main, 6, 2, name="b6"),
            ]
            members = [main] + branches
            heads_before = {td: _head(td) for td in members}
            states = {td: _chain_state(td) for td in members}
            self.assertEqual(_common_fold(members), 3)
            cp.compact_family(members)
            for td in members:
                self.assertEqual(_head(td), heads_before[td] - 3)
                self.assertEqual(_seg_count(td), heads_before[td] - 3 + 1)
                self.assertEqual(_chain_state(td), states[td])

    def test_folded_basis_is_stored_once_and_hard_linked(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 6)
            b4 = _build_branch(root, main, 4, 1, name="b4")
            b3 = _build_branch(root, main, 3, 1, name="b3")
            members = [main, b4, b3]
            cp.compact_family(members)
            basis = os.path.join(main, _seg(0))
            for td in members[1:]:
                self.assertTrue(
                    os.path.samefile(basis, os.path.join(td, _seg(0)))
                )
            self.assertEqual(os.stat(basis).st_nlink, len(members))

    def test_member_exclusive_tails_follow_the_basis_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 6)
            # Two branches fork at the same point 4, then diverge
            # differently from each other.
            x = _build_branch(root, main, 4, 2, name="x", flip=True)
            y = os.path.join(root, "y")
            cp.fork_chain(main, y, up_to=4)
            yseq, _ = _stack()
            yh = yseq.load(y)
            _train(yseq, y, 3, hidden=yh, start=4, flip=False)
            members = [main, x, y]
            states = {td: _chain_state(td) for td in members}
            main_head_before = _head(main)
            cp.compact_family(members)
            for td in members:
                self.assertEqual(_chain_state(td), states[td])
            # The two branch first tails differ (different inodes/files).
            self.assertFalse(
                os.path.samefile(
                    os.path.join(x, _seg(1)), os.path.join(y, _seg(1))
                )
            )
            self.assertEqual(_head(main), main_head_before - 4)

    def test_repeating_the_fold_is_a_noop(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 6)
            b4 = _build_branch(root, main, 4, 1, name="b4")
            b3 = _build_branch(root, main, 3, 1, name="b3")
            members = [main, b4, b3]
            cp.compact_family(members)
            states = {td: _chain_state(td) for td in members}
            listings = {td: _listing(td) for td in members}
            heads = {td: _head(td) for td in members}
            cp.compact_family(members)
            cp.compact_family(members)
            for td in members:
                self.assertEqual(_listing(td), listings[td])
                self.assertEqual(_head(td), heads[td])
                self.assertEqual(_chain_state(td), states[td])

    def test_no_foldable_delta_prefix_leaves_everything_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 4)
            # Fork at the basis and diverge immediately: no delta prefix
            # is common, so the family must not change at all.
            only = _build_branch(root, main, 0, 2, name="only")
            members = [main, only]
            snapshots = {td: _listing(td) for td in members}
            states = {td: _chain_state(td) for td in members}
            self.assertEqual(_common_fold(members), 0)
            cp.compact_family(members)
            for td in members:
                self.assertEqual(_listing(td), snapshots[td])
                self.assertEqual(_chain_state(td), states[td])

    def test_basis_only_family_is_a_noop(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            os.mkdir(main)
            seq, _ = _stack()
            seq.save(main)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=0)
            before = (_listing(main), _listing(branch))
            cp.compact_family([main, branch])
            self.assertEqual((_listing(main), _listing(branch)), before)

    def test_no_optimizer_step_advances(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 5)
            b3 = _build_branch(root, main, 3, 2, name="b3")
            b2 = _build_branch(root, main, 2, 1, name="b2")
            members = [main, b3, b2]
            steps_before = {td: cp.load_chain(td)["optim"]["t"] for td in members}
            cp.compact_family(members)
            steps_after = {td: cp.load_chain(td)["optim"]["t"] for td in members}
            self.assertEqual(steps_after, steps_before)
            self.assertTrue(all(v > 0 for v in steps_after.values()))

    def test_unstepped_t_zero_chains_fold_keeping_zero_moments(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            os.mkdir(main)
            seq, _ = _stack()
            seq.save(main)
            out, _ = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out))
            seq.update(_LR)
            seq.save(main)
            seq.save(main)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=1)
            state = _chain_state(main)
            cp.compact_family([main, branch])
            doc = cp.load_chain(main)
            self.assertEqual(doc["optim"]["t"], 0)
            self.assertTrue(
                all(
                    leaf == 0.0
                    for entry in doc["optim"]["m"] + doc["optim"]["v"]
                    for leaf in _flatten(entry["v"])
                )
            )
            self.assertEqual(_chain_state(main), state)

    def test_append_after_fold_extends_the_chain_normally(self):
        with tempfile.TemporaryDirectory() as root:
            main, seq = _build_main(root, 6)
            b4 = _build_branch(root, main, 4, 1, name="b4")
            b3 = _build_branch(root, main, 3, 1, name="b3")
            members = [main, b4, b3]
            cp.compact_family(members)
            head_after = _head(main)
            hidden = seq.load(main)
            _train(seq, main, 2, hidden=hidden, start=20)
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertEqual(_head(main), head_after + 2)


class FamilyCompactionSharingTests(unittest.TestCase):
    def test_old_shared_prefix_is_reclaimed_by_reachability(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 6)
            branch = _build_branch(root, main, 4, 1)
            members = [main, branch]
            cp.compact_family(members)
            for td in members:
                names = {n for n in os.listdir(td) if n.endswith(".seqd")}
                self.assertEqual(names, {_seg(i) for i in range(_head(td) + 1)})
            self.assertEqual(os.stat(os.path.join(main, _seg(0))).st_nlink, 2)

    def test_deleting_a_member_after_fold_reclaims_its_links(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 5)
            branch = _build_branch(root, main, 3, 1)
            main_state = _chain_state(main)
            cp.compact_family([main, branch])
            cp.delete_chain(branch)
            self.assertEqual(_chain_state(main), main_state)
            self.assertEqual(os.stat(os.path.join(main, _seg(0))).st_nlink, 1)
            self.assertFalse(os.path.exists(branch))


class FamilyCompactionErrorTests(unittest.TestCase):
    def _sound_family(self, root, main_steps=4, point=3, branch_steps=1):
        main, _ = _build_main(root, main_steps)
        branch = _build_branch(root, main, point, branch_steps)
        return main, branch

    def test_missing_member_directory_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 2)
            missing = os.path.join(root, "nope")
            snapshot = _listing(main)
            with self.assertRaises(FileNotFoundError):
                cp.compact_family([main, missing])
            self.assertEqual(_listing(main), snapshot)

    def test_missing_referenced_segment_is_filenotfound_and_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._sound_family(root)
            os.unlink(os.path.join(main, _seg(2)))
            snapshots = {td: _listing(td) for td in (main, branch)}
            with self.assertRaises(FileNotFoundError):
                cp.compact_family([main, branch])
            for td in (main, branch):
                self.assertEqual(_listing(td), snapshots[td])

    def test_truncated_shared_segment_rejects_the_fold_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._sound_family(root)
            target = os.path.join(branch, _seg(2))  # hard-linked to main
            raw = open(target, "rb").read()
            with open(target, "r+b") as fh:
                fh.truncate(len(raw) // 2)
            snapshots = {td: set(os.listdir(td)) for td in (main, branch)}
            with self.assertRaises(ValueError):
                cp.compact_family([main, branch])
            for td in (main, branch):
                self.assertEqual(set(os.listdir(td)), snapshots[td])

    def test_parameter_shape_mismatch_is_valueerror_naming_shapes(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 2)
            other_weights = _base_weights()
            other_weights["b1"] = [0.01, -0.02]  # shape [2] instead of [3]
            other = os.path.join(root, "other")
            os.mkdir(other)
            oseq, _ = _stack(other_weights)
            oseq.save(other)
            snapshots = {td: _listing(td) for td in (main, other)}
            with self.assertRaises(ValueError) as ctx:
                cp.compact_family([main, other])
            message = str(ctx.exception)
            self.assertIn("parameter shapes", message)
            self.assertIn("do not match", message)
            for td in (main, other):
                self.assertEqual(_listing(td), snapshots[td])

    def test_layer_order_mismatch_is_valueerror_naming_layers(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 1)
            doc = cp.load_chain(main)
            for layer in doc["layers"]:
                layer["kind"] = "SomethingElse"
            other = os.path.join(root, "other")
            os.mkdir(other)
            with open(os.path.join(other, _seg(0)), "wb") as fh:
                fh.write(cp.build_bytes(doc))
            with open(os.path.join(other, "head"), "wb") as fh:
                fh.write(b"0")
            snapshot = _listing(main)
            with self.assertRaises(ValueError) as ctx:
                cp.compact_family([main, other])
            message = str(ctx.exception)
            self.assertIn("layer order", message)
            self.assertIn("SomethingElse", message)
            self.assertEqual(_listing(main), snapshot)

    def test_duplicate_member_and_bad_arguments(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 1)
            with self.assertRaises(ValueError):
                cp.compact_family([main, main])
            with self.assertRaises(ValueError):
                cp.compact_family([])
            with self.assertRaises(TypeError):
                cp.compact_family(main)
            with self.assertRaises(TypeError):
                cp.compact_family([main, 3])

    def test_unwritable_member_is_oserror_and_family_intact(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write into read-only directories")
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._sound_family(root)
            states = {td: _chain_state(td) for td in (main, branch)}
            os.chmod(branch, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.compact_family([main, branch])
            finally:
                os.chmod(branch, 0o755)
            for td in (main, branch):
                self.assertEqual(_chain_state(td), states[td])
                self.assertTrue(cp.verify_chain(td).ok)

    def test_disk_full_during_arming_is_oserror_and_a_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._sound_family(root)
            states = {td: _chain_state(td) for td in (main, branch)}
            with mock.patch.object(
                cp.os,
                "link",
                side_effect=OSError(errno.ENOSPC, "simulated disk full"),
            ):
                with self.assertRaises(OSError) as ctx:
                    cp.compact_family([main, branch])
                self.assertEqual(ctx.exception.errno, errno.ENOSPC)
            for td in (main, branch):
                names = os.listdir(td)
                self.assertNotIn(cp._COMPACT_MARKER, names)
                self.assertNotIn(cp._LEASE_NAME, names)
                self.assertFalse(
                    any(n.startswith(cp._STAGED_PREFIX) for n in names)
                )
                self.assertEqual(_chain_state(td), states[td])
            cp.compact_family([main, branch])
            for td in (main, branch):
                self.assertEqual(_chain_state(td), states[td])


def _family_worker(member_dirs):
    from sequence_engine import checkpoint as cp

    cp.compact_family(member_dirs)


class FamilyCompactionCrashTests(unittest.TestCase):
    def _kill_after(self, proc, condition, timeout=30.0):
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

    def _marker_seen(self, members):
        return any(
            os.path.exists(os.path.join(td, cp._COMPACT_MARKER)) for td in members
        )

    def _staged_seen(self, members):
        return any(
            any(n.startswith(cp._STAGED_PREFIX) for n in os.listdir(td))
            for td in members
            if os.path.isdir(td)
        )

    def _crash_family(self, root, main_steps=30, points=(20, 10)):
        main, _ = _build_main(root, main_steps)
        branches = []
        for point in points:
            branches.append(_build_branch(root, main, point, 2, name=f"b{point}"))
        members = [main] + branches
        states = {td: _chain_state(td) for td in members}
        heads = {td: _head(td) for td in members}
        return members, states, heads

    def test_kill_before_markers_leaves_complete_chains(self):
        with tempfile.TemporaryDirectory() as root:
            members, states, heads = self._crash_family(root)
            proc = multiprocessing.Process(target=_family_worker, args=(members,))
            proc.start()
            self._kill_after(proc, lambda: self._staged_seen(members))
            fold = min(heads.values()) - 2  # branches added 2 segments
            for td in members:
                self.assertEqual(_chain_state(td), states[td])
                self.assertIn(_head(td), (heads[td], heads[td] - fold))

    def test_kill_after_markers_every_member_recovers_complete(self):
        with tempfile.TemporaryDirectory() as root:
            members, states, heads = self._crash_family(root)
            proc = multiprocessing.Process(target=_family_worker, args=(members,))
            proc.start()
            self._kill_after(proc, lambda: self._marker_seen(members))
            for td in members:
                self.assertEqual(_chain_state(td), states[td])
                names = os.listdir(td)
                self.assertFalse(
                    n_marker_or_staged(names)
                )

    def test_family_operation_after_kill_finishes_the_fold(self):
        with tempfile.TemporaryDirectory() as root:
            members, states, heads = self._crash_family(root, 24, (15, 8))
            proc = multiprocessing.Process(target=_family_worker, args=(members,))
            proc.start()
            self._kill_after(proc, lambda: self._marker_seen(members))
            cp.compact_family(members)
            for td in members:
                self.assertEqual(_chain_state(td), states[td])
            # The fold is complete and now idempotent.
            listings = {td: set(os.listdir(td)) for td in members}
            cp.compact_family(members)
            for td in members:
                self.assertEqual(set(os.listdir(td)), listings[td])
                self.assertEqual(_chain_state(td), states[td])


def n_marker_or_staged(names):
    return any(
        n == cp._COMPACT_MARKER or n.startswith(cp._STAGED_PREFIX) for n in names
    )


class FamilyCompactionConcurrencyTests(unittest.TestCase):
    def test_saves_and_loads_flow_during_a_family_fold(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 10)
            branches = [
                _build_branch(root, main, 6, 1, name="b6"),
                _build_branch(root, main, 3, 1, name="b3"),
            ]
            members = [main] + branches
            errors = []
            stop = False

            def appender():
                try:
                    while not stop:
                        doc = cp.load_chain(main)
                        cp.save_chain(doc, main)  # empty delta appends
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def reader():
                try:
                    while not stop:
                        for td in members:
                            cp.build_bytes(cp.load_chain(td))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=appender)]
            threads += [threading.Thread(target=reader) for _ in range(2)]
            for thread in threads:
                thread.start()
            for _ in range(10):
                cp.compact_family(members)
            stop = True
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            for td in members:
                self.assertTrue(cp.verify_chain(td).ok)

    def test_family_fold_serialises_with_single_chain_compactions(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 12)
            branch = _build_branch(root, main, 6, 2)
            members = [main, branch]
            errors = []

            def family_folder():
                try:
                    for _ in range(8):
                        cp.compact_family(members)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def single_folder():
                try:
                    for _ in range(8):
                        cp.compact_chain(main)
                        cp.compact_chain(branch)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=family_folder),
                threading.Thread(target=single_folder),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertTrue(cp.verify_chain(members).ok)
            for td in members:
                cp.load_chain(td)

    def test_fork_merge_and_delete_work_around_a_family_fold(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 8)
            branch = _build_branch(root, main, 4, 2)
            branch_state = _chain_state(branch)
            cp.compact_family([main, branch])
            child = os.path.join(root, "child")
            cp.fork_chain(branch, child)
            cp.merge_chains(child, main)
            self.assertTrue(cp.verify_chain([main, branch, child]).ok)
            cp.delete_chain(child)
            self.assertFalse(os.path.exists(child))
            # The branch member itself was never touched by the merge.
            self.assertEqual(_chain_state(branch), branch_state)


class FamilyCompactionVerificationTests(unittest.TestCase):
    def test_family_verify_covers_every_folded_member_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            main, _ = _build_main(root, 6)
            branches = [
                _build_branch(root, main, 4, 1, name="b4"),
                _build_branch(root, main, 2, 1, name="b2"),
            ]
            members = [main] + branches
            cp.compact_family(members)
            after = {td: set(os.listdir(td)) for td in members}
            report = cp.verify_chain(members)
            self.assertTrue(report.ok)
            self.assertEqual(len(report.chains), len(members))
            # Strictly read-only: listing unchanged after verification.
            for td in members:
                self.assertEqual(set(os.listdir(td)), after[td])

    def test_sequential_entry_point_folds_directories_and_memory(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            os.mkdir(main)
            seq.save(main)
            _train(seq, main, 4)
            branch = os.path.join(root, "branch")
            seq.fork(main, branch, up_to=2)
            bseq, _ = _stack()
            hidden = bseq.load(branch)
            _train(bseq, branch, 1, hidden=hidden, start=2, flip=True)
            states = {td: _chain_state(td) for td in (main, branch)}
            seq.compact_family([main, branch])
            for td in (main, branch):
                self.assertEqual(_chain_state(td), states[td])
            # Memory backend through the same entry point.
            c0 = cp.MemoryChain()
            s2, _ = _stack()
            s2.save(c0)
            _train(s2, c0, 3)
            c1 = s2.fork(c0, up_to=1)
            s3, _ = _stack()
            h3 = s3.load(c1)
            _train(s3, c1, 1, hidden=h3, start=1, flip=True)
            mem_states = [
                cp.build_bytes(cp.load_chain_memory(c)) for c in (c0, c1)
            ]
            s2.compact_family([c0, c1])
            self.assertEqual(
                [cp.build_bytes(cp.load_chain_memory(c)) for c in (c0, c1)],
                mem_states,
            )
            with self.assertRaises(TypeError):
                seq.compact_family(main)
            with self.assertRaises(TypeError):
                seq.compact_family([main, c0])


if __name__ == "__main__":
    unittest.main()
