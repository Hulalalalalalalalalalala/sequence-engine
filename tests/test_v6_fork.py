"""Tests for chain families: deriving forked chains that share their
prefix segments, independent evolution of sibling chains, reachability
based reclamation of shared segments and family-aware read-only
verification.

These complement the single-chain tests in ``test_v4_compact.py`` and
``test_v5_streaming.py`` by exercising the new guarantees directly:

* ``derive_chain`` / ``Sequential.derive`` fork a chain directory at any
  committed segment, sharing (hard-linking, never copying) the segments
  up to the fork point;
* both chains then save, load, stream-compact and verify in parallel
  without interfering, and each reassembles bit for bit the state an
  unforked chain with the same saves would hold;
* compacting one chain or dropping another reclaims a shared segment
  exactly when no remaining chain's head can reach it;
* ``verify_chain`` on a family member names the chain a bad segment
  belongs to and never writes a byte.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import unittest

from sequence_engine import MemoryChain, Sequential, Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import _base_weights, _build, _SEG1, _SEG2, _total

_LR = 0.1
_ADAM_LR = 0.05


def _stack(weights=None):
    return _build(weights if weights is not None else _base_weights())


def _seg(i):
    return f"seg-{i:010d}.seqd"


def _trained_dir(td, steps):
    seq, _ = _stack()
    seq.save(td)
    hidden = None
    for _ in range(steps):
        out, hidden = seq.forward(Tensor(_SEG1), hidden)
        seq.backward(_total(out))
        seq.update(_LR)
        seq.save(td)
    return seq, hidden


def _state(td, up_to=None):
    return cp.build_bytes(cp.load_chain(td, up_to=up_to))


def _chain_hash(td):
    digest = hashlib.sha256()
    for name in sorted(os.listdir(td)):
        path = os.path.join(td, name)
        if os.path.isfile(path):
            digest.update(name.encode("ascii"))
            with open(path, "rb") as fh:
                digest.update(fh.read())
    return digest.hexdigest()


def _head_value(td):
    with open(os.path.join(td, "head"), "rb") as fh:
        return int(fh.read())


def _seg_names(td):
    return sorted(name for name in os.listdir(td) if name.startswith("seg-"))


class DeriveTests(unittest.TestCase):
    def test_derive_shares_prefix_segments_and_starts_a_new_head(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 5)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=2)

            self.assertEqual(_head_value(branch), 2)
            self.assertEqual(_seg_names(branch), [_seg(i) for i in range(3)])
            # The prefix segments are shared, not copied: same inode.
            for i in range(3):
                self.assertEqual(
                    os.stat(os.path.join(main, _seg(i))).st_ino,
                    os.stat(os.path.join(branch, _seg(i))).st_ino,
                )
            # The branch reassembles exactly the state at the fork point.
            self.assertEqual(_state(branch), _state(main, up_to=2))
            # The source chain is undisturbed.
            self.assertEqual(_head_value(main), 5)
            self.assertEqual(_seg_names(main), [_seg(i) for i in range(6)])

    def test_derive_defaults_to_the_source_head(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 3)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch)
            self.assertEqual(_head_value(branch), 3)
            self.assertEqual(_state(branch), _state(main))

    def test_derive_at_the_basis_shares_only_the_basis(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 4)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=0)
            self.assertEqual(_seg_names(branch), [_seg(0)])
            self.assertEqual(_state(branch), _state(main, up_to=0))

    def test_derive_publishes_atomically_and_leaves_no_debris(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 4)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=3)
            self.assertEqual(
                sorted(os.listdir(parent)),
                ["branch"],
            )
            self.assertFalse(
                any(
                    name.startswith(".seqderive.tmp-")
                    for name in os.listdir(parent)
                )
            )
            self.assertFalse(
                any(
                    name.startswith((".seqc-", ".seqcompact", ".seqckp.tmp-"))
                    for name in os.listdir(main)
                )
            )

    def test_multi_level_family(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 6)
            branch = os.path.join(parent, "branch")
            grand = os.path.join(parent, "grand")
            cp.derive_chain(main, branch, at=4)
            cp.derive_chain(branch, grand, at=2)
            self.assertEqual(_state(grand), _state(main, up_to=2))
            self.assertEqual(_state(branch), _state(main, up_to=4))
            # All three verify as one family.
            for td in (main, branch, grand):
                self.assertTrue(cp.verify_chain(td).ok)

    def test_derive_rolls_forward_an_interrupted_source_compaction(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 5)
            expected = _state(main)
            # Stage an interrupted streaming fold at segment 2 (dead
            # owner: no lease is held), then derive -- the open rolls the
            # fold forward and the fork sees the compacted chain.
            folded = cp.build_bytes(cp.load_chain(main, up_to=2))
            with open(os.path.join(main, ".seqc-" + _seg(0)), "wb") as fh:
                fh.write(folded)
            with open(os.path.join(main, ".seqcompact"), "wb") as fh:
                fh.write(b'{"u":2}')
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch)
            self.assertEqual(_state(main), expected)
            self.assertEqual(_head_value(main), 3)
            self.assertEqual(_state(branch), expected)
            self.assertEqual(_head_value(branch), 3)

    def test_derive_from_a_read_only_source(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 3)
            branch = os.path.join(parent, "branch")
            os.chmod(main, 0o555)
            try:
                cp.derive_chain(main, branch, at=2)
            finally:
                os.chmod(main, 0o755)
            self.assertEqual(_state(branch), _state(main, up_to=2))


class FamilyEvolutionTests(unittest.TestCase):
    def test_identical_continuations_match_the_unforked_run_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            seq, hidden = _trained_dir(main, 3)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch)

            seq2, _ = _stack()
            hidden2 = seq2.load(branch)
            for _ in range(4):
                out, hidden = seq.forward(Tensor(_SEG2), hidden)
                seq.backward(_total(out))
                seq.adam_step(_ADAM_LR)
                seq.save(main)
                out2, hidden2 = seq2.forward(Tensor(_SEG2), hidden2)
                seq2.backward(_total(out2))
                seq2.adam_step(_ADAM_LR)
                seq2.save(branch)
                self.assertEqual(_state(branch), _state(main))
            self.assertEqual(_head_value(branch), _head_value(main))

    def test_divergent_continuations_stay_self_consistent(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            seq, hidden = _trained_dir(main, 3)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch)
            seq2, _ = _stack()
            hidden2 = seq2.load(branch)

            out, hidden = seq.forward(Tensor(_SEG2), hidden)
            seq.backward(_total(out))
            seq.update(_LR)
            seq.save(main)
            seq2.adam_step(_ADAM_LR)
            seq2.save(branch)

            full_main = bytearray()
            seq.save(full_main)
            full_branch = bytearray()
            seq2.save(full_branch)
            self.assertNotEqual(bytes(full_main), bytes(full_branch))
            self.assertEqual(_state(main), bytes(full_main))
            self.assertEqual(_state(branch), bytes(full_branch))

    def test_same_position_appends_do_not_interfere(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            seq, hidden = _trained_dir(main, 2)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch)
            seq2, _ = _stack()
            hidden2 = seq2.load(branch)

            # Both chains append their next delta at segment 3.
            out, hidden = seq.forward(Tensor(_SEG2), hidden)
            seq.backward(_total(out))
            seq.update(_LR)
            seq.save(main)
            seq2.zero_grad()
            seq2.update(0.5)
            seq2.save(branch)

            main_seg3 = os.path.join(main, _seg(3))
            branch_seg3 = os.path.join(branch, _seg(3))
            with open(main_seg3, "rb") as fh:
                main_bytes = fh.read()
            with open(branch_seg3, "rb") as fh:
                branch_bytes = fh.read()
            self.assertNotEqual(main_bytes, branch_bytes)
            self.assertNotEqual(
                os.stat(main_seg3).st_ino, os.stat(branch_seg3).st_ino
            )
            # Each chain still reassembles its own complete state.
            full_main = bytearray()
            seq.save(full_main)
            full_branch = bytearray()
            seq2.save(full_branch)
            self.assertEqual(_state(main), bytes(full_main))
            self.assertEqual(_state(branch), bytes(full_branch))

    def test_saves_loads_and_compactions_interleave_across_the_family(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 8)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=5)
            errors = []
            stop = False

            def appender(td):
                try:
                    while not stop:
                        doc = cp.load_chain(td)
                        cp.save_chain(doc, td)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def reader(td):
                try:
                    while not stop:
                        cp.build_bytes(cp.load_chain(td))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=appender, args=(main,)),
                threading.Thread(target=appender, args=(branch,)),
                threading.Thread(target=reader, args=(main,)),
                threading.Thread(target=reader, args=(branch,)),
            ]
            for thread in threads:
                thread.start()
            try:
                for _ in range(10):
                    cp.compact_chain(main)
                    cp.compact_chain(branch)
            finally:
                stop = True
                for thread in threads:
                    thread.join()
            self.assertEqual(errors, [])
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertTrue(cp.verify_chain(branch).ok)


class FamilyReclaimTests(unittest.TestCase):
    def test_compacting_the_main_chain_keeps_the_branch_reachable(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 6)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=3)
            branch_state = _state(branch)
            branch_listing = sorted(os.listdir(branch))

            # Partial compaction across the fork point, then a full one.
            cp.compact_chain(main, up_to=4)
            self.assertEqual(_state(branch), branch_state)
            self.assertEqual(sorted(os.listdir(branch)), branch_listing)
            cp.compact_chain(main)
            self.assertEqual(_seg_names(main), [_seg(0)])
            self.assertEqual(_state(branch), branch_state)
            self.assertEqual(sorted(os.listdir(branch)), branch_listing)
            self.assertTrue(cp.verify_chain(branch).ok)

    def test_main_tail_conversion_never_rewrites_shared_segments(self):
        # Fold below the fork point: the streaming conversion replaces the
        # main chain's own tail slots (which the branch still references)
        # by rename, so the branch keeps the original bytes throughout.
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 6)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=5)
            branch_state = _state(branch)
            main_head_state = _state(main)
            shared_before = {}
            for i in range(6):
                with open(os.path.join(branch, _seg(i)), "rb") as fh:
                    shared_before[i] = fh.read()

            cp.compact_chain(main, up_to=2)
            self.assertEqual(_state(main), main_head_state)
            for i in range(6):
                with open(os.path.join(branch, _seg(i)), "rb") as fh:
                    self.assertEqual(fh.read(), shared_before[i])
            self.assertEqual(_state(branch), branch_state)
            self.assertTrue(cp.verify_chain(branch).ok)
            self.assertTrue(cp.verify_chain(main).ok)

    def test_compacting_the_branch_keeps_the_main_chain_untouched(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 6)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=3)
            main_state = _state(main)
            main_listing = sorted(os.listdir(main))

            cp.compact_chain(branch)
            self.assertEqual(_seg_names(branch), [_seg(0)])
            self.assertEqual(_state(main), main_state)
            self.assertEqual(sorted(os.listdir(main)), main_listing)
            self.assertTrue(cp.verify_chain(main).ok)

    def test_dropping_a_branch_reclaims_only_unreachable_segments(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 5)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=3)
            main_state = _state(main)

            cp.drop_chain(branch)
            self.assertFalse(os.path.exists(branch))
            # The main chain is complete and untouched.
            self.assertEqual(_state(main), main_state)
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertEqual(_seg_names(main), [_seg(i) for i in range(6)])

    def test_shared_segments_survive_until_no_chain_can_reach_them(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 5)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=3)
            branch_state = _state(branch)

            # The main chain lets go of the shared prefix entirely; the
            # branch's references keep every shared segment alive.
            cp.compact_chain(main)
            cp.drop_chain(main)
            self.assertFalse(os.path.exists(main))
            self.assertEqual(_state(branch), branch_state)
            self.assertTrue(cp.verify_chain(branch).ok)
            self.assertEqual(_seg_names(branch), [_seg(i) for i in range(4)])

            # Dropping the last chain reclaims everything.
            cp.drop_chain(branch)
            self.assertEqual(os.listdir(parent), [])

    def test_drop_missing_directory_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                cp.drop_chain(os.path.join(td, "absent"))
        with self.assertRaises(TypeError):
            cp.drop_chain(123)


class FamilyVerifyTests(unittest.TestCase):
    def test_family_members_verify_and_report_the_chain(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 4)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=2)

            report_main = cp.verify_chain(main)
            report_branch = cp.verify_chain(branch)
            self.assertTrue(report_main.ok)
            self.assertEqual(report_main.chain, main)
            self.assertTrue(report_branch.ok)
            self.assertEqual(report_branch.chain, branch)
            self.assertEqual((report_branch.head, report_branch.segments), (2, 3))
            # A standalone chain reports no chain identity.
            with tempfile.TemporaryDirectory() as solo:
                _trained_dir(solo, 2)
                self.assertIsNone(cp.verify_chain(solo).chain)

    def test_verify_names_the_chain_of_the_first_bad_segment(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 4)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=2)

            # Corrupt a shared segment through the branch: both chains
            # reject it, each naming itself as the owning chain.
            shared = os.path.join(branch, _seg(1))
            with open(shared, "rb") as fh:
                good = fh.read()
            with open(shared, "wb") as fh:
                fh.write(good[: len(good) // 2])
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain(branch)
            self.assertIn("segment 1", str(ctx.exception))
            self.assertIn(branch, str(ctx.exception))
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain(main)
            self.assertIn("segment 1", str(ctx.exception))
            self.assertIn(main, str(ctx.exception))
            with open(shared, "wb") as fh:
                fh.write(good)
            self.assertTrue(cp.verify_chain(branch).ok)
            self.assertTrue(cp.verify_chain(main).ok)

            # A defect in a main-only segment leaves the branch sound.
            with open(os.path.join(main, _seg(4)), "wb") as fh:
                fh.write(good[:24])
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain(main)
            self.assertIn("segment 4", str(ctx.exception))
            self.assertIn(main, str(ctx.exception))
            self.assertTrue(cp.verify_chain(branch).ok)

    def test_verify_family_is_strictly_read_only(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 4)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch, at=2)
            before_main = _chain_hash(main)
            before_branch = _chain_hash(branch)
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertTrue(cp.verify_chain(branch).ok)
            self.assertEqual(_chain_hash(main), before_main)
            self.assertEqual(_chain_hash(branch), before_branch)

            os.chmod(branch, 0o555)
            try:
                self.assertTrue(cp.verify_chain(branch).ok)
            finally:
                os.chmod(branch, 0o755)
            self.assertEqual(_chain_hash(branch), before_branch)

    def test_sequential_verify_reports_family_chains(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            seq, _ = _trained_dir(main, 3)
            branch = os.path.join(parent, "branch")
            cp.derive_chain(main, branch)
            self.assertEqual(seq.verify(branch).chain, branch)


class DeriveErrorTests(unittest.TestCase):
    def test_existing_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 2)
            existing_dir = os.path.join(parent, "existing")
            os.mkdir(existing_dir)
            with self.assertRaises(ValueError):
                cp.derive_chain(main, existing_dir)
            existing_file = os.path.join(parent, "file")
            with open(existing_file, "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(ValueError):
                cp.derive_chain(main, existing_file)

    def test_bad_fork_points_are_rejected(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 2)
            for bad in (-1, True, 0.5, "1"):
                with self.assertRaises(ValueError, msg=f"at={bad!r}"):
                    cp.derive_chain(main, os.path.join(parent, "b"), at=bad)
            with self.assertRaises(ValueError):
                cp.derive_chain(main, os.path.join(parent, "b"), at=3)

    def test_missing_source_and_missing_segment_are_filenotfound(self):
        with tempfile.TemporaryDirectory() as parent:
            with self.assertRaises(FileNotFoundError):
                cp.derive_chain(os.path.join(parent, "absent"),
                                os.path.join(parent, "b"))
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 3)
            os.unlink(os.path.join(main, _seg(1)))
            with self.assertRaises(FileNotFoundError):
                cp.derive_chain(main, os.path.join(parent, "b"), at=2)

    def test_unwritable_destination_is_oserror(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            _trained_dir(main, 2)
            os.chmod(parent, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.derive_chain(main, os.path.join(parent, "b"))
            finally:
                os.chmod(parent, 0o755)

    def test_deriving_an_empty_chain_is_rejected(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            with self.assertRaises(ValueError):
                cp.derive_chain(main, os.path.join(parent, "b"))

    def test_type_checks(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            with self.assertRaises(TypeError):
                cp.derive_chain(123, os.path.join(parent, "b"))
            with self.assertRaises(TypeError):
                cp.derive_chain(main, 123)
            seq, _ = _stack()
            with self.assertRaises(TypeError):
                seq.derive(object())
            with self.assertRaises(TypeError):
                seq.derive(main)  # a directory derive needs a target
            with self.assertRaises(TypeError):
                seq.drop(object())

    def test_sequential_derive_and_drop_directories(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as parent:
            seq, _ = _trained_dir(main, 3)
            branch = os.path.join(parent, "branch")
            self.assertIsNone(seq.derive(main, branch, at=2))
            self.assertEqual(_state(branch), _state(main, up_to=2))
            seq.drop(branch)
            self.assertFalse(os.path.exists(branch))
            with self.assertRaises(FileNotFoundError):
                seq.drop(branch)

    def test_memory_chain_roundtrip_through_sequential(self):
        seq, _ = _stack()
        chain = MemoryChain()
        seq.save(chain)
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        seq.save(chain)
        forked = seq.derive(chain)
        self.assertIsInstance(forked, MemoryChain)
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(forked)),
            cp.build_bytes(cp.load_chain_memory(chain)),
        )
        seq.drop(forked)
        with self.assertRaises(ValueError):
            cp.load_chain_memory(forked)


if __name__ == "__main__":
    unittest.main()
