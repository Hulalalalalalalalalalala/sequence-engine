"""Tests for chain forks: branch chains sharing a prefix segment family.

A chain forked at segment *k* produces a second chain directory whose
segments ``0..k`` are hard links to the source chain's files, plus its
own ``head``; from then on each chain maintains only its own head and
appended deltas.  These tests cover:

* prefix sharing and head independence (the shared files are stored
  once; appends to the same segment position never interfere);
* bitwise equivalence with chains that never forked, through saves,
  loads, streaming compactions and branch deletion on both sides;
* deterministic, reachability-based reclamation of shared segments;
* the fork error taxonomy (ValueError / FileNotFoundError / OSError)
  and kill-at-any-point safety of the fork itself;
* read-only verification of a whole chain family, reporting the first
  bad segment together with the chain it belongs to.
"""

from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing
import os
import shutil
import struct
import tempfile
import threading
import time
import unittest
import zlib
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


def _renumber_delta(path, new_number):
    raw = _read(path)
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
    return body + cp.DELTA_END_MAGIC + struct.pack("<QI", len(payload) // 9, crc)


def _reshape_first_delta_entry(path, shape):
    raw = _read(path)
    hlen = struct.unpack("<Q", raw[12:20])[0]
    header = json.loads(raw[20 : 20 + hlen])
    payload = raw[20 + hlen : raw.rfind(cp.DELTA_END_MAGIC)]
    header["changed"][0]["s"] = shape
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
    return body + cp.DELTA_END_MAGIC + struct.pack("<QI", len(payload) // 9, crc)


class ForkSharingTests(unittest.TestCase):
    def test_fork_shares_prefix_files_and_heads_are_independent(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)

            # The branch holds exactly the shared prefix plus its head.
            self.assertEqual(
                sorted(os.listdir(branch)),
                ["head"] + [_seg(i) for i in range(3)],
            )
            self.assertEqual(_read(os.path.join(branch, "head")), b"2")
            self.assertEqual(_read(os.path.join(main, "head")), b"4")
            # The prefix is stored once: same inode, link count of two.
            for index in range(3):
                main_stat = os.stat(os.path.join(main, _seg(index)))
                branch_stat = os.stat(os.path.join(branch, _seg(index)))
                self.assertEqual(main_stat.st_ino, branch_stat.st_ino)
                self.assertEqual(main_stat.st_nlink, 2)
                self.assertEqual(
                    _read(os.path.join(main, _seg(index))),
                    _read(os.path.join(branch, _seg(index))),
                )
            # The branch reassembles exactly the fork-point state.
            self.assertEqual(
                _chain_state(branch),
                cp.build_bytes(cp.load_chain(main, up_to=2)),
            )

    def test_fork_default_fork_point_is_the_head(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            self.assertEqual(_read(os.path.join(branch, "head")), b"3")
            self.assertEqual(_chain_state(branch), _chain_state(main))

    def test_forked_chains_match_unforked_references_bitwise(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            seq, hidden = _trained_dir(main, 5)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=3)

            # The branch continues with its own op stream from the fork
            # state; the main chain continues its own way.
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 2, hidden=bhidden, start=3)
            _train(seq, main, 2, hidden=hidden, start=5)

            # Reference chains replaying the same op streams with no fork.
            ref_main = os.path.join(root, "ref_main")
            rseq, _ = _stack()
            os.mkdir(ref_main)
            rseq.save(ref_main)
            _train(rseq, ref_main, 7)
            ref_branch = os.path.join(root, "ref_branch")
            bref, _ = _stack()
            os.mkdir(ref_branch)
            bref.save(ref_branch)
            bh = _train(bref, ref_branch, 3)
            _train(bref, ref_branch, 2, hidden=bh, start=3)

            self.assertEqual(_chain_state(main), _chain_state(ref_main))
            self.assertEqual(_chain_state(branch), _chain_state(ref_branch))
            # The optimizer state rides along: the branch stepped Adam at
            # its own times and still matches the reference.
            self.assertEqual(
                cp.load_chain(branch)["optim"]["t"],
                cp.load_chain(ref_branch)["optim"]["t"],
            )

    def test_same_position_appends_do_not_interfere(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            seq, hidden = _trained_dir(main, 2)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)  # both heads at segment 2

            # The main chain appends segment 3 first.
            _train(seq, main, 1, hidden=hidden, start=2)
            main_seg3 = _read(os.path.join(main, _seg(3)))
            main_hash = _dir_hash(main)

            # The branch appends its own segment 3 with a different state.
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            out, bhidden = bseq.forward(Tensor(_SEG2), bhidden)
            bseq.backward(_total(out))
            bseq.adam_step(_ADAM_LR)
            bseq.save(branch)

            # The main chain's bytes did not move; the two segment-3
            # files carry different deltas.
            self.assertEqual(_dir_hash(main), main_hash)
            self.assertEqual(_read(os.path.join(main, _seg(3))), main_seg3)
            self.assertNotEqual(
                _read(os.path.join(branch, _seg(3))), main_seg3
            )
            # Each chain reassembles its own state.
            buf = bytearray()
            seq.save(buf)
            self.assertEqual(_chain_state(main), bytes(buf))
            bbuf = bytearray()
            bseq.save(bbuf)
            self.assertEqual(_chain_state(branch), bytes(bbuf))

    def test_fork_from_a_compaction_point(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 5)
            at_two = cp.build_bytes(cp.load_chain(main, up_to=2))
            cp.compact_chain(main, up_to=2)
            self.assertEqual(_read(os.path.join(main, "head")), b"3")
            main_state = _chain_state(main)

            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=0)
            self.assertEqual(_chain_state(branch), at_two)
            # The branch grows from the folded basis independently.
            bseq, _ = _stack()
            bhidden = bseq.load(branch)
            _train(bseq, branch, 2, hidden=bhidden, start=3)
            self.assertEqual(_read(os.path.join(branch, "head")), b"2")
            self.assertEqual(_chain_state(main), main_state)
            self.assertTrue(cp.verify_chain(branch).ok)


class ForkReclamationTests(unittest.TestCase):
    def test_compact_main_reclaims_only_its_own_references(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            branch_state = _chain_state(branch)
            main_state = _chain_state(main)

            cp.compact_chain(main)  # fold everything into a new basis
            self.assertEqual(sorted(os.listdir(main)), ["head", _seg(0)])
            self.assertEqual(_chain_state(main), main_state)

            # The branch is untouched: its shared prefix stayed reachable
            # through its own head, and the source chain's compaction
            # released only the source's references.
            self.assertEqual(_chain_state(branch), branch_state)
            self.assertTrue(cp.verify_chain(branch).ok)
            for index in range(3):
                stat = os.stat(os.path.join(branch, _seg(index)))
                self.assertEqual(stat.st_nlink, 1)

    def test_compact_branch_keeps_main_intact(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=3)
            main_state = _chain_state(main)
            branch_state = _chain_state(branch)

            cp.compact_chain(branch)
            self.assertEqual(sorted(os.listdir(branch)), ["head", _seg(0)])
            self.assertEqual(_chain_state(branch), branch_state)
            # The main chain still reaches its whole chain: its head was
            # beyond the branch's fork point, so every one of its
            # segments is still reachable and present.
            self.assertEqual(_chain_state(main), main_state)
            self.assertEqual(_read(os.path.join(main, "head")), b"4")
            self.assertTrue(cp.verify_chain(main).ok)

    def test_deleting_branch_reclaims_shared_segments(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            main_state = _chain_state(main)
            for index in range(3):
                self.assertEqual(
                    os.stat(os.path.join(main, _seg(index))).st_nlink, 2
                )

            shutil.rmtree(branch)
            # The branch's references are gone; the main chain keeps its
            # own and still reassembles its complete state.
            for index in range(3):
                self.assertEqual(
                    os.stat(os.path.join(main, _seg(index))).st_nlink, 1
                )
            self.assertEqual(_chain_state(main), main_state)
            self.assertTrue(cp.verify_chain(main).ok)

    def test_compacting_both_chains_leaves_no_shared_segments(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=2)
            main_state = _chain_state(main)
            branch_state = _chain_state(branch)
            cp.compact_chain(main)
            cp.compact_chain(branch)
            self.assertEqual(_chain_state(main), main_state)
            self.assertEqual(_chain_state(branch), branch_state)
            # Each chain now owns its single basis outright.
            self.assertEqual(os.stat(os.path.join(main, _seg(0))).st_nlink, 1)
            self.assertEqual(os.stat(os.path.join(branch, _seg(0))).st_nlink, 1)
            self.assertNotEqual(
                os.stat(os.path.join(main, _seg(0))).st_ino,
                os.stat(os.path.join(branch, _seg(0))).st_ino,
            )


class ForkErrorTests(unittest.TestCase):
    def test_fork_point_must_name_a_segment_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            for bad in (0.5, 1.5, "1", True, -1):
                with self.assertRaises(ValueError, msg=f"up_to={bad!r}"):
                    cp.fork_chain(main, os.path.join(root, "b"), up_to=bad)
            # A segment the chain has not committed does not exist.
            with self.assertRaises(ValueError):
                cp.fork_chain(main, os.path.join(root, "b"), up_to=4)
            with self.assertRaises(ValueError):
                cp.fork_chain(main, os.path.join(root, "b"), up_to=99)

    def test_fork_target_must_not_exist(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            existing_dir = os.path.join(root, "dir")
            os.mkdir(existing_dir)
            with self.assertRaises(ValueError):
                cp.fork_chain(main, existing_dir)
            existing_file = os.path.join(root, "file")
            with open(existing_file, "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(ValueError):
                cp.fork_chain(main, existing_file)
            # The source directory itself is an existing target too.
            with self.assertRaises(ValueError):
                cp.fork_chain(main, main)
            # A target inside the source chain would pollute it.
            with self.assertRaises(ValueError):
                cp.fork_chain(main, os.path.join(main, "nested"))

    def test_fork_missing_source_and_missing_segment(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                cp.fork_chain(os.path.join(root, "absent"),
                              os.path.join(root, "b"))
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            os.unlink(os.path.join(main, _seg(2)))
            # Segments at or before the fork point are the referenced
            # ones: a fork below the gap still works ...
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=1)
            self.assertTrue(cp.verify_chain(branch).ok)
            # ... a fork reaching across the gap names the missing file.
            with self.assertRaises(FileNotFoundError):
                cp.fork_chain(main, os.path.join(root, "b2"), up_to=3)
            with self.assertRaises(FileNotFoundError):
                cp.fork_chain(main, os.path.join(root, "b3"))

    def test_corrupt_prefix_rejected_before_target_appears(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            path = os.path.join(main, _seg(1))
            with open(path, "r+b") as fh:
                fh.truncate(os.path.getsize(path) // 2)
            branch = os.path.join(root, "branch")
            with self.assertRaises(ValueError):
                cp.fork_chain(main, branch, up_to=2)
            self.assertFalse(os.path.exists(branch))

    def test_fork_type_checks(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 1)
            with self.assertRaises(TypeError):
                cp.fork_chain(123, os.path.join(root, "b"))
            with self.assertRaises(TypeError):
                cp.fork_chain(main, 123)
            with self.assertRaises(TypeError):
                cp.fork_chain_memory(object())
            seq, _ = _stack()
            with self.assertRaises(TypeError):
                seq.fork(b"not-a-chain")
            with self.assertRaises(TypeError):
                seq.fork(main)  # a directory fork needs a target

    def test_unwritable_destination_is_oserror(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write into read-only directories")
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 1)
            os.chmod(root, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.fork_chain(main, os.path.join(root, "branch"))
            finally:
                os.chmod(root, 0o755)

    def test_failed_fork_leaves_no_target_and_no_staging(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            with mock.patch.object(
                cp.os, "link",
                side_effect=OSError(errno.ENOSPC, "disk full"),
            ):
                with self.assertRaises(OSError):
                    cp.fork_chain(main, branch)
            self.assertFalse(os.path.exists(branch))
            self.assertEqual(
                [name for name in os.listdir(root) if name != "main"], []
            )
            # The source chain is untouched.
            self.assertTrue(cp.verify_chain(main).ok)

    def test_killed_fork_staging_is_reclaimed_by_the_next_fork(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            # Debris of a killed earlier attempt, holding a shared
            # segment reference hostage.
            staging = os.path.join(root, ".seqfork.tmp-branch-dead")
            os.mkdir(staging)
            os.link(
                os.path.join(main, _seg(0)), os.path.join(staging, _seg(0))
            )
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=1)
            self.assertFalse(os.path.exists(staging))
            self.assertEqual(
                sorted(os.listdir(branch)), ["head", _seg(0), _seg(1)]
            )
            self.assertTrue(cp.verify_chain(branch).ok)


class ForkCrashTests(unittest.TestCase):
    def test_killed_fork_leaves_source_complete_and_target_atomic(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 40)
            main_state = _chain_state(main)
            branch = os.path.join(root, "branch")

            proc = multiprocessing.Process(
                target=_slow_fork_worker, args=(main, branch, 40)
            )
            proc.start()
            # Kill once the staging directory exists, mid-linking.
            deadline = time.monotonic() + 30.0
            saw_staging = False
            while proc.is_alive() and time.monotonic() < deadline:
                if any(
                    name.startswith(".seqfork.tmp-")
                    for name in os.listdir(root)
                ):
                    saw_staging = True
                    break
                time.sleep(0.001)
            if proc.is_alive():
                proc.kill()
            proc.join()
            self.assertTrue(saw_staging)

            # The source chain is one complete state; the target either
            # never appeared or is a complete chain -- never half one.
            self.assertEqual(_chain_state(main), main_state)
            if os.path.exists(branch):
                self.assertTrue(cp.verify_chain(branch).ok)
            # A fresh fork to the same target reclaims any debris and
            # produces a complete branch.
            if os.path.exists(branch):
                shutil.rmtree(branch)
            cp.fork_chain(main, branch, up_to=40)
            self.assertEqual(_chain_state(branch), main_state)
            self.assertEqual(
                [
                    name
                    for name in os.listdir(root)
                    if name.startswith(".seqfork.tmp-")
                ],
                [],
            )


def _slow_fork_worker(source, target, up_to):
    from sequence_engine import checkpoint as cp

    original_link = os.link

    def slow_link(src, dst):
        time.sleep(0.02)
        return original_link(src, dst)

    cp.os.link = slow_link
    try:
        cp.fork_chain(source, target, up_to)
    finally:
        cp.os.link = original_link


class ForkConcurrencyTests(unittest.TestCase):
    def test_fork_rolls_forward_an_interrupted_compaction_first(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 4)
            head_state = _chain_state(main)
            # A streaming compaction killed right after its marker: the
            # staged basis and the marker are on disk, the owner is dead.
            folded = cp.build_bytes(cp.load_chain(main, up_to=2))
            cp._atomic_write(main, cp._staged_name(0), folded)
            cp._atomic_write(main, cp._COMPACT_MARKER, b'{"u":2}')
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch)
            # The fork first completed the interrupted fold, then
            # branched from the compacted head.
            self.assertEqual(_read(os.path.join(main, "head")), b"2")
            self.assertEqual(_read(os.path.join(branch, "head")), b"2")
            self.assertEqual(_chain_state(main), head_state)
            self.assertEqual(_chain_state(branch), head_state)
            # The folded prefix became the shared basis.
            self.assertEqual(
                cp.build_bytes(cp.load_chain(branch, up_to=0)), folded
            )
            self.assertTrue(cp.verify_chain([main, branch]).ok)

    def test_fork_while_compactions_run_on_the_source(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 30)
            errors = []
            stop = False

            def compactor():
                try:
                    while not stop:
                        cp.compact_chain(main)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            thread = threading.Thread(target=compactor)
            thread.start()
            branches = []
            try:
                for i in range(5):
                    target = os.path.join(root, f"b{i}")
                    cp.fork_chain(main, target)
                    branches.append(target)
            finally:
                stop = True
                thread.join()
            self.assertEqual(errors, [])
            # Every fork produced a complete, verifiable chain, and the
            # source chain is one complete state.
            for target in branches:
                self.assertTrue(cp.verify_chain(target).ok)
            self.assertTrue(cp.verify_chain(main).ok)

    def test_appends_and_loads_flow_across_the_family(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 6)
            branch = os.path.join(root, "branch")
            cp.fork_chain(main, branch, up_to=3)
            errors = []
            stop = False

            def appender(directory):
                try:
                    while not stop:
                        doc = cp.load_chain(directory)
                        cp.save_chain(doc, directory)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def reader(directory):
                try:
                    while not stop:
                        cp.build_bytes(cp.load_chain(directory))
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
            for _ in range(5):
                cp.compact_chain(main)
                cp.compact_chain(branch)
            stop = True
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            # Both chains still verify as one family.
            report = cp.verify_chain([main, branch])
            self.assertTrue(report.ok)


class FamilyVerifyTests(unittest.TestCase):
    def _family(self, root, fork_at=2, main_steps=4):
        main = os.path.join(root, "main")
        seq, hidden = _trained_dir(main, main_steps)
        branch = os.path.join(root, "branch")
        cp.fork_chain(main, branch, up_to=fork_at)
        bseq, _ = _stack()
        bhidden = bseq.load(branch)
        _train(bseq, branch, 1, hidden=bhidden, start=fork_at)
        return main, branch

    def test_sound_family_returns_a_family_report(self):
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._family(root)
            report = cp.verify_chain([main, branch])
            self.assertTrue(report.ok)
            self.assertTrue(report)
            self.assertEqual(report.members, (main, branch))
            self.assertEqual(len(report.chains), 2)
            self.assertEqual(report.chains[0].head, 4)
            self.assertEqual(report.chains[1].head, 3)
            self.assertIn("ok=True", repr(report))
            # The explicit family entry point and the container agree.
            self.assertTrue(cp.verify_family([main, branch]).ok)
            seq, _ = _stack()
            self.assertTrue(seq.verify([main, branch]).ok)
            # A family of one is just the single chain.
            self.assertTrue(cp.verify_chain([main]).ok)

    def test_family_verify_names_the_owning_chain(self):
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._family(root)

            # A corrupted shared segment is reported against the first
            # chain that reaches it, at its segment position.
            shared = os.path.join(main, _seg(1))
            with open(shared, "r+b") as fh:
                fh.truncate(os.path.getsize(shared) // 2)
            before = (_dir_hash(main), _dir_hash(branch))
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain([main, branch])
            message = str(ctx.exception)
            self.assertIn("chain 0", message)
            self.assertIn(main, message)
            self.assertIn("segment 1", message)
            # The shared damage is visible from the branch as well.
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain(branch)
            self.assertIn("segment 1", str(ctx.exception))
            # Verification wrote nothing anywhere.
            self.assertEqual((_dir_hash(main), _dir_hash(branch)), before)

    def test_family_verify_locates_branch_only_defects(self):
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._family(root)

            # A truncated branch-only tail segment names the branch.
            tail = os.path.join(branch, _seg(3))
            with open(tail, "r+b") as fh:
                fh.truncate(os.path.getsize(tail) // 2)
            before = (_dir_hash(main), _dir_hash(branch))
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain([main, branch])
            message = str(ctx.exception)
            self.assertIn("chain 1", message)
            self.assertIn(branch, message)
            self.assertIn("segment 3", message)
            # The main chain alone is still sound.
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertEqual((_dir_hash(main), _dir_hash(branch)), before)

    def test_family_verify_locates_reordered_and_shape_drift(self):
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._family(root)

            forged = _renumber_delta(os.path.join(branch, _seg(3)), 99)
            with open(os.path.join(branch, _seg(3)), "wb") as fh:
                fh.write(forged)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_family([main, branch])
            self.assertIn("segment 3", str(ctx.exception))
            self.assertIn(branch, str(ctx.exception))

        with tempfile.TemporaryDirectory() as root:
            main, branch = self._family(root)
            forged = _reshape_first_delta_entry(
                os.path.join(branch, _seg(3)), [9, 9]
            )
            with open(os.path.join(branch, _seg(3)), "wb") as fh:
                fh.write(forged)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_family([main, branch])
            self.assertIn("segment 3", str(ctx.exception))

    def test_family_verify_missing_member_and_type_errors(self):
        with tempfile.TemporaryDirectory() as root:
            main, branch = self._family(root)
            with self.assertRaises(FileNotFoundError):
                cp.verify_chain([main, os.path.join(root, "absent")])
            with self.assertRaises(ValueError):
                cp.verify_chain([])  # a family needs at least one chain
            with self.assertRaises(TypeError):
                cp.verify_family(main)  # a bare path is not a family
            with self.assertRaises(TypeError):
                cp.verify_chain([main, 123])
            seq, _ = _stack()
            with self.assertRaises(TypeError):
                seq.verify([main, object()])


class MemoryForkTests(unittest.TestCase):
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
        return chain, seq, hidden

    def test_memory_fork_shares_prefix_and_evolves_independently(self):
        chain, seq, hidden = self._memory_chain(4)
        branch = cp.fork_chain_memory(chain, up_to=2)
        self.assertEqual(branch.read_head(), b"2")
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(branch)),
            cp.build_bytes(cp.load_chain_memory(chain, up_to=2)),
        )
        # Prefix segment bytes are shared by reference.
        for index in range(3):
            name = _seg(index)
            self.assertIs(
                chain._objects[name], branch._objects[name]
            )
        # Appends land in each chain's own tail.
        seq.save(chain)
        bseq, _ = _stack()
        bseq.load(branch)
        bseq.update(_LR)
        bseq.save(branch)
        self.assertEqual(chain.read_head(), b"5")
        self.assertEqual(branch.read_head(), b"3")
        self.assertNotEqual(
            chain._objects[_seg(3)], branch._objects[_seg(3)]
        )
        # Compacting one leaves the other bit for bit intact.
        branch_state = cp.build_bytes(cp.load_chain_memory(branch))
        cp.compact_chain_memory(chain)
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(branch)), branch_state
        )
        self.assertTrue(cp.verify_chain_memory(branch).ok)

    def test_memory_fork_errors(self):
        chain, _, _ = self._memory_chain(2)
        for bad in (0.5, "1", True, -1, 3):
            with self.assertRaises(ValueError, msg=f"up_to={bad!r}"):
                cp.fork_chain_memory(chain, bad)
        with self.assertRaises(ValueError):
            cp.fork_chain_memory(MemoryChain())  # no committed basis
        seq, _ = _stack()
        with self.assertRaises(TypeError):
            seq.fork(chain, "somewhere")  # memory forks take no target


class SequentialForkTests(unittest.TestCase):
    def test_sequential_fork_directory_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            seq, hidden = _trained_dir(main, 3)
            branch = os.path.join(root, "branch")
            self.assertIsNone(seq.fork(main, branch, up_to=2))
            self.assertEqual(_read(os.path.join(branch, "head")), b"2")
            # The branch loads into a fresh container and continues.
            resumed, _ = _stack()
            hidden2 = resumed.load(branch)
            self.assertIsNotNone(hidden2)
            _train(resumed, branch, 1, hidden=hidden2, start=3)
            self.assertTrue(seq.verify(branch).ok)

    def test_sequential_fork_memory_returns_branch(self):
        chain = MemoryChain()
        seq, _ = _stack()
        seq.save(chain)
        seq.update(_LR)
        seq.save(chain)
        branch = seq.fork(chain)
        self.assertIsInstance(branch, MemoryChain)
        self.assertEqual(branch.read_head(), b"1")
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(branch)),
            cp.build_bytes(cp.load_chain_memory(chain)),
        )


if __name__ == "__main__":
    unittest.main()
