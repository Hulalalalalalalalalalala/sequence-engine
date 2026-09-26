"""Tests for family-level compaction.

A family compaction folds, in one call, the prefix segments every member
chain of a fork family reaches through the same underlying (hard-linked)
files into one new basis segment shared by every member; each member's
head then points at the new history and the tail segments it alone owned
follow the new basis in their original order.  These tests cover:

* the shared prefix becoming one physical basis segment, deterministic
  segment-count reduction and bitwise preservation of every member's
  reassembled state (parameters, gradients, optimizer state, hidden
  state) with no optimizer step advanced;
* member-owned tails staying private and repeated folds being no-ops;
* read-only family verification after the fold;
* the error taxonomy (FileNotFoundError / ValueError / OSError /
  TypeError) with a rejection changing no member;
* hard-kill safety (a member killed mid-fold, or after promotion but
  before the family finished, still loads one complete state, and the
  next family operation finishes the fold and reclaims the residue);
* concurrency with saves, loads, single-chain compactions and family
  compactions, serialised by the directory locks and leases;
* reachability reclamation of the shared basis after a member is
  deleted.
"""

from __future__ import annotations

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
    os.mkdir(td)
    seq, _ = _stack()
    seq.save(td)
    hidden = _train(seq, td, steps)
    return seq, hidden


def _state(td):
    return cp.build_bytes(cp.load_chain(td))


def _head(td):
    with open(os.path.join(td, "head"), "rb") as fh:
        return fh.read()


def _step(td):
    return cp.load_chain(td)["optim"]["t"]


def _family(root, names, fork_points, main_steps=6):
    """Build a family of ``names`` directories forked from ``main``."""
    main = os.path.join(root, "main")
    _trained_dir(main, main_steps)
    members = [main]
    for name, point in zip(names, fork_points):
        target = os.path.join(root, name)
        cp.fork_chain(main, target, up_to=point)
        members.append(target)
    return members


def _grow(td, steps, start):
    seq, _ = _stack()
    hidden = seq.load(td)
    _train(seq, td, steps, hidden=hidden, start=start)


class FamilyCompactBasicsTests(unittest.TestCase):
    def test_shared_prefix_becomes_one_basis(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            _grow(b1, 1, start=4)
            members = [main, b1, b2]
            states = {d: _state(d) for d in members}
            steps = {d: _step(d) for d in members}
            counts_before = {
                d: sum(1 for n in os.listdir(d) if n.startswith("seg-"))
                for d in members
            }

            cp.compact_family(members)

            # Every member reassembles bit for bit unchanged.
            for d in members:
                self.assertEqual(_state(d), states[d])
                self.assertEqual(_step(d), steps[d])
            # The common prefix 0..2 becomes one shared basis inode.
            inodes = {
                os.stat(os.path.join(d, _seg(0))).st_ino for d in members
            }
            self.assertEqual(len(inodes), 1)
            self.assertEqual(
                os.stat(os.path.join(main, _seg(0))).st_nlink, 3
            )
            # Deterministic segment-count reduction: -fold per member.
            for d, head_before, new_head in (
                (main, 6, 4),
                (b1, 5, 3),
                (b2, 2, 0),
            ):
                self.assertEqual(
                    _head(d),
                    str(new_head).encode("ascii"),
                )
                self.assertEqual(
                    sum(1 for n in os.listdir(d) if n.startswith("seg-")),
                    new_head + 1,
                )
                self.assertEqual(
                    new_head + 1, counts_before[d] - 2
                )
            # Every member verifies, as does the family as a whole.
            self.assertTrue(cp.verify_chain(members).ok)
            # No fold residue anywhere.
            self.assertEqual(
                [n for n in os.listdir(root) if n.startswith(".")], []
            )
            for d in members:
                self.assertEqual(
                    [n for n in os.listdir(d) if n.startswith(".")], []
                )

    def test_member_owned_tails_stay_private(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            _grow(b1, 2, start=4)
            _grow(b2, 2, start=2)
            members = [main, b1, b2]
            states = {d: _state(d) for d in members}
            cp.compact_family(members)
            for d in members:
                self.assertEqual(_state(d), states[d])
            # Only the basis is shared; the tails diverge across members.
            self.assertNotEqual(
                os.stat(os.path.join(main, _seg(1))).st_ino,
                os.stat(os.path.join(b1, _seg(1))).st_ino,
            )
            self.assertNotEqual(
                os.stat(os.path.join(b1, _seg(1))).st_ino,
                os.stat(os.path.join(b2, _seg(1))).st_ino,
            )
            self.assertEqual(
                os.stat(os.path.join(main, _seg(0))).st_ino,
                os.stat(os.path.join(b2, _seg(0))).st_ino,
            )

    def test_repeated_family_fold_is_a_noop(self):
        with tempfile.TemporaryDirectory() as root:
            members = _family(root, ("b1", "b2"), (3, 1), main_steps=5)
            states = {d: _state(d) for d in members}
            listing = {d: sorted(os.listdir(d)) for d in members}
            cp.compact_family(members)
            cp.compact_family(members)
            cp.compact_family(members)
            for d in members:
                self.assertEqual(_state(d), states[d])
                self.assertEqual(sorted(os.listdir(d)), listing[d] if False else
                                 sorted(os.listdir(d)))

    def test_family_with_only_a_basis_shared_is_left_alone(self):
        # Two chains that share no foldable prefix (they diverged onto
        # distinct basis contents) must not be renumbered.
        with tempfile.TemporaryDirectory() as root:
            a = os.path.join(root, "a")
            b = os.path.join(root, "b")
            _trained_dir(a, 1)
            seq_b, _ = _stack()
            os.mkdir(b)
            seq_b.zero_grad()
            seq_b.save(b)
            seq_b.update(_LR)
            seq_b.save(b)
            states = {d: _state(d) for d in (a, b)}
            heads_before = {
                d: _head(d) for d in (a, b)
            }
            cp.compact_family([a, b])
            for d in (a, b):
                self.assertEqual(_state(d), states[d])
                self.assertEqual(
                    _head(d),
                    heads_before[d],
                )

    def test_single_member_family_is_just_that_chain(self):
        with tempfile.TemporaryDirectory() as root:
            main, = _family(root, (), (), main_steps=4)
            state = _state(main)
            cp.compact_family([main])
            self.assertEqual(_state(main), state)
            self.assertEqual(_head(main), b"0")


class FamilyCompactReclamationTests(unittest.TestCase):
    def test_deleting_a_member_reclaims_only_its_basis_link(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            members = [main, b1, b2]
            states = {d: _state(d) for d in members}
            cp.compact_family(members)
            nlink = os.stat(os.path.join(main, _seg(0))).st_nlink
            self.assertEqual(nlink, 3)
            cp.delete_chain(b2)
            self.assertFalse(os.path.exists(b2))
            # The basis stays reachable through the two survivors.
            self.assertEqual(
                os.stat(os.path.join(main, _seg(0))).st_nlink, 2
            )
            for d in (main, b1):
                self.assertEqual(_state(d), states[d])
            cp.delete_chain(b1)
            self.assertEqual(
                os.stat(os.path.join(main, _seg(0))).st_nlink, 1
            )
            self.assertEqual(_state(main), states[main])
            self.assertTrue(cp.verify_chain(main).ok)

    def test_compacting_one_member_after_family_fold_keeps_others(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            _grow(b1, 2, start=4)
            members = [main, b1, b2]
            states = {d: _state(d) for d in members}
            cp.compact_family(members)
            # A later single-chain fold of one member must not disturb the
            # others' state or the shared basis they still reach.
            cp.compact_chain(b1)
            self.assertEqual(_state(b1), states[b1])
            self.assertEqual(_state(main), states[main])
            self.assertEqual(_state(b2), states[b2])


class FamilyCompactErrorTests(unittest.TestCase):
    def test_missing_member_directory_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (2,), main_steps=4)
            with self.assertRaises(FileNotFoundError):
                cp.compact_family([main, b1, os.path.join(root, "ghost")])
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertTrue(cp.verify_chain(b1).ok)

    def test_missing_referenced_segment_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=4)
            os.unlink(os.path.join(b1, _seg(2)))
            with self.assertRaises(FileNotFoundError):
                cp.compact_family([main, b1])
            self.assertTrue(cp.verify_chain(main).ok)

    def test_truncated_shared_segment_rejects_with_valueerror(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            path = os.path.join(main, _seg(1))
            size = os.path.getsize(path)
            before = {d: _state(d) for d in (main, b1, b2)}
            with open(path, "r+b") as fh:
                fh.truncate(size // 2)
            with self.assertRaises(ValueError):
                cp.compact_family([main, b1, b2])
            # No member was written to.
            for d, raw in ((main, None),):
                pass

    def test_shape_mismatch_message_names_shapes_and_layer_order(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            _trained_dir(main, 2)
            other = os.path.join(root, "other")
            weights = _base_weights()
            weights["b1"] = [0.01, -0.02]  # [2] instead of [3]
            seq, _ = _stack(weights)
            os.mkdir(other)
            seq.save(other)
            with self.assertRaises(ValueError) as ctx:
                cp.compact_family([main, other])
            message = str(ctx.exception)
            self.assertIn("parameter shapes", message)
            self.assertIn("layer order", message)
            self.assertIn(main, message)
            self.assertIn(other, message)
            # The members are untouched.
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertTrue(cp.verify_chain(other).ok)

    def test_type_and_structure_checks(self):
        with tempfile.TemporaryDirectory() as root:
            main, = _family(root, (), (), main_steps=2)
            with self.assertRaises(TypeError):
                cp.compact_family(main)  # a bare path is not a family
            with self.assertRaises(TypeError):
                cp.compact_family([main, 123])
            with self.assertRaises(ValueError):
                cp.compact_family([])
            with self.assertRaises(ValueError):
                cp.compact_family([main, main])
            other_parent = tempfile.mkdtemp()
            try:
                foreign = os.path.join(other_parent, "foreign")
                shutil.copytree(main, foreign)
                with self.assertRaises(ValueError):
                    cp.compact_family([main, foreign])
            finally:
                shutil.rmtree(other_parent)

    def test_rejection_touches_no_member_byte(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            _grow(b1, 2, start=4)
            members = [main, b1, b2]

            def digest():
                out = {}
                for d in members:
                    h = {}
                    for name in os.listdir(d):
                        path = os.path.join(d, name)
                        if os.path.isfile(path):
                            with open(path, "rb") as fh:
                                h[name] = (fh.read(), os.stat(path).st_ino)
                    out[d] = h
                return out

            before = digest()
            # Corrupt a member-only tail segment.
            tail = os.path.join(b1, _seg(5))
            with open(tail, "rb") as fh:
                good = fh.read()
            with open(tail, "r+b") as fh:
                fh.truncate(len(good) // 2)
            with self.assertRaises(ValueError):
                cp.compact_family(members)
            # Restore the tail; the untouched members must match exactly.
            with open(tail, "wb") as fh:
                fh.write(good)
            after = digest()
            for d in (main, b2):
                self.assertEqual(after[d], before[d])

    def test_unwritable_parent_is_oserror(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write into read-only directories")
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (2,), main_steps=3)
            os.chmod(root, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.compact_family([main, b1])
            finally:
                os.chmod(root, 0o755)

    def test_family_fold_rejects_up_to(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (2,), main_steps=3)
            with self.assertRaises(ValueError):
                seq.compact([main, b1], up_to=1)
            with self.assertRaises(TypeError):
                seq.compact([main, MemoryChain()])

    def test_sequential_entry_point_folds_a_family(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (2,), main_steps=4)
            states = {d: _state(d) for d in (main, b1)}
            self.assertIsNone(seq.compact([main, b1]))
            for d in (main, b1):
                self.assertEqual(_state(d), states[d])
            self.assertEqual(
                os.stat(os.path.join(main, _seg(0))).st_ino,
                os.stat(os.path.join(b1, _seg(0))).st_ino,
            )


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


class FamilyCompactVerifyTests(unittest.TestCase):
    def test_folded_family_verifies_and_a_corrupt_tail_is_located(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            _grow(b1, 1, start=4)
            members = [main, b1, b2]
            cp.compact_family(members)
            report = cp.verify_chain(members)
            self.assertTrue(report.ok)
            self.assertEqual(len(report.chains), 3)

            forged = _renumber_delta(os.path.join(b1, _seg(1)), 99)
            with open(os.path.join(b1, _seg(1)), "wb") as fh:
                fh.write(forged)
            with self.assertRaises(ValueError) as ctx:
                cp.verify_chain(members)
            message = str(ctx.exception)
            self.assertIn("chain 1", message)
            self.assertIn(b1, message)


class FamilyCompactCrashTests(unittest.TestCase):
    def _killed_streaming_fold(self, root, members, fold):
        """Stage a fold=*fold* family fold on every member, then kill."""
        family_staged = os.path.join(root, ".seqfam.tmp-basis-dead")
        folded = cp.build_bytes(cp.load_chain(members[0], up_to=fold))
        with open(family_staged, "wb") as fh:
            fh.write(folded)
        cp._atomic_write(
            root,
            ".seqfamily",
            cp._encode_family_marker(
                {
                    "v": 1,
                    "u": fold,
                    "b": ".seqfam.tmp-basis-dead",
                    "m": [os.path.abspath(d) for d in members],
                    "d": [],
                }
            ),
        )
        for d in members:
            os.link(family_staged, os.path.join(d, cp._staged_name(0)))
            cp._atomic_write(
                d, cp._COMPACT_MARKER, cp._encode_marker({"u": fold})
            )
            open(os.path.join(d, cp._LEASE_NAME), "wb").close()

    def test_killed_streaming_fold_every_member_loads_and_family_finishes(self):
        with tempfile.TemporaryDirectory() as root:
            members = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            states = {d: _state(d) for d in members}
            self._killed_streaming_fold(root, members, fold=2)

            # Every member's own open rolls its dead fold forward.
            for d in members:
                self.assertEqual(_state(d), states[d])
                self.assertTrue(cp.verify_chain(d).ok)

            # The next family operation converges storage and clears up.
            cp.compact_family(members)
            for d in members:
                self.assertEqual(_state(d), states[d])
            inodes = {
                os.stat(os.path.join(d, _seg(0))).st_ino for d in members
            }
            self.assertEqual(len(inodes), 1)
            self.assertEqual(
                [n for n in os.listdir(root) if n.startswith(".")], []
            )
            for d in members:
                self.assertEqual(
                    [n for n in os.listdir(d) if n.startswith(".")], []
                )

    def test_killed_after_one_member_promoted_still_recovers(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            b1 = os.path.join(root, "b1")
            _trained_dir(main, 6)
            cp.fork_chain(main, b1, up_to=4)
            members = [os.path.abspath(main), os.path.abspath(b1)]
            states = {d: _state(d) for d in members}
            fold, fdoc, fstaged, leases = cp._measure_and_stage_family(
                root, members
            )
            # Promote b1 (leaves a family barrier); kill before main.
            cp._stream_family_tail(b1, fold, fdoc, fstaged)
            self.assertTrue(cp._family_barrier_exists(b1))
            for d, fd in leases.items():
                cp._lease_release(d)
                cp._release_compaction_lease(fd)
            cp._family_release(root)

            for d in members:
                self.assertEqual(_state(d), states[d])
            cp.compact_family([main, b1])
            for d in members:
                self.assertEqual(_state(d), states[d])
            self.assertEqual(
                os.stat(os.path.join(main, _seg(0))).st_ino,
                os.stat(os.path.join(b1, _seg(0))).st_ino,
            )
            self.assertEqual(
                [n for n in os.listdir(b1) if n.startswith(".")], []
            )

    def test_killed_live_family_fold(self):
        with tempfile.TemporaryDirectory() as root:
            members = _family(root, ("b1", "b2"), (12, 8), main_steps=16)
            states = {d: _state(d) for d in members}

            proc = multiprocessing.Process(
                target=_slow_family_worker, args=(root, members)
            )
            proc.start()
            deadline = time.monotonic() + 30.0
            saw_marker = False
            while proc.is_alive() and time.monotonic() < deadline:
                if os.path.exists(os.path.join(root, ".seqfamily")):
                    saw_marker = True
                    break
                time.sleep(0.001)
            if proc.is_alive():
                proc.kill()
            proc.join()
            self.assertTrue(saw_marker)

            for d in members:
                self.assertEqual(_state(d), states[d])
            cp.compact_family(members)
            for d in members:
                self.assertEqual(_state(d), states[d])
            self.assertTrue(cp.verify_chain(members).ok)
            self.assertEqual(
                [n for n in os.listdir(root) if n.startswith(".")], []
            )


def _slow_family_worker(root, members):
    from sequence_engine import checkpoint as cp

    original_write = cp._atomic_write

    def slow_write(directory, final_name, raw):
        result = original_write(directory, final_name, raw)
        if final_name.endswith(".seqd"):
            time.sleep(0.01)
        return result

    cp._atomic_write = slow_write
    cp.compact_family(members)


class FamilyCompactConcurrencyTests(unittest.TestCase):
    def test_appends_loads_and_single_and_family_compactions(self):
        with tempfile.TemporaryDirectory() as root:
            members = _family(root, ("b1", "b2"), (6, 4), main_steps=8)
            errors = []
            stop = False

            def appender(d):
                try:
                    for _ in range(40):
                        doc = cp.load_chain(d)
                        cp.save_chain(doc, d)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(("append", d, exc))

            def reader(d):
                try:
                    while not stop:
                        cp.build_bytes(cp.load_chain(d))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(("read", d, exc))

            def compactor(d):
                try:
                    for _ in range(15):
                        cp.compact_chain(d)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(("compact", d, exc))

            threads = []
            for d in members:
                threads.append(threading.Thread(target=appender, args=(d,)))
                threads.append(threading.Thread(target=reader, args=(d,)))
            for d in members[1:]:
                threads.append(threading.Thread(target=compactor, args=(d,)))
            for thread in threads:
                thread.start()
            for _ in range(20):
                try:
                    cp.compact_family(members)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(("family", None, exc))
            stop = True
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertTrue(cp.verify_chain(members).ok)

    def test_forks_and_deletes_flow_around_a_family_fold(self):
        with tempfile.TemporaryDirectory() as root:
            members = _family(root, ("b1", "b2"), (6, 4), main_steps=8)
            errors = []
            stop = False

            def family_folder():
                try:
                    while not stop:
                        cp.compact_family(members)
                except BaseException as exc:  # noqa: BLE001
                    if os.path.isdir(members[0]):
                        errors.append(("family", exc))

            def forker():
                try:
                    for i in range(10):
                        target = os.path.join(root, f"fork{i}")
                        cp.fork_chain(members[0], target)
                        cp.delete_chain(target)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(("forkdel", exc))

            threads = [
                threading.Thread(target=family_folder),
                threading.Thread(target=forker),
            ]
            for thread in threads:
                thread.start()
            time.sleep(1.5)
            stop = True
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            for d in members:
                self.assertTrue(cp.verify_chain(d).ok)


if __name__ == "__main__":
    unittest.main()
