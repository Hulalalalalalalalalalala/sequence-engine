"""Tests for whole-family export and import.

Export packs a chain family into one self-contained artifact: shared
segments are stored once as deduplicated blocks, every member's head and
its member-owned tail positions are preserved, and import restores the
family with the same member count, segment positions, hard-link sharing
layout and bit-for-bit state (parameters, gradients, optimizer state
and step count, hidden state).  These tests cover:

* the round trip: members, heads, positions, shared blocks, private
  tails, bitwise state, no optimizer step advanced;
* deterministic artifacts (repeated exports are byte-identical) and
  read-only, append-free exports;
* the error taxonomy (FileNotFoundError / ValueError / OSError /
  TypeError), including the guarantee that a rejected export changes no
  member and a rejected import changes no byte at the target;
* artifact rejection (truncation, corruption, missing fields,
  reordering, shape/layer-order disagreement);
* hard-kill safety: an import killed while staging members leaves only
  complete members or staging debris, re-running it finishes the family
  with a result bit for bit identical to an uninterrupted import and the
  debris is swept by the next family operation;
* concurrency with saves, loads and verifies during export, and family
  operations (fork, merge, compact, delete, verify) on the restored
  family;
* the Sequential-level entry points.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
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


def _step(td):
    return cp.load_chain(td)["optim"]["t"]


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _segments(td):
    return sorted(name for name in os.listdir(td) if name.startswith("seg-"))


def _family(root, names, fork_points, main_steps=8):
    main = os.path.join(root, "main")
    _trained_dir(main, main_steps)
    members = [main]
    for name, point in zip(names, fork_points):
        target = os.path.join(root, name)
        cp.fork_chain(main, target, up_to=point)
        members.append(target)
    return members


class ExportImportBasicsTests(unittest.TestCase):
    def test_round_trip_preserves_members_heads_positions_and_state(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (6, 3), main_steps=8)
            # The branches grow private tails.
            seq, hidden = _stack()
            hidden = seq.load(b1)
            _train(seq, b1, 2, hidden=hidden, start=6)
            members = [main, b1, b2]
            states = {d: _state(d) for d in members}
            steps = {d: _step(d) for d in members}
            heads = {d: _read(os.path.join(d, "head")) for d in members}
            segs = {d: _segments(d) for d in members}

            artifact = os.path.join(root, "family.seqx")
            self.assertIsNone(cp.export_family(members, artifact))

            target = os.path.join(root, "restored")
            os.mkdir(target)
            self.assertIsNone(cp.import_family(artifact, target))

            self.assertEqual(sorted(os.listdir(target)), ["b1", "b2", "main"])
            for member in members:
                name = os.path.basename(member)
                restored = os.path.join(target, name)
                self.assertEqual(_state(restored), states[member])
                self.assertEqual(_step(restored), steps[member])
                self.assertEqual(
                    _read(os.path.join(restored, "head")), heads[member]
                )
                self.assertEqual(_segments(restored), segs[member])
            self.assertTrue(
                cp.verify_chain(
                    [os.path.join(target, name) for name in ("main", "b1", "b2")]
                ).ok
            )
            # No residue anywhere.
            self.assertEqual(
                [n for n in os.listdir(target) if n.startswith(".")], []
            )

    def test_shared_segments_are_one_file_and_tails_stay_private(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (6, 3), main_steps=8)
            seq, hidden = _stack()
            hidden = seq.load(b1)
            _train(seq, b1, 2, hidden=hidden, start=6)
            artifact = os.path.join(root, "family.seqx")
            cp.export_family([main, b1, b2], artifact)

            target = os.path.join(root, "restored")
            os.mkdir(target)
            cp.import_family(artifact, target)

            rmain, rb1, rb2 = (os.path.join(target, n) for n in ("main", "b1", "b2"))
            # The prefix shared by all three is stored once on disk.
            for index in range(4):
                name = _seg(index)
                inodes = {
                    os.stat(os.path.join(d, name)).st_ino for d in (rmain, rb1, rb2)
                }
                self.assertEqual(len(inodes), 1, index)
                self.assertEqual(os.stat(os.path.join(rmain, name)).st_nlink, 3)
            # main and b1 still share their longer common prefix.
            for index in range(4, 7):
                name = _seg(index)
                self.assertEqual(
                    os.stat(os.path.join(rmain, name)).st_ino,
                    os.stat(os.path.join(rb1, name)).st_ino,
                )
                self.assertEqual(os.stat(os.path.join(rmain, name)).st_nlink, 2)
            # b2 diverged at 3: no sharing beyond its fork point.
            self.assertNotEqual(
                os.stat(os.path.join(rmain, _seg(7))).st_ino,
                os.stat(os.path.join(rb1, _seg(7))).st_ino,
            )
            # Member-owned tails are private files with one link.
            self.assertEqual(os.stat(os.path.join(rb1, _seg(7))).st_nlink, 1)
            self.assertEqual(os.stat(os.path.join(rmain, _seg(8))).st_nlink, 1)

    def test_repeated_export_is_identical_and_changes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=5)
            members = [main, b1]
            states = {d: _state(d) for d in members}
            listings = {d: _segments(d) + ["head"] for d in members}
            first = os.path.join(root, "one.seqx")
            second = os.path.join(root, "two.seqx")
            cp.export_family(members, first)
            first_bytes = _read(first)
            for _ in range(3):
                cp.export_family(members, first)
                self.assertEqual(_read(first), first_bytes)
            cp.export_family(members, second)
            self.assertEqual(_read(second), first_bytes)
            for member in members:
                self.assertEqual(_state(member), states[member])
                self.assertEqual(_segments(member) + ["head"], listings[member])

    def test_export_advances_no_step_and_import_matches_bitwise(self):
        with tempfile.TemporaryDirectory() as root:
            main, = _family(root, (), (), main_steps=6)
            before_t = _step(main)
            artifact = os.path.join(root, "f.seqx")
            cp.export_family([main], artifact)
            self.assertEqual(_step(main), before_t)
            target = os.path.join(root, "out")
            os.mkdir(target)
            cp.import_family(artifact, target)
            restored = os.path.join(target, "main")
            self.assertEqual(_step(restored), before_t)
            self.assertEqual(_state(restored), _state(main))
            # The restored member keeps training exactly where the export
            # left off: an appended delta matches an uninterrupted chain.
            seq, hidden = _stack()
            hidden = seq.load(restored)
            _train(seq, restored, 2, hidden=hidden, start=6)
            ref_dir = os.path.join(root, "refmain")
            cp.fork_chain(main, ref_dir)
            rseq, rhidden = _stack()
            rhidden = rseq.load(ref_dir)
            _train(rseq, ref_dir, 2, hidden=rhidden, start=6)
            self.assertEqual(_state(restored), _state(ref_dir))


class ExportImportErrorTests(unittest.TestCase):
    def test_occupied_target_is_valueerror_and_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main, = _family(root, (), (), main_steps=3)
            artifact = os.path.join(root, "f.seqx")
            cp.export_family([main], artifact)

            resident = os.path.join(root, "resident")
            os.mkdir(resident)
            other = os.path.join(resident, "chain")
            os.mkdir(other)
            chain_seq, _ = _stack()
            chain_seq.save(other)
            _train(chain_seq, other, 1)
            before = {
                name: _read(os.path.join(other, name))
                for name in os.listdir(other)
            }
            with self.assertRaises(ValueError):
                cp.import_family(artifact, resident)
            # Not one byte of the family already there changed.
            self.assertEqual(sorted(os.listdir(resident)), ["chain"])
            for name, raw in before.items():
                self.assertEqual(_read(os.path.join(other, name)), raw)

            # Importing into the family's own parent is rejected too.
            with self.assertRaises(ValueError):
                cp.import_family(artifact, root)
            self.assertEqual(
                sorted(os.listdir(root)),
                ["f.seqx", "main", "resident"],
            )

    def test_missing_member_and_segment_are_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (2,), main_steps=4)
            artifact = os.path.join(root, "f.seqx")
            with self.assertRaises(FileNotFoundError):
                cp.export_family([main, os.path.join(root, "ghost")], artifact)
            self.assertFalse(os.path.exists(artifact))
            # A missing referenced segment is a FileNotFoundError.
            hole = os.path.join(b1, _seg(0))
            held = _read(hole)
            os.unlink(hole)
            try:
                with self.assertRaises(FileNotFoundError):
                    cp.export_family([main, b1], artifact)
            finally:
                with open(hole, "wb") as fh:
                    fh.write(held)
            self.assertFalse(os.path.exists(artifact))

    def test_corrupt_or_mismatched_member_is_valueerror_on_export(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (2,), main_steps=4)
            # Give the branch one private tail segment so corruption does
            # not touch the shared (hard-linked) prefix.
            bseq, bhidden = _stack()
            bhidden = bseq.load(b1)
            _train(bseq, b1, 1, hidden=bhidden, start=2)
            artifact = os.path.join(root, "f.seqx")
            main_state = _state(main)
            b1_state = _state(b1)

            tail = os.path.join(b1, _seg(3))
            original = _read(tail)
            with open(tail, "r+b") as fh:
                fh.truncate(len(original) // 2)
            with self.assertRaises(ValueError):
                cp.export_family([main, b1], artifact)
            with open(tail, "wb") as fh:
                fh.write(original)
            self.assertFalse(os.path.exists(artifact))
            self.assertEqual(_state(main), main_state)
            self.assertEqual(_state(b1), b1_state)

            # Members whose parameter shapes disagree reject the export.
            other_weights = _base_weights()
            other_weights["b1"] = [0.01, -0.02]
            other_seq, _ = _stack(other_weights)
            other = os.path.join(root, "other")
            os.mkdir(other)
            other_seq.save(other)
            with self.assertRaises(ValueError):
                cp.export_family([main, other], artifact)
            self.assertFalse(os.path.exists(artifact))

    def test_missing_artifact_and_target_are_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            target = os.path.join(root, "out")
            os.mkdir(target)
            with self.assertRaises(FileNotFoundError):
                cp.import_family(os.path.join(root, "nope.seqx"), target)
            main, = _family(root, (), (), main_steps=1)
            artifact = os.path.join(root, "f.seqx")
            cp.export_family([main], artifact)
            with self.assertRaises(FileNotFoundError):
                cp.import_family(artifact, os.path.join(root, "ghost"))
            # A plain file at the target path is not a directory either.
            flat = os.path.join(root, "flat")
            with open(flat, "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(ValueError):
                cp.import_family(artifact, flat)
            self.assertEqual(os.listdir(target), [])

    def test_corrupt_artifact_is_valueerror_and_target_stays_empty(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (2,), main_steps=4)
            artifact = os.path.join(root, "f.seqx")
            cp.export_family([main, b1], artifact)
            raw = _read(artifact)
            target = os.path.join(root, "out")
            os.mkdir(target)

            def reject(blob, message):
                bad = os.path.join(root, "bad.seqx")
                with open(bad, "wb") as fh:
                    fh.write(blob)
                with self.assertRaises(ValueError, msg=message):
                    cp.import_family(bad, target)
                self.assertEqual(os.listdir(target), [], message)

            reject(b"", "empty artifact")
            reject(b"not a family artifact", "foreign bytes")
            reject(raw[: len(raw) // 2], "truncated artifact")
            reject(raw[:-1], "torn trailer")
            flipped = bytearray(raw)
            flipped[len(flipped) // 2] ^= 0xFF
            reject(bytes(flipped), "corrupt block bytes")
            reject(raw[:8] + struct.pack("<I", 99) + raw[12:], "bad version")

    def test_out_of_order_and_structurally_bad_manifests_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (2,), main_steps=4)
            artifact = os.path.join(root, "f.seqx")
            cp.export_family([main, b1], artifact)
            raw = _read(artifact)
            target = os.path.join(root, "out")
            os.mkdir(target)

            def repack(mutate):
                hlen = struct.unpack("<Q", raw[12:20])[0]
                header = json.loads(raw[20 : 20 + hlen])
                end = raw.rfind(cp.FAMILY_EXPORT_END_MAGIC)
                blocks = raw[20 + hlen + 8 : end]
                mutate(header)
                new_header = json.dumps(
                    header, ensure_ascii=False, separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                body = (
                    raw[:8]
                    + struct.pack("<I", 1)
                    + struct.pack("<Q", len(new_header))
                    + new_header
                    + struct.pack("<Q", len(header["segments"]))
                    + blocks
                )
                crc = zlib.crc32(body[20:])
                bad = os.path.join(root, "bad.seqx")
                with open(bad, "wb") as fh:
                    fh.write(body + cp.FAMILY_EXPORT_END_MAGIC + struct.pack("<I", crc))
                return bad

            def reorder(header):
                reach = header["members"][0]["segments"]
                reach[0], reach[1] = reach[1], reach[0]

            with self.assertRaises(ValueError):
                cp.import_family(repack(reorder), target)
            with self.assertRaises(ValueError):
                cp.import_family(repack(lambda h: h.pop("members")), target)
            with self.assertRaises(ValueError):
                cp.import_family(repack(lambda h: h.update(extra=1)), target)

            def duplicate(header):
                header["members"].append(dict(header["members"][0]))

            with self.assertRaises(ValueError):
                cp.import_family(repack(duplicate), target)

            def missing_block(header):
                header["members"][0]["segments"].append(99999)
                header["members"][0]["head"] += 1

            with self.assertRaises(ValueError):
                cp.import_family(repack(missing_block), target)
            self.assertEqual(os.listdir(target), [])

    def test_unwritable_destination_raises_oserror(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write into read-only directories")
        with tempfile.TemporaryDirectory() as root:
            main, = _family(root, (), (), main_steps=2)
            os.chmod(root, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.export_family([main], os.path.join(root, "f.seqx"))
            finally:
                os.chmod(root, 0o755)
            with tempfile.TemporaryDirectory() as root2:
                main2, = _family(root2, (), (), main_steps=2)
                artifact = os.path.join(root2, "f.seqx")
                cp.export_family([main2], artifact)
                target = os.path.join(root2, "readonly")
                os.mkdir(target)
                os.chmod(target, 0o555)
                try:
                    with self.assertRaises(OSError):
                        cp.import_family(artifact, target)
                finally:
                    os.chmod(target, 0o755)
                self.assertEqual(os.listdir(target), [])

    def test_type_checks(self):
        with tempfile.TemporaryDirectory() as root:
            main, = _family(root, (), (), main_steps=1)
            artifact = os.path.join(root, "f.seqx")
            cp.export_family([main], artifact)
            with self.assertRaises(TypeError):
                cp.export_family(main, artifact)
            with self.assertRaises(TypeError):
                cp.export_family([123], artifact)
            with self.assertRaises(TypeError):
                cp.export_family([main], 123)
            with self.assertRaises(TypeError):
                cp.import_family(123, root)
            with self.assertRaises(TypeError):
                cp.import_family(artifact, 123)
            with self.assertRaises(ValueError):
                cp.export_family([], artifact)
            seq, _ = _stack()
            with self.assertRaises(TypeError):
                seq.export_family([main], 123)
            with self.assertRaises(TypeError):
                seq.import_family(artifact)


class ImportResumeTests(unittest.TestCase):
    def _artifact_and_states(self, root):
        main, b1, b2 = _family(root, ("b1", "b2"), (12, 6), main_steps=18)
        members = [main, b1, b2]
        artifact = os.path.join(root, "family.seqx")
        cp.export_family(members, artifact)
        states = {
            os.path.basename(d): _state(d)
            for d in members
        }
        return artifact, states

    def test_killed_import_resumes_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as root:
            artifact, states = self._artifact_and_states(root)
            oneshot = os.path.join(root, "oneshot")
            os.mkdir(oneshot)
            cp.import_family(artifact, oneshot)

            target = os.path.join(root, "restored")
            os.mkdir(target)
            proc = multiprocessing.Process(
                target=_slow_import_worker, args=(artifact, target)
            )
            proc.start()
            deadline = time.monotonic() + 30.0
            saw_progress = False
            while proc.is_alive() and time.monotonic() < deadline:
                if any(
                    name.startswith(".seqimp.tmp-") or name in ("main", "b1", "b2")
                    for name in os.listdir(target)
                ):
                    saw_progress = True
                    break
                time.sleep(0.001)
            if proc.is_alive():
                proc.kill()
            proc.join()
            self.assertTrue(saw_progress)

            # Whatever state the kill left: every member already at its
            # final name is a complete chain, and the target carries no
            # half member at a final name.
            for name in ("main", "b1", "b2"):
                path = os.path.join(target, name)
                if os.path.exists(path):
                    self.assertTrue(cp.verify_chain(path).ok)

            # Re-running the import finishes the family.
            cp.import_family(artifact, target)
            self.assertEqual(sorted(os.listdir(target)), ["b1", "b2", "main"])
            for name, state in states.items():
                self.assertEqual(_state(os.path.join(target, name)), state)
            # Bit for bit identical to the uninterrupted import, sharing
            # layout included.
            for name in ("main", "b1", "b2"):
                a = os.path.join(target, name)
                b = os.path.join(oneshot, name)
                self.assertEqual(_segments(a), _segments(b))
                self.assertEqual(
                    _read(os.path.join(a, "head")), _read(os.path.join(b, "head"))
                )
                for seg_name in _segments(a):
                    self.assertEqual(
                        _read(os.path.join(a, seg_name)),
                        _read(os.path.join(b, seg_name)),
                    )
            for index in range(7):
                paths = [
                    os.path.join(target, name, _seg(index))
                    for name in ("main", "b1", "b2")
                ]
                self.assertEqual(len({os.stat(p).st_ino for p in paths}), 1)
            self.assertEqual(
                [n for n in os.listdir(target) if n.startswith(".")], []
            )
            self.assertTrue(
                cp.verify_chain(
                    [os.path.join(target, n) for n in ("main", "b1", "b2")]
                ).ok
            )

    def test_resume_tolerates_appends_to_committed_members(self):
        with tempfile.TemporaryDirectory() as root:
            artifact, states = self._artifact_and_states(root)
            oneshot = os.path.join(root, "oneshot")
            os.mkdir(oneshot)
            cp.import_family(artifact, oneshot)

            target = os.path.join(root, "restored")
            os.mkdir(target)
            proc = multiprocessing.Process(
                target=_slow_import_worker, args=(artifact, target)
            )
            proc.start()
            deadline = time.monotonic() + 30.0
            while proc.is_alive() and time.monotonic() < deadline:
                if os.path.exists(os.path.join(target, "main", "head")):
                    break
                time.sleep(0.001)
            if proc.is_alive():
                proc.kill()
            proc.join()
            self.assertTrue(os.path.exists(os.path.join(target, "main")))

            # Training continues on the committed member while the import
            # is interrupted: the resume keeps the appended delta.
            seq, hidden = _stack()
            hidden = seq.load(os.path.join(target, "main"))
            _train(seq, os.path.join(target, "main"), 1,
                   hidden=hidden, start=18)
            cp.import_family(artifact, target)
            self.assertEqual(sorted(os.listdir(target)), ["b1", "b2", "main"])
            self.assertTrue(cp.verify_chain(os.path.join(target, "main")).ok)
            # The interrupted-import prefix matches the artifact exactly;
            # the post-interruption append is preserved.
            self.assertEqual(_read(os.path.join(target, "main", "head")), b"19")
            for name, state in states.items():
                if name == "main":
                    continue
                self.assertEqual(_state(os.path.join(target, name)), state)

    def test_resume_rejects_a_member_that_does_not_match(self):
        with tempfile.TemporaryDirectory() as root:
            artifact, states = self._artifact_and_states(root)
            digest = hashlib.sha256(_read(artifact)).hexdigest()
            target = os.path.join(root, "restored")
            os.mkdir(target)
            oneshot = os.path.join(root, "oneshot")
            os.mkdir(oneshot)
            cp.import_family(artifact, oneshot)
            # Fabricate a killed-import state: marker plus one tampered
            # member directory.
            with open(os.path.join(target, ".seqimport"), "wb") as fh:
                fh.write(json.dumps(
                    {"v": 1, "d": digest}, sort_keys=True, separators=(",", ":")
                ).encode("ascii"))
            os.rename(os.path.join(oneshot, "main"), os.path.join(target, "main"))
            victim = os.path.join(target, "main", _seg(1))
            with open(victim, "r+b") as fh:
                fh.write(b"tamper")
            listing_before = sorted(os.listdir(target))
            with self.assertRaises(ValueError):
                cp.import_family(artifact, target)
            self.assertEqual(sorted(os.listdir(target)), listing_before)
            self.assertFalse(os.path.exists(os.path.join(target, "b1")))

    def test_staging_debris_is_swept_by_the_next_family_operation(self):
        with tempfile.TemporaryDirectory() as root:
            artifact, _states = self._artifact_and_states(root)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            debris = os.path.join(target, ".seqimp.tmp-b1-dead")
            os.mkdir(debris)
            with open(os.path.join(debris, "junk"), "wb") as fh:
                fh.write(b"x")
            # A fresh import sweeps the dead staging first, then succeeds.
            cp.import_family(artifact, target)
            self.assertEqual(sorted(os.listdir(target)), ["b1", "b2", "main"])

            # And the same sweep runs as part of the other family
            # operations over a parent holding dead import staging.
            other = os.path.join(root, "swept")
            os.mkdir(other)
            more_debris = os.path.join(other, ".seqimp.tmp-main-dead")
            os.mkdir(more_debris)
            main, = _family(other, (), (), main_steps=2)
            self.assertTrue(os.path.isdir(more_debris))
            cp.compact_chain(main, up_to=0)
            self.assertFalse(os.path.exists(more_debris))


def _slow_import_worker(artifact, target):
    from sequence_engine import checkpoint as cp

    original_link = os.link

    def slow_link(src, dst):
        time.sleep(0.02)
        return original_link(src, dst)

    cp.os.link = slow_link
    try:
        cp.import_family(artifact, target)
    finally:
        cp.os.link = original_link


class ExportImportConcurrencyTests(unittest.TestCase):
    def test_saves_and_verifies_race_exports(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (4,), main_steps=6)
            members = [main, b1]
            errors = []
            stop = False

            def appender():
                try:
                    while not stop:
                        doc = cp.load_chain(main)
                        cp.save_chain(doc, main)  # empty deltas
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def branch_appender():
                try:
                    while not stop:
                        doc = cp.load_chain(b1)
                        cp.save_chain(doc, b1)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=appender),
                threading.Thread(target=branch_appender),
            ]
            for thread in threads:
                thread.start()
            try:
                for index in range(10):
                    artifact = os.path.join(root, f"f{index}.seqx")
                    cp.export_family(members, artifact)
                    target = os.path.join(root, f"out{index}")
                    os.mkdir(target)
                    cp.import_family(artifact, target)
                    self.assertTrue(
                        cp.verify_chain(
                            [os.path.join(target, n) for n in ("main", "b1")]
                        ).ok
                    )
            finally:
                stop = True
                for thread in threads:
                    thread.join()
            self.assertEqual(errors, [])
            self.assertTrue(cp.verify_chain(members).ok)
            # The last artifact restores a state the source family can
            # also reach (exports saw only committed chains).
            last = os.path.join(root, "out9")
            for name in ("main", "b1"):
                self.assertTrue(cp.verify_chain(os.path.join(last, name)).ok)

    def test_restored_family_takes_part_in_family_operations(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            seq, hidden = _stack()
            hidden = seq.load(b1)
            _train(seq, b1, 1, hidden=hidden, start=4)
            artifact = os.path.join(root, "f.seqx")
            cp.export_family([main, b1, b2], artifact)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            cp.import_family(artifact, target)
            rmain, rb1, rb2 = (
                os.path.join(target, n) for n in ("main", "b1", "b2")
            )

            # Fork from a restored member.
            rb3 = os.path.join(target, "b3")
            cp.fork_chain(rmain, rb3, up_to=3)
            self.assertTrue(cp.verify_chain([rmain, rb1, rb2, rb3]).ok)

            # Merge between two restored members.
            cp.merge_chains(rb1, rb2)
            self.assertEqual(_state(rb2), _state(rb1))

            # Family compaction folds the restored shared prefix once.
            cp.compact_family([rmain, rb1, rb2, rb3])
            for member in (rmain, rb1, rb2, rb3):
                self.assertTrue(cp.verify_chain(member).ok)
            name = _seg(0)
            self.assertEqual(
                len({os.stat(os.path.join(d, name)).st_ino
                     for d in (rmain, rb1, rb2, rb3)}),
                1,
            )

            # Deleting a branch reclaims the shared bytes by reachability.
            state_before = _state(rmain)
            cp.delete_chain(rb3)
            self.assertFalse(os.path.exists(rb3))
            self.assertEqual(_state(rmain), state_before)
            self.assertTrue(cp.verify_chain([rmain, rb1, rb2]).ok)

            # Verify is strictly read-only over the restored family.
            def listing(d):
                return {n: _read(os.path.join(d, n)) for n in os.listdir(d)}

            snapshots = {d: listing(d) for d in (rmain, rb1, rb2)}
            cp.verify_chain([rmain, rb1, rb2])
            for d, snapshot in snapshots.items():
                self.assertEqual(listing(d), snapshot)


class SequentialExportImportTests(unittest.TestCase):
    def test_sequential_directory_family_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            seq, _hidden = _stack()
            main = os.path.join(root, "main")
            os.mkdir(main)
            seq.save(main)
            out, hidden = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out))
            seq.adam_step(_ADAM_LR)
            seq.save(main)
            b1 = os.path.join(root, "b1")
            seq.fork(main, b1, up_to=1)
            artifact = os.path.join(root, "f.seqx")
            self.assertIsNone(seq.export_family([main, b1], artifact))
            target = os.path.join(root, "out")
            os.mkdir(target)
            self.assertIsNone(seq.import_family(artifact, target))
            self.assertEqual(sorted(os.listdir(target)), ["b1", "main"])
            self.assertEqual(
                _state(os.path.join(target, "main")), _state(main)
            )
            self.assertEqual(
                _state(os.path.join(target, "b1")), _state(b1)
            )

    def test_sequential_memory_family_round_trip(self):
        seq, _ = _stack()
        main = MemoryChain()
        seq.save(main)
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        seq.update(_LR)
        seq.save(main)
        branch = seq.fork(main, up_to=0)
        artifact = seq.export_family([main, branch])
        self.assertIsInstance(artifact, bytes)
        restored = seq.import_family(artifact)
        self.assertEqual(len(restored), 2)
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(restored[0])),
            cp.build_bytes(cp.load_chain_memory(main)),
        )
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(restored[1])),
            cp.build_bytes(cp.load_chain_memory(branch)),
        )


if __name__ == "__main__":
    unittest.main()
