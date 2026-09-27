"""Tests for family archive export and import.

A family export packs every segment every member's head reaches into one
self-contained archive file (shared segments stored once, each member's
name, head and segment layout recorded); an import restores the archive
as a chain family with the same member count, segment layout and
sharing.  These tests cover:

* the round trip: every member's reassembled state (parameters,
  gradients, optimizer state, hidden state) is bit for bit preserved,
  heads and segment counts match, the one-copy sharing layout is
  reproduced, and no member's optimizer step moves;
* export determinism (the same family exports to identical bytes) and
  re-exporting an imported family reproducing the original archive;
* the error taxonomy (FileNotFoundError / ValueError / OSError /
  TypeError) with a rejection changing no byte of the family or target;
* archive corruption (truncation, bad framing, missing or extra fields,
  out-of-order segments, shape/layer-order disagreement) rejecting the
  whole import with no partial files left behind;
* hard-kill safety: an import interrupted mid-flight is finished by
  re-running it, bit for bit the one-run result, with the staging
  residue reclaimed by the next family operation;
* concurrency with saves, appends and compactions on the members,
  serialised by the directory locks and leases.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import threading
import unittest
import zlib
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


def _digest(directory):
    """A byte-and-inode fingerprint of every file directly in *directory*."""
    out = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                out[name] = (fh.read(), os.stat(path).st_ino)
    return out


def _sharing_groups(base, names):
    """The sets of (member, segment) pairs sharing one physical file."""
    groups = {}
    for name in names:
        for entry in os.listdir(os.path.join(base, name)):
            if entry.startswith("seg-"):
                ino = os.stat(os.path.join(base, name, entry)).st_ino
                groups.setdefault(ino, set()).add((name, entry))
    return sorted(frozenset(group) for group in groups.values())


def _read_archive(raw):
    """Split a family archive into its JSON header and payload bytes."""
    (header_len,) = struct.unpack("<Q", raw[12:20])
    header = json.loads(raw[20 : 20 + header_len])
    payload = raw[20 + header_len : raw.rfind(cp._ARCHIVE_END_MAGIC)]
    return header, payload


def _reframe(header, payload):
    """Re-encode an archive from a (possibly mutated) header and payload."""
    blobs = []
    offset = 0
    for length in header.get("blobs", []):
        blobs.append(payload[offset : offset + length])
        offset += length
    return cp._frame_archive(header, blobs)


class FamilyExportImportBasicsTests(unittest.TestCase):
    def test_round_trip_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            _grow(b1, 1, start=4)
            members = [main, b1, b2]
            states = {d: _state(d) for d in members}
            steps = {d: _step(d) for d in members}
            heads = {d: _head(d) for d in members}
            counts = {
                d: sum(1 for n in os.listdir(d) if n.startswith("seg-"))
                for d in members
            }

            archive = os.path.join(root, "family.seqfa")
            self.assertIsNone(cp.export_family(members, archive))
            target = os.path.join(root, "restored")
            os.mkdir(target)
            self.assertIsNone(cp.import_family(archive, target))

            self.assertEqual(sorted(os.listdir(target)), ["b1", "b2", "main"])
            restored = [os.path.join(target, os.path.basename(d)) for d in members]
            for original, new in zip(members, restored):
                self.assertEqual(_state(new), states[original])
                self.assertEqual(_step(new), steps[original])
                self.assertEqual(_head(new), heads[original])
                self.assertEqual(
                    sum(1 for n in os.listdir(new) if n.startswith("seg-")),
                    counts[original],
                )
            # The one-copy sharing layout is reproduced: the shared prefix
            # is one physical file per segment, member tails are private.
            self.assertEqual(
                _sharing_groups(root, ("main", "b1", "b2")),
                _sharing_groups(target, ("main", "b1", "b2")),
            )
            # The restored family verifies read-only.
            report = cp.verify_chain(restored)
            self.assertTrue(report.ok)
            self.assertEqual(len(report.chains), 3)

    def test_round_trip_after_family_compaction_and_merge(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            members = [main, b1, b2]
            cp.compact_family(members)
            cp.merge_chains(b1, b2)
            states = {d: _state(d) for d in members}
            archive = os.path.join(root, "family.seqfa")
            cp.export_family(members, archive)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            cp.import_family(archive, target)
            restored = [os.path.join(target, os.path.basename(d)) for d in members]
            for original, new in zip(members, restored):
                self.assertEqual(_state(new), states[original])
            # The shared folded basis is one physical file again.
            self.assertEqual(
                len(
                    {
                        os.stat(os.path.join(target, n, _seg(0))).st_ino
                        for n in ("main", "b1", "b2")
                    }
                ),
                1,
            )
            self.assertTrue(cp.verify_chain(restored).ok)

    def test_export_is_deterministic_and_advances_no_step(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=5)
            members = [main, b1]
            steps = {d: _step(d) for d in members}
            digests = {d: _digest(d) for d in members}

            first = os.path.join(root, "one.seqfa")
            second = os.path.join(root, "two.seqfa")
            cp.export_family(members, first)
            cp.export_family(members, second)
            with open(first, "rb") as fh:
                raw_one = fh.read()
            with open(second, "rb") as fh:
                raw_two = fh.read()
            self.assertEqual(raw_one, raw_two)
            # The export moved no optimizer step and touched no member byte.
            for d in members:
                self.assertEqual(_step(d), steps[d])
                self.assertEqual(_digest(d), digests[d])

    def test_reexport_of_imported_family_reproduces_archive(self):
        with tempfile.TemporaryDirectory() as root:
            members = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            _grow(members[1], 2, start=4)
            archive = os.path.join(root, "family.seqfa")
            cp.export_family(members, archive)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            cp.import_family(archive, target)
            restored = [os.path.join(target, os.path.basename(d)) for d in members]
            again = os.path.join(root, "again.seqfa")
            cp.export_family(restored, again)
            with open(archive, "rb") as fh:
                first_bytes = fh.read()
            with open(again, "rb") as fh:
                self.assertEqual(first_bytes, fh.read())

    def test_single_member_family(self):
        with tempfile.TemporaryDirectory() as root:
            (main,) = _family(root, (), (), main_steps=4)
            state = _state(main)
            archive = os.path.join(root, "single.seqfa")
            cp.export_family([main], archive)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            cp.import_family(archive, target)
            self.assertEqual(os.listdir(target), ["main"])
            self.assertEqual(_state(os.path.join(target, "main")), state)

    def test_export_overwrites_existing_archive(self):
        with tempfile.TemporaryDirectory() as root:
            (main,) = _family(root, (), (), main_steps=3)
            archive = os.path.join(root, "family.seqfa")
            cp.export_family([main], archive)
            with open(archive, "rb") as fh:
                first = fh.read()
            _grow(main, 2, start=3)
            cp.export_family([main], archive)
            with open(archive, "rb") as fh:
                second = fh.read()
            self.assertNotEqual(first, second)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            cp.import_family(archive, target)
            self.assertEqual(_state(os.path.join(target, "main")), _state(main))

    def test_import_advances_no_optimizer_step(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=5)
            archive = os.path.join(root, "family.seqfa")
            cp.export_family([main, b1], archive)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            cp.import_family(archive, target)
            for name, original in (("main", main), ("b1", b1)):
                self.assertEqual(
                    _step(os.path.join(target, name)), _step(original)
                )

    def test_sequential_entry_points(self):
        seq, _ = _stack()
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=5)
            archive = os.path.join(root, "family.seqfa")
            self.assertIsNone(seq.export_family([main, b1], archive))
            target = os.path.join(root, "restored")
            os.mkdir(target)
            self.assertIsNone(seq.import_family(archive, target))
            self.assertEqual(
                _state(os.path.join(target, "main")), _state(main)
            )
            self.assertEqual(_state(os.path.join(target, "b1")), _state(b1))

    def test_completed_import_leaves_no_residue(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=5)
            archive = os.path.join(root, "family.seqfa")
            cp.export_family([main, b1], archive)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            cp.import_family(archive, target)
            self.assertEqual(
                [n for n in os.listdir(target) if n.startswith(".")], []
            )
            self.assertEqual(sorted(os.listdir(target)), ["b1", "main"])
            # The export staging is gone from the family directory too.
            self.assertEqual(
                [n for n in os.listdir(root) if n.startswith(".seq")], []
            )


class FamilyExportImportErrorTests(unittest.TestCase):
    def test_missing_member_directory_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (2,), main_steps=4)
            with self.assertRaises(FileNotFoundError):
                cp.export_family(
                    [main, b1, os.path.join(root, "ghost")],
                    os.path.join(root, "a.seqfa"),
                )
            self.assertFalse(os.path.exists(os.path.join(root, "a.seqfa")))

    def test_missing_referenced_segment_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=4)
            os.unlink(os.path.join(b1, _seg(2)))
            with self.assertRaises(FileNotFoundError):
                cp.export_family([main, b1], os.path.join(root, "a.seqfa"))
            self.assertTrue(cp.verify_chain(main).ok)

    def test_missing_archive_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            target = os.path.join(root, "restored")
            os.mkdir(target)
            with self.assertRaises(FileNotFoundError):
                cp.import_family(os.path.join(root, "ghost.seqfa"), target)
            self.assertEqual(os.listdir(target), [])

    def test_missing_import_target_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            (main,) = _family(root, (), (), main_steps=3)
            archive = os.path.join(root, "a.seqfa")
            cp.export_family([main], archive)
            with self.assertRaises(FileNotFoundError):
                cp.import_family(archive, os.path.join(root, "ghost"))

    def test_type_and_structure_checks(self):
        with tempfile.TemporaryDirectory() as root:
            (main,) = _family(root, (), (), main_steps=2)
            archive = os.path.join(root, "a.seqfa")
            with self.assertRaises(TypeError):
                cp.export_family(main, archive)  # a bare path is not a family
            with self.assertRaises(TypeError):
                cp.export_family([main, 123], archive)
            with self.assertRaises(TypeError):
                cp.export_family([main], 123)
            with self.assertRaises(ValueError):
                cp.export_family([], archive)
            with self.assertRaises(ValueError):
                cp.export_family([main, main], archive)
            other_parent = tempfile.mkdtemp()
            try:
                foreign = os.path.join(other_parent, "foreign")
                os.mkdir(foreign)
                seq, _ = _stack()
                seq.save(foreign)
                with self.assertRaises(ValueError):
                    cp.export_family([main, foreign], archive)
            finally:
                import shutil

                shutil.rmtree(other_parent)
            # The export target must not be a directory or live inside a
            # member chain.
            with self.assertRaises(ValueError):
                cp.export_family([main], main)
            with self.assertRaises(ValueError):
                cp.export_family([main], os.path.join(main, "inside.seqfa"))
            # Import argument checks.
            with self.assertRaises(TypeError):
                cp.import_family(123, root)
            with self.assertRaises(TypeError):
                cp.import_family(archive, 123)
            with self.assertRaises(ValueError):
                cp.import_family(main, root)  # the archive is a directory
            cp.export_family([main], archive)
            with self.assertRaises(ValueError):
                cp.import_family(archive, archive)  # target not a directory

    def test_occupied_target_is_valueerror_and_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=5)
            archive = os.path.join(root, "family.seqfa")
            cp.export_family([main, b1], archive)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            cp.import_family(archive, target)
            before = {n: _digest(os.path.join(target, n)) for n in ("main", "b1")}
            listing = sorted(os.listdir(target))
            with self.assertRaises(ValueError):
                cp.import_family(archive, target)
            # Not one byte of the existing family changed and nothing was
            # added (no marker, no staging residue).
            self.assertEqual(sorted(os.listdir(target)), listing)
            for name, digest in before.items():
                self.assertEqual(_digest(os.path.join(target, name)), digest)

    def test_corrupt_chain_rejects_export_without_touching_members(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            members = [main, b1, b2]
            digests = {d: _digest(d) for d in members}
            tail = os.path.join(b1, _seg(3))
            with open(tail, "rb") as fh:
                good = fh.read()
            with open(tail, "r+b") as fh:
                fh.truncate(len(good) // 2)
            with self.assertRaises(ValueError):
                cp.export_family(members, os.path.join(root, "a.seqfa"))
            with open(tail, "wb") as fh:
                fh.write(good)
            for d in members:
                self.assertEqual(_digest(d), digests[d])
            self.assertFalse(os.path.exists(os.path.join(root, "a.seqfa")))
            # The failed export left no staging residue behind.
            self.assertEqual(
                [n for n in os.listdir(root) if n.startswith(".seq")], []
            )

    def test_shape_mismatch_rejects_export_naming_shapes_and_layers(self):
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
                cp.export_family([main, other], os.path.join(root, "a.seqfa"))
            message = str(ctx.exception)
            self.assertIn("parameter shapes", message)
            self.assertIn("layer order", message)
            self.assertFalse(os.path.exists(os.path.join(root, "a.seqfa")))
            self.assertTrue(cp.verify_chain(main).ok)
            self.assertTrue(cp.verify_chain(other).ok)

    def test_unwritable_export_destination_is_oserror(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write into read-only directories")
        with tempfile.TemporaryDirectory() as root:
            (main,) = _family(root, (), (), main_steps=3)
            destination = os.path.join(root, "out")
            os.mkdir(destination)
            os.chmod(destination, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.export_family([main], os.path.join(destination, "a.seqfa"))
            finally:
                os.chmod(destination, 0o755)

    def test_unwritable_import_target_is_oserror(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write into read-only directories")
        with tempfile.TemporaryDirectory() as root:
            (main,) = _family(root, (), (), main_steps=3)
            archive = os.path.join(root, "a.seqfa")
            cp.export_family([main], archive)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            os.chmod(target, 0o555)
            try:
                with self.assertRaises(OSError):
                    cp.import_family(archive, target)
            finally:
                os.chmod(target, 0o755)
            self.assertEqual(os.listdir(target), [])


class FamilyArchiveCorruptionTests(unittest.TestCase):
    def _archive(self, root, names=("b1",), points=(3,), main_steps=5):
        members = _family(root, names, points, main_steps=main_steps)
        archive = os.path.join(root, "family.seqfa")
        cp.export_family(members, archive)
        with open(archive, "rb") as fh:
            return members, fh.read()

    def _assert_rejected(self, root, raw):
        archive = os.path.join(root, "bad.seqfa")
        with open(archive, "wb") as fh:
            fh.write(raw)
        target = tempfile.mkdtemp(dir=root)
        with self.assertRaises(ValueError):
            cp.import_family(archive, target)
        # The rejection left no partial import behind.
        self.assertEqual(os.listdir(target), [])

    def test_truncated_archive(self):
        with tempfile.TemporaryDirectory() as root:
            _, raw = self._archive(root)
            self._assert_rejected(root, raw[: len(raw) // 2])
            self._assert_rejected(root, raw[:10])
            self._assert_rejected(root, raw[:-5])

    def test_bad_magic_and_crc(self):
        with tempfile.TemporaryDirectory() as root:
            _, raw = self._archive(root)
            self._assert_rejected(root, b"X" + raw[1:])
            # Flip one payload byte: the CRC catches it.
            flipped = bytearray(raw)
            flipped[-20] ^= 0x01
            self._assert_rejected(root, bytes(flipped))

    def test_unknown_version(self):
        with tempfile.TemporaryDirectory() as root:
            _, raw = self._archive(root)
            header, payload = _read_archive(raw)
            header["v"] = 99
            self._assert_rejected(root, _reframe(header, payload))

    def test_missing_and_extra_header_fields(self):
        with tempfile.TemporaryDirectory() as root:
            _, raw = self._archive(root)
            header, payload = _read_archive(raw)
            del header["blobs"]
            self._assert_rejected(root, _reframe(header, payload))
            header, payload = _read_archive(raw)
            header["extra"] = 1
            self._assert_rejected(root, _reframe(header, payload))
            header, payload = _read_archive(raw)
            del header["members"][0]["head"]
            self._assert_rejected(root, _reframe(header, payload))

    def test_out_of_order_segments(self):
        with tempfile.TemporaryDirectory() as root:
            _, raw = self._archive(root, main_steps=5)
            header, payload = _read_archive(raw)
            # Swap two delta blobs of the main member: the segment numbers
            # no longer match their positions.
            segs = header["members"][0]["segs"]
            self.assertGreaterEqual(len(segs), 3)
            segs[1], segs[2] = segs[2], segs[1]
            self._assert_rejected(root, _reframe(header, payload))

    def test_head_and_segment_table_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            _, raw = self._archive(root)
            header, payload = _read_archive(raw)
            header["members"][0]["head"] += 1
            self._assert_rejected(root, _reframe(header, payload))
            header, payload = _read_archive(raw)
            header["members"][0]["segs"][0] = 999
            self._assert_rejected(root, _reframe(header, payload))

    def test_unreferenced_blob(self):
        with tempfile.TemporaryDirectory() as root:
            _, raw = self._archive(root)
            header, payload = _read_archive(raw)
            header["blobs"].append(4)
            payload += b"junk"
            self._assert_rejected(root, _reframe(header, payload))

    def test_bad_member_names(self):
        with tempfile.TemporaryDirectory() as root:
            _, raw = self._archive(root)
            for bad in ("a/b", ".", "..", ".seqimport", ""):
                header, payload = _read_archive(raw)
                header["members"][0]["name"] = bad
                self._assert_rejected(root, _reframe(header, payload))
            header, payload = _read_archive(raw)
            header["members"][1]["name"] = header["members"][0]["name"]
            self._assert_rejected(root, _reframe(header, payload))

    def test_cross_member_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            members, raw = self._archive(root)
            # A second, differently shaped single-member family.
            weights = _base_weights()
            weights["b1"] = [0.01, -0.02]  # [2] instead of [3]
            other = os.path.join(root, "other")
            os.mkdir(other)
            seq, _ = _stack(weights)
            seq.save(other)
            other_archive = os.path.join(root, "other.seqfa")
            cp.export_family([other], other_archive)
            with open(other_archive, "rb") as fh:
                other_raw = fh.read()
            header, payload = _read_archive(raw)
            other_header, other_payload = _read_archive(other_raw)
            offset = len(header["blobs"])
            header["blobs"] += other_header["blobs"]
            merged_member = other_header["members"][0]
            merged_member["segs"] = [i + offset for i in merged_member["segs"]]
            header["members"].append(merged_member)
            payload += other_payload
            bad = os.path.join(root, "bad.seqfa")
            with open(bad, "wb") as fh:
                fh.write(_reframe(header, payload))
            target = os.path.join(root, "restored")
            os.mkdir(target)
            with self.assertRaises(ValueError) as ctx:
                cp.import_family(bad, target)
            self.assertIn("parameter shapes", str(ctx.exception))
            self.assertEqual(os.listdir(target), [])

    def test_empty_member_list(self):
        with tempfile.TemporaryDirectory() as root:
            _, raw = self._archive(root)
            header, payload = _read_archive(raw)
            header["members"] = []
            self._assert_rejected(root, _reframe(header, payload))


class FamilyImportInterruptionTests(unittest.TestCase):
    def _killed_import(self, archive, target, kill_at):
        """Run an import killed (hard) during the member rename phase."""
        real_rename = os.rename
        calls = {"n": 0}

        def flaky_rename(src, dst):
            calls["n"] += 1
            if calls["n"] == kill_at:
                raise KeyboardInterrupt("simulated kill")
            return real_rename(src, dst)

        with mock.patch.object(cp.os, "rename", flaky_rename):
            with self.assertRaises(KeyboardInterrupt):
                cp.import_family(archive, target)

    def test_interrupted_import_resumes_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1, b2 = _family(root, ("b1", "b2"), (4, 2), main_steps=6)
            _grow(b1, 1, start=4)
            members = [main, b1, b2]
            archive = os.path.join(root, "family.seqfa")
            cp.export_family(members, archive)

            reference = os.path.join(root, "reference")
            os.mkdir(reference)
            cp.import_family(archive, reference)

            target = os.path.join(root, "restored")
            os.mkdir(target)
            self._killed_import(archive, target, kill_at=2)
            # The resume marker stands and one member landed.
            self.assertTrue(os.path.exists(os.path.join(target, ".seqimport")))
            cp.import_family(archive, target)

            names = ("main", "b1", "b2")
            for name in names:
                done = os.path.join(target, name)
                ref = os.path.join(reference, name)
                self.assertEqual(sorted(os.listdir(done)), sorted(os.listdir(ref)))
                for entry in os.listdir(done):
                    with open(os.path.join(done, entry), "rb") as fh:
                        done_bytes = fh.read()
                    with open(os.path.join(ref, entry), "rb") as fh:
                        self.assertEqual(done_bytes, fh.read())
            # The sharing layout is the one-run layout, and no residue or
            # marker remains.
            self.assertEqual(
                _sharing_groups(reference, names), _sharing_groups(target, names)
            )
            self.assertEqual(
                [n for n in os.listdir(target) if n.startswith(".")], []
            )
            self.assertTrue(
                cp.verify_chain([os.path.join(target, n) for n in names]).ok
            )

    def test_interrupted_before_any_member_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=5)
            archive = os.path.join(root, "family.seqfa")
            cp.export_family([main, b1], archive)
            states = {d: _state(d) for d in (main, b1)}
            target = os.path.join(root, "restored")
            os.mkdir(target)
            self._killed_import(archive, target, kill_at=1)
            self.assertTrue(os.path.exists(os.path.join(target, ".seqimport")))
            cp.import_family(archive, target)
            for name, original in (("main", main), ("b1", b1)):
                self.assertEqual(
                    _state(os.path.join(target, name)), states[original]
                )
            self.assertEqual(
                [n for n in os.listdir(target) if n.startswith(".")], []
            )

    def test_staging_residue_is_swept_by_next_family_operation(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=5)
            archive = os.path.join(root, "family.seqfa")
            cp.export_family([main, b1], archive)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            cp.import_family(archive, target)
            # Dead staging residue of a killed import/export is reclaimed
            # by the next family operation in that directory.
            for prefix in (".seqimp.tmp-dead", ".seqexp.tmp-dead"):
                staging = os.path.join(target, prefix)
                os.mkdir(staging)
                with open(os.path.join(staging, "junk"), "wb") as fh:
                    fh.write(b"x")
            cp.compact_chain(os.path.join(target, "main"))
            self.assertEqual(
                [n for n in os.listdir(target) if n.startswith(".seq")], []
            )

    def test_foreign_interrupted_import_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=5)
            archive = os.path.join(root, "family.seqfa")
            cp.export_family([main, b1], archive)
            exported_main = _state(main)
            other = os.path.join(root, "other.seqfa")
            _grow(main, 1, start=5)
            cp.export_family([main, b1], other)
            target = os.path.join(root, "restored")
            os.mkdir(target)
            # An interrupted import of one archive stands in the target;
            # importing a different archive there is refused.
            with open(archive, "rb") as fh:
                archive_id = f"{zlib.crc32(fh.read()) & 0xFFFFFFFF:08x}"
            cp._atomic_write(
                target,
                ".seqimport",
                cp._encode_import_marker(
                    {"v": 1, "id": archive_id, "m": ["main", "b1"]}
                ),
            )
            with self.assertRaises(ValueError):
                cp.import_family(other, target)
            self.assertEqual(os.listdir(target), [".seqimport"])
            # Re-running the archive the marker names finishes the import.
            cp.import_family(archive, target)
            self.assertEqual(sorted(os.listdir(target)), ["b1", "main"])
            self.assertEqual(_state(os.path.join(target, "main")), exported_main)


class FamilyExportImportConcurrencyTests(unittest.TestCase):
    def test_appends_and_compactions_during_export(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=5)
            members = [main, b1]
            archive = os.path.join(root, "family.seqfa")
            stop = threading.Event()
            errors = []

            def appender():
                seq, _ = _stack()
                try:
                    hidden = seq.load(b1)
                    while not stop.is_set():
                        out, hidden = seq.forward(Tensor(_SEG1), hidden)
                        seq.backward(_total(out))
                        seq.update(_LR)
                        seq.save(b1)
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

            def compactor():
                try:
                    while not stop.is_set():
                        try:
                            cp.compact_chain(main)
                        except ValueError:
                            pass
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

            threads = [
                threading.Thread(target=appender),
                threading.Thread(target=compactor),
            ]
            for thread in threads:
                thread.start()
            for _ in range(20):
                cp.export_family(members, archive)
            stop.set()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            # The exported snapshot is one consistent family: it imports
            # and verifies read-only.
            target = os.path.join(root, "restored")
            os.mkdir(target)
            cp.import_family(archive, target)
            restored = [os.path.join(target, n) for n in ("main", "b1")]
            self.assertTrue(cp.verify_chain(restored).ok)

    def test_exports_serialise_with_each_other(self):
        with tempfile.TemporaryDirectory() as root:
            main, b1 = _family(root, ("b1",), (3,), main_steps=5)
            members = [main, b1]
            errors = []

            def exporter(path):
                try:
                    for _ in range(10):
                        cp.export_family(members, path)
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

            first = os.path.join(root, "one.seqfa")
            second = os.path.join(root, "two.seqfa")
            threads = [
                threading.Thread(target=exporter, args=(first,)),
                threading.Thread(target=exporter, args=(second,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            with open(first, "rb") as fh:
                first_bytes = fh.read()
            with open(second, "rb") as fh:
                self.assertEqual(first_bytes, fh.read())


if __name__ == "__main__":
    unittest.main()
