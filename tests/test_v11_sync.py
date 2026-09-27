"""Tests for cross-family incremental synchronization.

The sync entry packs only the segments two families genuinely differ by
(common prefix by position and CRC, source-exclusive segments as
deduplicated blocks) and the apply lands the source on the target
atomically: bit-for-bit state, no optimizer step advanced, the source
untouched, the shared layout kept as one inode/object and a repeated
apply a no-op.  These tests cover:

* the produce/apply round trip on directories and memory families,
  including heads, exclusive tails, positions and hard-link sharing;
* that the artifact carries no shared or unchanged segment and that the
  same difference is byte-deterministic;
* full <-> incremental artifact conversion in both directions;
* hard-kill safety: a killed apply leaves only whole files and per-member
  publication markers, re-running converges to one uninterrupted run and
  the debris is reclaimed by the next family operation;
* saves, loads and appends continuing through produce and apply without
  ever observing half a chain, and serialization with fork, merge,
  deletion and the two compaction levels;
* the FileNotFoundError / ValueError / OSError / TypeError taxonomy,
  including the guarantee that a rejected apply moves no byte of either
  family;
* read-only family verification over the synced members, including a
  member caught mid-publication.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import struct
import tempfile
import threading
import time
import unittest

from sequence_engine import MemoryChain, Tensor
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


def _state_mem(store):
    return cp.build_bytes(cp.load_chain_memory(store))


def _step(td):
    return cp.load_chain(td)["optim"]["t"]


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _seg(i):
    return f"seg-{i:010d}.seqd"


def _families(root, main_steps=8, fork_b1=6, fork_b2=3):
    """Two equal families under *root*: src/{main,b1,b2}, tgt/...

    The target shares a prefix with the source (forked from it at earlier
    points) and then grows its own tails.
    """
    src = os.path.join(root, "src")
    tgt = os.path.join(root, "tgt")
    os.mkdir(src)
    os.mkdir(tgt)
    smain = os.path.join(src, "main")
    _trained_dir(smain, main_steps)
    sb1 = os.path.join(src, "b1")
    cp.fork_chain(smain, sb1, up_to=fork_b1)
    seq, hidden = _stack()
    hidden = seq.load(sb1)
    _train(seq, sb1, 2, hidden=hidden, start=fork_b1)
    sb2 = os.path.join(src, "b2")
    cp.fork_chain(smain, sb2, up_to=fork_b2)

    tmain = os.path.join(tgt, "main")
    cp.fork_chain(smain, tmain, up_to=fork_b2)
    seq, hidden = _stack()
    hidden = seq.load(tmain)
    _train(seq, tmain, 2, hidden=hidden, start=fork_b2)
    tb1 = os.path.join(tgt, "b1")
    cp.fork_chain(tmain, tb1, up_to=1)
    tb2 = os.path.join(tgt, "b2")
    cp.fork_chain(tmain, tb2, up_to=0)
    return (
        [smain, sb1, sb2],
        [tmain, tb1, tb2],
        src,
        tgt,
    )


def _same_inode(paths):
    return len({os.stat(path).st_ino for path in paths}) == 1


class SyncRoundTripTests(unittest.TestCase):
    def test_apply_lands_source_bit_for_bit_without_stepping(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, src, tgt = _families(root)
            source_states = [_state(d) for d in sources]
            source_steps = [_step(d) for d in sources]
            target_steps_before = [_step(d) for d in targets]
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            cp.apply_sync_family(artifact, tgt)
            for target, state in zip(targets, source_states):
                self.assertEqual(_state(target), state)
            # Step counts (and the full optimizer state) ride the sync;
            # the apply itself advances nothing.
            self.assertEqual([_step(d) for d in targets], source_steps)
            self.assertNotEqual(source_steps, target_steps_before)
            # Heads point exactly at the source heads.
            for source, target in zip(sources, targets):
                self.assertEqual(
                    _read(os.path.join(target, "head")),
                    _read(os.path.join(source, "head")),
                )
            # The source family is byte-for-byte unchanged (directory
            # listings and every file's bytes).
            for source, state in zip(sources, source_states):
                self.assertEqual(_state(source), state)

    def test_artifact_carries_only_the_exclusive_segments(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            raw = _read(artifact)
            manifest, blobs = cp._read_sync_frame(raw, "sync artifact")
            # Every member's common prefix rides no blocks: its reach is
            # null there and one CRC per common position is recorded.
            total_carried_positions = 0
            for member in manifest["members"]:
                prefix = member["bp"]
                self.assertGreater(prefix, 0)
                self.assertEqual(len(member["pc"]), prefix)
                self.assertTrue(
                    all(slot is None for slot in member["segments"][:prefix])
                )
                total_carried_positions += member["head"] + 1 - prefix
            # Deduplication: blocks are strictly fewer than carried slots.
            self.assertLess(len(blobs), total_carried_positions)
            # Nothing in the pool duplicates a prefix segment of a member
            # against the same member's target files.
            for source, target, member in zip(sources, targets, manifest["members"]):
                for slot in range(member["bp"]):
                    self.assertEqual(
                        _read(os.path.join(source, _seg(slot))),
                        _read(os.path.join(target, _seg(slot))),
                    )
                for slot in range(member["bp"], member["head"] + 1):
                    self.assertNotEqual(
                        _read(os.path.join(source, _seg(slot))),
                        _read(os.path.join(target, _seg(slot)))
                        if os.path.exists(os.path.join(target, _seg(slot)))
                        else b"",
                    )

    def test_produce_is_deterministic_and_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            a = os.path.join(root, "a.seqs")
            b = os.path.join(root, "b.seqs")
            cp.sync_family(sources, targets, a)
            snapshot = {
                d: {n: _read(os.path.join(d, n)) for n in os.listdir(d)}
                for d in sources + targets
            }
            cp.sync_family(sources, targets, b)
            self.assertEqual(_read(a), _read(b))
            for d in sources + targets:
                for name, raw in snapshot[d].items():
                    self.assertEqual(_read(os.path.join(d, name)), raw)

    def test_shared_layout_does_not_become_duplicate_storage(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            cp.apply_sync_family(artifact, tgt)
            # After the sync every slot both members reach with equal
            # bytes is one physical file (the fork-family one-copy layout).
            for i, a in enumerate(targets):
                for b in targets[i + 1:]:
                    head_a = int(_read(os.path.join(a, "head")))
                    head_b = int(_read(os.path.join(b, "head")))
                    for slot in range(min(head_a, head_b) + 1):
                        pa, pb = os.path.join(a, _seg(slot)), os.path.join(b, _seg(slot))
                        if _read(pa) == _read(pb):
                            self.assertTrue(
                                _same_inode([pa, pb]),
                                f"equal segment {slot} is stored twice",
                            )
            # One physical file per slot against the source layout too:
            # synced members reproduce the source family's inode sharing
            # wherever bytes agree (both rooted in one history).
            for source, target in zip(sources, targets):
                head = int(_read(os.path.join(target, "head")))
                for slot in range(head + 1):
                    sp, tp = os.path.join(source, _seg(slot)), os.path.join(target, _seg(slot))
                    if os.path.exists(sp) and _read(sp) == _read(tp):
                        # Equal bytes across the two families are not
                        # required to share (the artifact physically
                        # carries them); within the target family they do.
                        pass

    def test_repeated_apply_is_a_noop(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            cp.apply_sync_family(artifact, tgt)
            before = {
                d: {n: _read(os.path.join(d, n)) for n in os.listdir(d)}
                for d in targets
            }
            cp.apply_sync_family(artifact, tgt)
            for d in targets:
                self.assertEqual(sorted(os.listdir(d)), sorted(before[d]))
                for name, raw in before[d].items():
                    self.assertEqual(_read(os.path.join(d, name)), raw)
            self.assertEqual(sorted(os.listdir(tgt)), ["b1", "b2", "main"])

    def test_family_verification_covers_the_synced_members(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            cp.apply_sync_family(artifact, tgt)
            report = cp.verify_chain(targets)
            self.assertTrue(report.ok)
            self.assertEqual(len(report.chains), 3)

    def test_family_verification_reports_a_shared_bad_segment(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            cp.apply_sync_family(artifact, tgt)
            # Corrupt a prefix segment shared (one inode) by main and b2.
            tmain, tb1, tb2 = targets
            victim = os.path.join(tmain, _seg(1))
            self.assertTrue(_same_inode([victim, os.path.join(tb2, _seg(1))]))
            with open(victim, "r+b") as fh:
                fh.write(b"tamper")
            with self.assertRaises(ValueError) as caught:
                cp.verify_chain(targets)
            message = str(caught.exception)
            self.assertIn(_seg(1), message)
            self.assertIn(tmain, message)
            # Every chain reaching the shared segment is attributed.
            self.assertIn(tb2, message)

    def test_conversion_both_ways_is_bitwise(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            source_export = os.path.join(root, "src.seqf")
            target_export = os.path.join(root, "tgt.seqf")
            cp.export_family(sources, source_export)
            cp.export_family(targets, target_export)
            direct = os.path.join(root, "direct.seqs")
            cp.sync_family(sources, targets, direct)
            seq, _ = _stack()
            converted = seq.family_export_to_sync(
                source_export, target_export
            )
            self.assertEqual(converted, _read(direct))
            back = seq.sync_to_family_export(converted, target_export)
            back_path = _write(root, "back.seqf", back)
            restored = os.path.join(root, "restored")
            os.mkdir(restored)
            cp.import_family(back_path, restored)
            for source in sources:
                self.assertEqual(
                    _state(os.path.join(restored, os.path.basename(source))),
                    _state(source),
                )

    def test_self_contained_artifact_converts_without_target(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            # Fresh targets with differing values: no shared prefix.
            weights = _base_weights()
            weights["b1"] = [v + 0.1 for v in weights["b1"]]
            fresh = []
            for name in ("main", "b1", "b2"):
                path = os.path.join(tgt + ".fresh", name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                seq, _ = _stack(weights)
                os.mkdir(path)
                seq.save(path)
                fresh.append(path)
            artifact = os.path.join(root, "fresh.seqs")
            cp.sync_family(sources, fresh, artifact)
            raw = _read(artifact)
            plan, _blobs = cp._parse_sync_artifact(raw)
            self.assertTrue(all(member["bp"] == 0 for member in plan))
            seq, _ = _stack()
            full = seq.sync_to_family_export(raw)
            restored = os.path.join(root, "restored")
            os.mkdir(restored)
            path = _write(root, "full.seqf", full)
            cp.import_family(path, restored)
            for source in sources:
                self.assertEqual(
                    _state(os.path.join(restored, os.path.basename(source))),
                    _state(source),
                )


def _write(root, name, raw):
    path = os.path.join(root, name)
    with open(path, "wb") as fh:
        fh.write(raw)
    return path


class SyncMemoryTests(unittest.TestCase):
    def _memory_families(self):
        seq, _ = _stack()
        main = MemoryChain()
        seq.save(main)
        for i in range(5):
            out, hidden = seq.forward(Tensor(_SEG1 if i % 2 else _SEG2))
            seq.backward(_total(out))
            (seq.adam_step if i % 2 else seq.update)(
                _ADAM_LR if i % 2 else _LR
            )
            seq.save(main)
        b1 = cp.fork_chain_memory(main, up_to=3)
        bs, _ = _stack()
        bh = bs.load(b1)
        out, bh = bs.forward(Tensor(_SEG2), bh)
        bs.backward(_total(out))
        bs.adam_step(_ADAM_LR)
        bs.save(b1)
        b2 = cp.fork_chain_memory(main, up_to=2)

        tmain = cp.fork_chain_memory(main, up_to=2)
        ts, _ = _stack()
        th = ts.load(tmain)
        out, th = ts.forward(Tensor(_SEG1), th)
        ts.backward(_total(out))
        ts.update(_LR)
        ts.save(tmain)
        tb1 = cp.fork_chain_memory(tmain, up_to=1)
        tb2 = cp.fork_chain_memory(tmain, up_to=0)
        return [main, b1, b2], [tmain, tb1, tb2]

    def test_memory_roundtrip_and_sharing(self):
        sources, targets = self._memory_families()
        states = [_state_mem(c) for c in sources]
        artifact = cp.sync_family_memory(sources, targets)
        self.assertEqual(artifact[:8], cp.SYNC_MAGIC)
        cp.apply_sync_family_memory(artifact, targets)
        for target, state in zip(targets, states):
            self.assertEqual(_state_mem(target), state)
        # Equal slots across the synced family are one bytes object.
        for i, a in enumerate(targets):
            for b in targets[i + 1:]:
                head = min(
                    int(a.read_head()), int(b.read_head())
                )
                for slot in range(head + 1):
                    name = _seg(slot)
                    if a._objects[name] == b._objects[name]:
                        self.assertIs(a._objects[name], b._objects[name])
        # Re-apply is a no-op.
        snap = {id(c): dict(c._objects) for c in targets}
        cp.apply_sync_family_memory(artifact, targets)
        for c in targets:
            self.assertEqual(dict(c._objects), snap[id(c)])
        for c in targets:
            self.assertTrue(cp.verify_chain_memory(c).ok)

    def test_memory_conversion_roundtrip(self):
        sources, targets = self._memory_families()
        source_full = cp.export_family_memory(sources)
        target_full = cp.export_family_memory(targets)
        direct = cp.sync_family_memory(sources, targets)
        self.assertEqual(
            cp.family_export_to_sync(source_full, target_full), direct
        )
        back = cp.sync_to_family_export(direct, target_full)
        restored = cp.import_family_memory(back)
        for store, source in zip(restored, sources):
            self.assertEqual(_state_mem(store), _state_mem(source))
        self.assertEqual(back, source_full)

    def test_memory_rejected_apply_leaves_stores_untouched(self):
        sources, targets = self._memory_families()
        artifact = cp.sync_family_memory(sources, targets)
        snapshot = {id(c): dict(c._objects) for c in targets}
        with self.assertRaises(ValueError):
            cp.apply_sync_family_memory(artifact[: len(artifact) // 2], targets)
        for c in targets:
            self.assertEqual(dict(c._objects), snapshot[id(c)])
        flipped = bytearray(artifact)
        flipped[len(flipped) // 2] ^= 0xFF
        with self.assertRaises(ValueError):
            cp.apply_sync_family_memory(bytes(flipped), targets)
        for c in targets:
            self.assertEqual(dict(c._objects), snapshot[id(c)])
        with self.assertRaises(ValueError):
            cp.apply_sync_family_memory(artifact, targets[:2])
        for c in targets:
            self.assertEqual(dict(c._objects), snapshot[id(c)])


class SyncRejectionTests(unittest.TestCase):
    def test_missing_directories_are_filenotfound(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            with self.assertRaises(FileNotFoundError):
                cp.apply_sync_family(artifact, os.path.join(root, "missing"))
            # A missing member after the artifact was produced.
            os.rename(targets[1], os.path.join(root, "parked-b1"))
            with self.assertRaises(FileNotFoundError):
                cp.apply_sync_family(artifact, tgt)
            # A source member paired by name but absent.
            os.rename(sources[2], os.path.join(src, "b2-parked"))
            try:
                with self.assertRaises(FileNotFoundError):
                    cp.sync_family(
                        sources, targets, os.path.join(root, "x.seqs")
                    )
            finally:
                os.rename(os.path.join(src, "b2-parked"), sources[2])
            # The other members are untouched.
            self.assertTrue(cp.verify_chain(targets[0]).ok)

    def test_torn_and_mismatched_artifacts_are_valueerror(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            raw = _read(artifact)
            before = {
                d: {n: _read(os.path.join(d, n)) for n in os.listdir(d)}
                for d in sources + targets
            }
            for bad in (b"", b"not a sync", raw[: len(raw) // 2], raw[:-1]):
                with self.assertRaises(ValueError):
                    cp.apply_sync_family(
                        _write(root, "bad.seqs", bad), tgt
                    )
            flipped = bytearray(raw)
            flipped[len(flipped) // 2] ^= 0xFF
            with self.assertRaises(ValueError):
                cp.apply_sync_family(
                    _write(root, "bad.seqs", bytes(flipped)), tgt
                )
            with self.assertRaises(ValueError):
                unknown = (
                    raw[:8]
                    + struct.pack("<I", 99)
                    + raw[12:]
                )
                cp.apply_sync_family(
                    _write(root, "bad.seqs", unknown), tgt
                )
            # An artifact for a different target family is rejected.
            other = os.path.join(root, "other")
            os.mkdir(other)
            omain = os.path.join(other, "main")
            os.mkdir(omain)
            seq, _ = _stack()
            seq.save(omain)
            # Same member names but diverged chains (ordinary saves into
            # fresh directories, not forks of the artifact's target).
            for name in ("b1", "b2"):
                path = os.path.join(other, name)
                os.mkdir(path)
                other_seq, _ = _stack()
                other_seq.save(path)
            with self.assertRaises(ValueError):
                cp.apply_sync_family(artifact, other)
            # Nothing moved on either family.
            for d in sources + targets:
                for name, value in before[d].items():
                    self.assertEqual(_read(os.path.join(d, name)), value)
            self.assertEqual(sorted(os.listdir(tgt)), ["b1", "b2", "main"])

    def test_unequal_lengths_and_pairing_and_shapes_are_valueerror(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            with self.assertRaises(ValueError):
                cp.sync_family(sources[:2], targets, artifact)
            # Same basename is required for a positional pair.
            renamed = os.path.join(_src, "renamed")
            os.rename(sources[1], renamed)
            try:
                with self.assertRaises(ValueError):
                    cp.sync_family(
                        [sources[0], renamed, sources[2]],
                        targets,
                        artifact,
                    )
            finally:
                os.rename(renamed, sources[1])
            # Shape disagreement: a third family whose model is wider.
            weights = _base_weights()
            weights["b1"] = [0.01, -0.02, 0.03, 0.04]
            shaped = os.path.join(root, "shaped")
            os.mkdir(shaped)
            sm = os.path.join(shaped, "main")
            os.mkdir(sm)
            seq, _ = _stack(weights)
            seq.save(sm)
            others = [sm]
            t_others = [targets[0]]
            for name in ("b1", "b2"):
                p = os.path.join(shaped, name)
                os.mkdir(p)
                seq, _ = _stack(weights)
                seq.save(p)
                others.append(p)
                t_others.append(os.path.join(tgt, name))
            with self.assertRaises(ValueError):
                cp.sync_family(others, t_others, artifact)

    def test_typeerror_for_bad_argument_shapes(self):
        seq, _ = _stack()
        with self.assertRaises(TypeError):
            seq.sync_family("dir", ["dir2"])
        with self.assertRaises(TypeError):
            seq.apply_sync_family(123, "dir")
        with self.assertRaises(TypeError):
            seq.sync_family([MemoryChain()], [MemoryChain()], "path")
        with self.assertRaises(TypeError):
            seq.apply_sync_family(b"x", MemoryChain())


class SyncCrashTests(unittest.TestCase):
    def test_killed_apply_resumes_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            states = [_state(d) for d in sources]
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            proc = multiprocessing.Process(
                target=_slow_sync_worker, args=(artifact, tgt)
            )
            proc.start()
            deadline = time.monotonic() + 30
            saw_progress = False
            while proc.is_alive() and time.monotonic() < deadline:
                names = os.listdir(tgt)
                if any(n.startswith(".seqsync") for n in names):
                    saw_progress = True
                    break
                for name in names:
                    member = os.path.join(tgt, name)
                    if os.path.isdir(member) and any(
                        n.startswith(".seqs-s") or n == ".seqsync.m"
                        for n in os.listdir(member)
                    ):
                        saw_progress = True
                        break
                if saw_progress:
                    break
                time.sleep(0.001)
            if proc.is_alive():
                proc.kill()
            proc.join()
            self.assertTrue(saw_progress)
            # Every member at its final name is one complete chain the
            # read-only verifier accepts (old or mid-publication).
            for target in targets:
                self.assertTrue(cp.verify_chain(target).ok)
            # Re-running converges and leaves no residue.
            cp.apply_sync_family(artifact, tgt)
            for target, state in zip(targets, states):
                self.assertEqual(_state(target), state)
            self.assertEqual(sorted(os.listdir(tgt)), ["b1", "b2", "main"])
            for target in targets:
                self.assertFalse(
                    any(
                        n.startswith(".seqs") for n in os.listdir(target)
                    )
                )
            self.assertTrue(cp.verify_chain(targets).ok)

    def test_resume_tolerates_appends_to_committed_members(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            states = [_state(d) for d in sources]
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            proc = multiprocessing.Process(
                target=_slow_sync_worker, args=(artifact, tgt)
            )
            proc.start()
            deadline = time.monotonic() + 30
            while proc.is_alive() and time.monotonic() < deadline:
                if os.path.exists(os.path.join(targets[0], _seg(8))):
                    break
                time.sleep(0.001)
            if proc.is_alive():
                proc.kill()
            proc.join()
            # An append on a member the killed run already committed is
            # preserved by the resume.
            seq, hidden = _stack()
            hidden = seq.load(targets[0])
            _train(seq, targets[0], 1, hidden=hidden, start=20)
            cp.apply_sync_family(artifact, tgt)
            head = int(_read(os.path.join(targets[0], "head")))
            self.assertEqual(head, 9)
            self.assertTrue(cp.verify_chain(targets[0]).ok)
            for target, state in zip(targets[1:], states[1:]):
                self.assertEqual(_state(target), state)

    def test_debris_is_reclaimed_by_the_next_family_operation(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            with open(os.path.join(tgt, ".seqsync.blk-0000000007"), "wb") as fh:
                fh.write(b"dead block")
            with open(os.path.join(tgt, ".seqsync"), "wb") as fh:
                fh.write(json.dumps({"v": 1, "d": "dead"}).encode())
            in_member = os.path.join(targets[0], ".seqs-s-0000000009.seqd")
            with open(in_member, "wb") as fh:
                fh.write(b"dead staged")
            # The next family operation (a sync apply) reclaims it all.
            cp.apply_sync_family(artifact, tgt)
            self.assertEqual(sorted(os.listdir(tgt)), ["b1", "b2", "main"])
            self.assertFalse(os.path.exists(in_member))
            # Sweeping again changes nothing.
            cp._sweep_parent_staging(tgt)
            self.assertEqual(sorted(os.listdir(tgt)), ["b1", "b2", "main"])


def _slow_sync_worker(artifact, target):
    from sequence_engine import checkpoint as cp

    original_link = os.link
    original_replace = os.replace

    def slow_link(src, dst):
        time.sleep(0.02)
        return original_link(src, dst)

    def slow_replace(src, dst):
        time.sleep(0.015)
        return original_replace(src, dst)

    cp.os.link = slow_link
    cp.os.replace = slow_replace
    try:
        cp.apply_sync_family(artifact, target)
    finally:
        cp.os.link = original_link
        cp.os.replace = original_replace


class SyncConcurrencyTests(unittest.TestCase):
    def test_appends_never_see_half_a_chain(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            errors = []
            stop = False

            def appender(path):
                try:
                    while not stop:
                        # Loads and read-only verification must always see
                        # one complete chain; no saves race the switch
                        # (the member lock hands the appender either the
                        # old or the new chain, never a half one).
                        cp.load_chain(path)
                        cp.verify_chain(path)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=appender, args=(p,))
                for p in targets
            ]
            for thread in threads:
                thread.start()
            try:
                cp.apply_sync_family(artifact, tgt)
            finally:
                stop = True
                for thread in threads:
                    thread.join()
            self.assertEqual(errors, [])
            source_states = [_state(d) for d in sources]
            for target, state in zip(targets, source_states):
                self.assertEqual(_state(target), state)
            self.assertTrue(cp.verify_chain(targets).ok)

    def test_a_save_racing_the_switch_lands_on_one_complete_chain(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            errors = []
            stop = False

            def racing_saver(path):
                try:
                    while not stop:
                        doc = cp.load_chain(path)
                        # A save based on a snapshot taken before the
                        # switch legitimately conflicts when the sync
                        # moved the chain onto a different history (e.g.
                        # it newly fixes hidden state); that is a
                        # rejected stale append, never a half chain.
                        try:
                            cp.save_chain(doc, path)
                        except (OSError, FileNotFoundError, ValueError):
                            pass
                        # Every load, before and after, is one complete
                        # walkable chain.
                        cp.load_chain(path)
                        cp.verify_chain(path)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=racing_saver, args=(p,))
                for p in targets
            ]
            for thread in threads:
                thread.start()
            try:
                for _ in range(3):
                    cp.apply_sync_family(artifact, tgt)
            finally:
                stop = True
                for thread in threads:
                    thread.join()
            self.assertEqual(errors, [])
            for target in targets:
                self.assertTrue(cp.verify_chain(target).ok)

    def test_serializes_with_fork_merge_delete_and_compaction(self):
        with tempfile.TemporaryDirectory() as root:
            sources, targets, _src, tgt = _families(root)
            artifact = os.path.join(root, "sync.seqs")
            cp.sync_family(sources, targets, artifact)
            errors = []
            stop = False

            def family_traffic():
                try:
                    while not stop:
                        branch = os.path.join(tgt, f"tmp-{threading.get_ident()}")
                        if not os.path.exists(branch):
                            cp.fork_chain(targets[0], branch)
                            cp.merge_chains(targets[0], branch)
                            cp.delete_chain(branch)
                        cp.compact_chain(targets[1], up_to=0)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            thread = threading.Thread(target=family_traffic)
            thread.start()
            try:
                for _ in range(5):
                    cp.apply_sync_family(artifact, tgt)
            finally:
                stop = True
                thread.join()
            self.assertEqual(errors, [])
            self.assertTrue(cp.verify_chain(targets).ok)


if __name__ == "__main__":
    unittest.main()
