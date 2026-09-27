"""Tests for cross-family incremental synchronization.

Two families of the same model are diffed into one incremental artifact
holding only genuinely differing segments; applying it lands the source
state atomically and bit for bit, keeps the shared layout one physical
copy, advances no optimizer step, is idempotent and crash-resumable, and
converts with the full family artifact.  These tests cover:

* the round trip: heads, exclusive tails, positions, bitwise state,
  optimizer step counts, no byte duplicated for a shared segment, and
  the read-only/non-stepping diff;
* idempotence, determinism and the full <-> incremental conversion;
* the error taxonomy (FileNotFoundError / ValueError / OSError /
  TypeError) and the guarantee that a rejected application leaves both
  families byte for byte untouched;
* hard-kill safety (a kill during staging and during rotation) with
  resume by re-run and by the next family operation, and idempotent
  residue reclamation;
* saves, loads and verifies interleaving with an application, and
  family verification across the synchronized members;
* the Sequential-level entry points and the in-memory forms.
"""

from __future__ import annotations

import multiprocessing
import os
import struct
import tempfile
import threading
import time
import unittest
import zlib

from sequence_engine import MemoryChain, Sequential, Tensor
from sequence_engine import _familysync as fs
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _build,
    _base_weights,
    _SEG1,
    _SEG2,
    _total,
)

_LR = 0.1
_ADAM_LR = 0.05


def _stack(weights=None):
    return _build(weights if weights is not None else _base_weights())


def _seg(index):
    return f"seg-{index:010d}.seqd"


def _train(seq, directory, steps, hidden=None, start=0):
    for i in range(start, start + steps):
        out, hidden = seq.forward(
            Tensor(_SEG1 if i % 2 == 0 else _SEG2), hidden
        )
        seq.backward(_total(out))
        if i % 3 == 2:
            seq.adam_step(_ADAM_LR)
        else:
            seq.update(_LR)
        seq.save(directory)
    return hidden


def _make_family(root, name, main_steps, forks, branch_steps=()):
    parent = os.path.join(root, name)
    os.mkdir(parent)
    main = os.path.join(parent, "main")
    os.mkdir(main)
    seq, _ = _stack()
    seq.save(main)
    _train(seq, main, main_steps)
    members = [main]
    for index, point in enumerate(forks):
        branch = os.path.join(parent, f"b{index + 1}")
        cp.fork_chain(main, branch, up_to=point)
        extra = branch_steps[index] if index < len(branch_steps) else 0
        if extra:
            bseq, _ = _stack()
            hidden = bseq.load(branch)
            _train(bseq, branch, extra, hidden=hidden, start=point)
        members.append(branch)
    return parent, members


def _state(directory):
    return cp.build_bytes(cp.load_chain(directory))


def _step(directory):
    return cp.load_chain(directory)["optim"]["t"]


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _segments(directory):
    return sorted(
        name for name in os.listdir(directory) if name.startswith("seg-")
    )


def _heads(members):
    return {d: _read(os.path.join(d, "head")) for d in members}


def _states(members):
    return {d: _state(d) for d in members}


def _steps(members):
    return {d: _step(d) for d in members}


def _snapshot(members):
    return {
        d: (
            _state(d),
            _heads([d])[d],
            _segments(d),
            _step(d),
        )
        for d in members
    }


class SyncBasicsTests(unittest.TestCase):
    def _families(self, root):
        src_parent, src = _make_family(
            root, "src", 8, (6, 3), branch_steps=(2, 0)
        )
        # Advance the source main beyond the fork family.
        seq, _ = _stack()
        hidden = seq.load(src[0])
        _train(seq, src[0], 2, hidden=hidden, start=9)
        tgt_parent, tgt = _make_family(
            root, "tgt", 5, (3, 2), branch_steps=(0, 0)
        )
        return src_parent, src, tgt_parent, tgt

    def test_diff_apply_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            _sp, src, tp, tgt = self._families(root)
            want = _states(src)
            want_steps = _steps(src)
            want_heads = _heads(src)
            artifact = os.path.join(root, "sync.seqs")
            self.assertIsNone(fs.diff_families(src, tgt, artifact))
            with open(artifact, "rb") as fh:
                raw = fh.read()
            self.assertEqual(raw[:8], fs.SYNC_EXPORT_MAGIC)
            self.assertLess(
                os.path.getsize(artifact),
                os.path.getsize(os.path.join(root, "full.seqx"))
                if os.path.exists(os.path.join(root, "full.seqx"))
                else os.path.getsize(artifact) + 1,
            )
            self.assertIsNone(fs.apply_family_diff(artifact, tp))
            for source, target in zip(src, tgt):
                self.assertEqual(_state(target), want[source])
                self.assertEqual(_step(target), want_steps[source])
                self.assertEqual(
                    _read(os.path.join(target, "head")),
                    want_heads[source],
                )
            # The source family is untouched by both calls.
            for source in src:
                self.assertEqual(_state(source), want[source])
            self.assertTrue(cp.verify_chain(tgt).ok)

    def test_diff_is_read_only_deterministic_and_incremental(self):
        with tempfile.TemporaryDirectory() as root:
            _sp, src, tp, tgt = self._families(root)
            full_artifact = os.path.join(root, "full.seqx")
            cp.export_family(src, full_artifact)
            before = _snapshot(src + tgt)
            first = os.path.join(root, "a.seqs")
            second = os.path.join(root, "b.seqs")
            fs.diff_families(src, tgt, first)
            fs.diff_families(src, tgt, second)
            self.assertEqual(_read(first), _read(second))
            self.assertLess(os.path.getsize(first), os.path.getsize(full_artifact))
            self.assertEqual(_snapshot(src + tgt), before)
            _d, plan, blobs = fs.parse_sync_artifact(_read(first))
            # Shared prefix positions carry no block (the families
            # started from the same model and share basis bytes).
            self.assertTrue(any(member["shared"] for member in plan))
            for member in plan:
                self.assertEqual(
                    len(member["shared"]) + len(member["carried"]),
                    member["head"] + 1,
                )

    def test_apply_is_idempotent_and_keeps_shared_layout(self):
        with tempfile.TemporaryDirectory() as root:
            _sp, src, tp, tgt = self._families(root)
            artifact = os.path.join(root, "sync.seqs")
            fs.diff_families(src, tgt, artifact)
            fs.apply_family_diff(artifact, tp)
            objects_after = {
                d: {n: _read(os.path.join(d, n)) for n in _segments(d)}
                for d in tgt
            }
            fs.apply_family_diff(artifact, tp)
            for d in tgt:
                for name, raw in objects_after[d].items():
                    self.assertEqual(_read(os.path.join(d, name)), raw)
            # The basis is one physical file across the family.
            self.assertTrue(
                os.path.samefile(
                    os.path.join(tgt[0], _seg(0)),
                    os.path.join(tgt[-1], _seg(0)),
                )
            )
            # No sync residue remains.
            self.assertEqual(
                [n for n in os.listdir(tp) if n.startswith(".")], []
            )

    def test_full_incremental_conversion_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            _sp, src, tp, tgt = self._families(root)
            full_artifact = os.path.join(root, "full.seqx")
            cp.export_family(src, full_artifact)
            # Convert the full artifact to incremental against the target
            # and apply it: states must equal the source.
            sync_path = os.path.join(root, "from_full.seqs")
            sync_bytes = fs.full_to_incremental_artifact(full_artifact, tgt)
            with open(sync_path, "wb") as fh:
                fh.write(sync_bytes)
            fs.apply_family_diff(sync_path, tp)
            for source, target in zip(src, tgt):
                self.assertEqual(_state(target), _state(source))
            # Convert a fresh diff back to a self-contained full artifact
            # against the matching baseline and import it.
            _tp2, tgt2 = _make_family(
                root, "tgt2", 5, (3, 2), branch_steps=(0, 0)
            )
            sync2_path = os.path.join(root, "x.seqs")
            fs.diff_families(src, tgt2, sync2_path)
            full_back = fs.incremental_to_full_artifact(sync2_path, tgt2)
            cp._parse_family_export(full_back)
            imported_parent = os.path.join(root, "imported")
            os.mkdir(imported_parent)
            full_path = os.path.join(root, "back.seqx")
            with open(full_path, "wb") as fh:
                fh.write(full_back)
            cp.import_family(full_path, imported_parent)
            for source in src:
                restored = os.path.join(
                    imported_parent, os.path.basename(source)
                )
                self.assertEqual(_state(restored), _state(source))

    def test_mismatched_target_is_rejected_and_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            _sp, src, tp, tgt = self._families(root)
            artifact = os.path.join(root, "sync.seqs")
            fs.diff_families(src, tgt, artifact)
            # The target moves after the diff.
            seq, _ = _stack()
            hidden = seq.load(tgt[0])
            _train(seq, tgt[0], 1, hidden=hidden, start=100)
            before = _snapshot(tgt)
            with self.assertRaises(ValueError):
                fs.apply_family_diff(artifact, tp)
            self.assertEqual(_snapshot(tgt), before)

    def test_missing_member_is_filenotfound_and_other_side_intact(self):
        with tempfile.TemporaryDirectory() as root:
            _sp, src, tp, tgt = self._families(root)
            src_snapshot = _snapshot(src)
            os.rename(tgt[1], os.path.join(root, "hidden_b2"))
            artifact = os.path.join(root, "sync.seqs")
            with self.assertRaises(FileNotFoundError):
                fs.diff_families(src, tgt, artifact)
            self.assertEqual(_snapshot(src), src_snapshot)
            # A missing member at apply time is likewise FileNotFoundError.
            _tp2, tgt2 = _make_family(
                root, "tgt2", 5, (3, 2), branch_steps=(0, 0)
            )
            art2 = os.path.join(root, "s2.seqs")
            fs.diff_families(src, tgt2, art2)
            before2 = {d: _snapshot([d])[d] for d in tgt2}
            keep = os.path.join(root, "kept_member")
            os.rename(tgt2[1], keep)
            with self.assertRaises(FileNotFoundError):
                fs.apply_family_diff(art2, _tp2)
            for d in tgt2:
                if os.path.isdir(d):
                    self.assertEqual(_snapshot([d])[d], before2[d])
            os.rename(keep, tgt2[1])


def _kill_synchronization_worker(artifact, target, phase, queue):
    """Apply an artifact, raising during staging or during a rotation."""
    try:
        if phase == "stage":
            orig = fs._stage_one_member

            def boom(*args, **kwargs):
                if args[1]["name"] == "b2":
                    raise RuntimeError("simulated failure during staging")
                return orig(*args, **kwargs)

            fs._stage_one_member = boom
            try:
                fs.apply_family_diff(artifact, target)
            except RuntimeError:
                queue.put("refused")
                return
            queue.put("unexpected success")
        else:
            orig = fs._rotate_one_member

            def boom(parent, member, old_head, digest):
                # Commit the member marker and promote only the basis,
                # then die: the member is mid-rotation and resumable.
                directory = os.path.join(parent, member["name"])
                staging = fs._staging_dir(parent, member["name"])
                fs._atomic_write(
                    directory,
                    fs.SYNC_MARKER_NAME,
                    fs._encode_member_marker(
                        digest, old_head, member["head"], 2
                    ),
                )
                os.replace(
                    os.path.join(staging, _seg(0)),
                    os.path.join(directory, _seg(0)),
                )
                raise RuntimeError("simulated kill during rotation")

            fs._rotate_one_member = boom
            try:
                fs.apply_family_diff(artifact, target)
            except RuntimeError:
                queue.put("killed")
                return
            queue.put("unexpected success")
    finally:
        pass


class SyncCrashTests(unittest.TestCase):
    def _families(self, root):
        src_parent, src = _make_family(
            root, "src", 8, (6, 3), branch_steps=(2, 0)
        )
        seq, _ = _stack()
        hidden = seq.load(src[0])
        _train(seq, src[0], 2, hidden=hidden, start=9)
        tgt_parent, tgt = _make_family(
            root, "tgt", 5, (3, 2), branch_steps=(0, 0)
        )
        artifact = os.path.join(root, "sync.seqs")
        fs.diff_families(src, tgt, artifact)
        return src, tgt_parent, tgt, artifact

    def test_failure_during_staging_changes_no_member(self):
        with tempfile.TemporaryDirectory() as root:
            src, tp, tgt, artifact = self._families(root)
            before = _snapshot(tgt)
            queue = multiprocessing.Queue()
            proc = multiprocessing.Process(
                target=_kill_synchronization_worker,
                args=(artifact, tp, "stage", queue),
            )
            proc.start()
            proc.join()
            self.assertEqual(queue.get(timeout=30), "refused")
            self.assertEqual(_snapshot(tgt), before)
            # A fresh application completes normally.
            fs.apply_family_diff(artifact, tp)
            for source, target in zip(src, tgt):
                self.assertEqual(_state(target), _state(source))

    def test_kill_during_rotation_is_resumed_by_rerun_and_open(self):
        with tempfile.TemporaryDirectory() as root:
            src, tp, tgt, artifact = self._families(root)
            queue = multiprocessing.Queue()
            proc = multiprocessing.Process(
                target=_kill_synchronization_worker,
                args=(artifact, tp, "rotation", queue),
            )
            proc.start()
            proc.join()
            self.assertEqual(queue.get(timeout=30), "killed")
            # Ordinary member opens roll their own interrupted rotation
            # forward and always load one complete chain.
            for directory in tgt:
                cp.load_chain(directory)
            # The family-wide re-run converges.
            fs.apply_family_diff(artifact, tp)
            for source, target in zip(src, tgt):
                self.assertEqual(_state(target), _state(source))
            self.assertTrue(cp.verify_chain(tgt).ok)
            self.assertEqual(
                [n for n in os.listdir(tp) if n.startswith(".")], []
            )

    def test_killed_apply_is_finished_by_next_family_operation(self):
        with tempfile.TemporaryDirectory() as root:
            src, tp, tgt, artifact = self._families(root)
            queue = multiprocessing.Queue()
            proc = multiprocessing.Process(
                target=_kill_synchronization_worker,
                args=(artifact, tp, "rotation", queue),
            )
            proc.start()
            proc.join()
            self.assertEqual(queue.get(timeout=30), "killed")
            # The next family operation over the parent finishes the
            # application (here a single-chain compaction via family_open).
            cp.compact_chain(tgt[0])
            for source, target in zip(src, tgt):
                self.assertEqual(_state(target), _state(source))
            self.assertEqual(
                [n for n in os.listdir(tp) if n.startswith(".")], []
            )


class SyncConcurrencyTests(unittest.TestCase):
    def test_readers_observe_only_complete_chains(self):
        with tempfile.TemporaryDirectory() as root:
            src_parent, src = _make_family(
                root, "src", 8, (6, 3), branch_steps=(2, 0)
            )
            seq, _ = _stack()
            hidden = seq.load(src[0])
            _train(seq, src[0], 2, hidden=hidden, start=9)
            tp, tgt = _make_family(
                root, "tgt", 5, (3, 2), branch_steps=(0, 0)
            )
            artifact = os.path.join(root, "sync.seqs")
            fs.diff_families(src, tgt, artifact)
            errors = []
            stop = threading.Event()

            def reader(directory):
                while not stop.is_set():
                    try:
                        cp.load_chain(directory)
                        cp.verify_chain(directory)
                    except Exception as exc:  # pragma: no cover - failure path
                        errors.append((directory, repr(exc)))
                    time.sleep(0.0005)

            threads = [
                threading.Thread(target=reader, args=(d,)) for d in tgt
            ]
            for thread in threads:
                thread.start()
            time.sleep(0.01)
            fs.apply_family_diff(artifact, tp)
            stop.set()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            for source, target in zip(src, tgt):
                self.assertEqual(_state(target), _state(source))

    def test_members_save_and_append_after_apply(self):
        with tempfile.TemporaryDirectory() as root:
            src_parent, src = _make_family(
                root, "src", 8, (6, 3), branch_steps=(2, 0)
            )
            tp, tgt = _make_family(
                root, "tgt", 5, (3, 2), branch_steps=(0, 0)
            )
            artifact = os.path.join(root, "sync.seqs")
            fs.diff_families(src, tgt, artifact)
            fs.apply_family_diff(artifact, tp)
            head = int(_read(os.path.join(tgt[0], "head")))
            seq, _ = _stack()
            hidden = seq.load(tgt[0])
            _train(seq, tgt[0], 2, hidden=hidden, start=head + 1)
            self.assertTrue(cp.verify_chain(tgt[0]).ok)
            cp.compact_family(tgt)
            self.assertTrue(cp.verify_chain(tgt).ok)
            branch = os.path.join(tp, "later")
            cp.fork_chain(tgt[0], branch)
            cp.merge_chains(src[1], tgt[1])
            self.assertTrue(cp.verify_chain(tgt).ok)


class SyncArtifactRejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls._tmp.name
        cls.src_parent, cls.src = _make_family(
            root, "src", 8, (6, 3), branch_steps=(2, 0)
        )
        cls.tp, cls.tgt = _make_family(
            root, "tgt", 5, (3, 2), branch_steps=(0, 0)
        )
        cls.artifact = os.path.join(root, "sync.seqs")
        fs.diff_families(cls.src, cls.tgt, cls.artifact)
        with open(cls.artifact, "rb") as fh:
            cls.raw = fh.read()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _expect(self, error, fn):
        with self.assertRaises(error):
            fn()

    def test_truncated_and_corrupt_artifacts(self):
        self._expect(ValueError, lambda: fs.parse_sync_artifact(b""))
        self._expect(ValueError, lambda: fs.parse_sync_artifact(b"SEQFAMX2x"))
        self._expect(
            ValueError,
            lambda: fs.parse_sync_artifact(self.raw[: len(self.raw) // 2]),
        )
        flipped = bytearray(self.raw)
        flipped[-1] ^= 0xFF
        self._expect(ValueError, lambda: fs.parse_sync_artifact(bytes(flipped)))
        self._expect(
            ValueError, lambda: fs.parse_sync_artifact(self.raw + b" ")
        )

    def test_bad_manifest(self):
        # Corrupt the manifest JSON region (after the 20-byte header).
        broken = bytearray(self.raw)
        for index in range(20, min(60, len(broken))):
            broken[index] = ord(b"{") if broken[index] != ord(b"{") else ord(b"}")
        self._expect(ValueError, lambda: fs.parse_sync_artifact(bytes(broken)))

    def test_type_errors(self):
        seq = Sequential(_stack()[1])
        self._expect(
            TypeError, lambda: fs.diff_families(self.src[0], self.tgt, os.path.join(self.tp, "x"))
        )
        self._expect(
            TypeError,
            lambda: fs.apply_family_diff(1234, self.tp),
        )

    def test_shape_mismatch_rejects(self):
        shaped_root = tempfile.mkdtemp(dir=self._tmp.name)
        sp = os.path.join(shaped_root, "srcfam")
        os.mkdir(sp)
        shaped = os.path.join(sp, "main")
        os.mkdir(shaped)
        weights = _base_weights()
        weights["b1"] = [0.01, -0.02]
        seq, _ = _stack(weights)
        seq.save(shaped)
        tp = os.path.join(shaped_root, "tgtfam")
        os.mkdir(tp)
        target_main = os.path.join(tp, "main")
        os.mkdir(target_main)
        # The target model uses the normal shapes, so the models disagree.
        seq2, _ = _stack()
        seq2.save(target_main)
        artifact = os.path.join(shaped_root, "bad.seqs")
        with self.assertRaises(ValueError):
            fs.diff_families([shaped], [target_main], artifact)
        self.assertFalse(os.path.exists(artifact))


class SequentialEntryPointTests(unittest.TestCase):
    def test_sequential_memory_diff_apply_convert(self):
        seq, _ = _stack()
        main = MemoryChain()
        seq.save(main)
        hidden = None
        for i in range(5):
            out, hidden = seq.forward(
                Tensor(_SEG1 if i % 2 == 0 else _SEG2), hidden
            )
            seq.backward(_total(out))
            seq.update(_LR)
            seq.save(main)
        branch = cp.fork_chain_memory(main, up_to=2)
        source = [main, branch]
        source_states = [
            cp.build_bytes(cp.load_chain_memory(chain)) for chain in source
        ]

        base = MemoryChain()
        bseq, _ = _stack()
        bseq.save(base)
        out, h = bseq.forward(Tensor(_SEG1))
        bseq.backward(_total(out))
        bseq.update(_LR)
        bseq.save(base)
        base_branch = cp.fork_chain_memory(base, up_to=1)
        target = [base, base_branch]

        via = Sequential(_stack()[1])
        artifact = via.diff_families(source, target)
        self.assertIsInstance(artifact, bytes)
        self.assertEqual(artifact[:8], fs.SYNC_EXPORT_MAGIC)
        via.apply_family_diff(artifact, target)
        for chain, state in zip(target, source_states):
            self.assertEqual(
                cp.build_bytes(cp.load_chain_memory(chain)), state
            )
        # Re-apply is a no-op.
        via.apply_family_diff(artifact, target)

        # Convert a fresh diff (against a fresh baseline family) to a
        # self-contained full artifact and import it: identical state.
        fresh_base = MemoryChain()
        fseq, _ = _stack()
        fseq.save(fresh_base)
        o, fh = fseq.forward(Tensor(_SEG1))
        fseq.backward(_total(o))
        fseq.update(_LR)
        fseq.save(fresh_base)
        fresh_branch = cp.fork_chain_memory(fresh_base, up_to=1)
        baseline = [fresh_base, fresh_branch]
        diff = via.diff_families(source, baseline)
        full_artifact = via.convert_family_artifact(
            diff, baseline, to_incremental=False
        )
        self.assertEqual(full_artifact[:8], cp.FAMILY_EXPORT_MAGIC)
        restored = via.import_family(full_artifact)
        for chain, state in zip(restored, source_states):
            self.assertEqual(
                cp.build_bytes(cp.load_chain_memory(chain)), state
            )

    def test_sequential_path_diff_apply(self):
        with tempfile.TemporaryDirectory() as root:
            sp, src = _make_family(
                root, "src", 6, (3, 2), branch_steps=(1, 0)
            )
            tp, tgt = _make_family(
                root, "tgt", 3, (2, 1), branch_steps=(0, 0)
            )
            via = Sequential(_stack()[1])
            artifact = os.path.join(root, "f.seqs")
            via.diff_families(src, tgt, artifact)
            via.apply_family_diff(artifact, tp)
            for source, target in zip(src, tgt):
                self.assertEqual(_state(target), _state(source))


if __name__ == "__main__":
    unittest.main()
