"""Tests for chain compaction, crash-safe commits and backward recovery.

Covers:

* ``compact_chain`` / ``MemoryChain.compact`` merging a basis plus deltas
  into a new basis with bitwise-identical reassembled state,
* deterministic segment-count reduction, idempotence and the empty/tail
  edge cases,
* old (version-2) segments participating in compaction and being resaved in
  the current version,
* crash windows: a kill before the head moves keeps the old complete chain;
  a kill during the sweep keeps the new one,
* FileNotFoundError / OSError / ValueError contracts,
* one-shot layer caches being rebuilt so a retried backward is bitwise equal
  to an uninterrupted pass.
"""

from __future__ import annotations

import errno
import json
import os
import struct
import tempfile
import threading
import unittest
from unittest import mock

from sequence_engine import Sequential, Tensor, compact_chain
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _RNNStep,
    _base_weights,
    _build,
    _SEG1,
    _SEG2,
    _SEG3,
    _total,
    _ADAM_LR,
    _LR,
)


def _stack(weights=None):
    return _build(weights if weights is not None else _base_weights())


def _full_bytes(seq):
    buf = bytearray()
    seq.save(buf)
    return bytes(buf)


def _drive_chain(target, steps=("hidden", "update", "hidden", "adam")):
    """Append a mixed history; return the model at the chain head."""
    seq, _ = _stack()
    seq.save(target)  # basis: no hidden, optimizer never stepped
    hidden = None
    for action in steps:
        if action == "hidden":
            segment = _SEG1 if hidden is None else _SEG2
            out, hidden = seq.forward(Tensor(segment), hidden)
            seq.backward(_total(out))
        elif action == "update":
            seq.update(_LR)
        elif action == "adam":
            seq.adam_step(_ADAM_LR)
        seq.save(target)
    return seq


class MemoryCompactionTests(unittest.TestCase):
    def test_full_merge_is_bitwise_identical_and_current_version(self):
        chain = cp.MemoryChain()
        seq = _drive_chain(chain)
        expected = _full_bytes(seq)
        self.assertEqual(cp.build_bytes(cp.load_chain_memory(chain)), expected)

        cp.compact_chain_memory(chain)
        self.assertEqual(len(chain), 1)
        self.assertEqual(cp.build_bytes(cp.load_chain_memory(chain)), expected)
        basis = chain.read_segment(cp._segment_name(0, 1))
        self.assertEqual(struct.unpack("<I", basis[8:12])[0], 3)
        # The head now names generation 1 / segment 0.
        self.assertEqual(cp._decode_head(chain.read_head()), (0, 1))

    def test_compaction_advances_no_optimizer_step(self):
        chain = cp.MemoryChain()
        seq = _drive_chain(chain)
        t_before = cp.load_chain_memory(chain)["optim"]["t"]
        self.assertEqual(t_before, seq._adam_t)
        cp.compact_chain_memory(chain)
        self.assertEqual(cp.load_chain_memory(chain)["optim"]["t"], t_before)

        # The optimizer semantics carry on exactly: the next Adam step of the
        # compacted chain equals that of an uninterrupted model.
        compacted, _ = _stack()
        compacted.load(chain)
        compacted.adam_step(_ADAM_LR)
        seq.adam_step(_ADAM_LR)
        self.assertEqual(
            [p.tolist() for p in compacted.parameters()],
            [p.tolist() for p in seq.parameters()],
        )
        self.assertEqual(compacted._adam_t, seq._adam_t)

    def test_t_zero_chain_keeps_unstepped_optimizer(self):
        chain = cp.MemoryChain()
        seq = _drive_chain(chain, steps=("hidden", "update"))
        self.assertEqual(seq._adam_t, 0)
        cp.compact_chain_memory(chain)
        doc = cp.load_chain_memory(chain)
        self.assertEqual(doc["optim"]["t"], 0)
        model, _ = _stack()
        model.load(chain)
        self.assertEqual(model._adam_t, 0)

    def test_repeated_compaction_is_a_deterministic_noop(self):
        chain = cp.MemoryChain()
        _drive_chain(chain)
        cp.compact_chain_memory(chain)
        snapshot = dict(chain._objects)
        cp.compact_chain_memory(chain)
        cp.compact_chain_memory(chain)
        self.assertEqual(dict(chain._objects), snapshot)

    def test_partial_merge_keeps_a_tail_and_reduces_count(self):
        chain = cp.MemoryChain()
        seq = _drive_chain(chain)  # basis + five deltas = six segments
        expected = _full_bytes(seq)
        cp.compact_chain_memory(chain, keep_tail=3)
        self.assertEqual(len(chain), 3)
        self.assertEqual(cp.build_bytes(cp.load_chain_memory(chain)), expected)
        names = {
            cp._segment_identity(n)
            for n in chain._objects
            if n != "head"
        }
        self.assertEqual(names, {(0, 1), (1, 1), (2, 1)})

    def test_keep_tail_beyond_chain_compacts_nothing(self):
        chain = cp.MemoryChain()
        _drive_chain(chain)
        before = dict(chain._objects)
        cp.compact_chain_memory(chain, keep_tail=1000)
        self.assertEqual(dict(chain._objects), before)

    def test_bad_keep_tail_rejected(self):
        chain = cp.MemoryChain()
        _drive_chain(chain)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                cp.compact_chain_memory(chain, bad)
        with self.assertRaises(ValueError):
            cp.compact_chain_memory(chain, True)
        with self.assertRaises(ValueError):
            cp.compact_chain_memory(chain, 2.0)

    def test_compact_method_and_append_then_full_save_after_compaction(self):
        chain = cp.MemoryChain()
        _drive_chain(chain)
        chain.compact()
        self.assertEqual(len(chain), 1)
        # Append a fresh delta onto the compacted chain.
        model, _ = _stack()
        model.load(chain)
        out, hidden = model.forward(Tensor(_SEG3))
        model.backward(_total(out))
        chain_expected = _full_bytes(model)
        model.save(chain)
        self.assertEqual(cp.build_bytes(cp.load_chain_memory(chain)), chain_expected)
        # A full file snapshot of the continued model matches the chain.
        other, _ = _stack()
        other.load(chain)
        self.assertEqual(_full_bytes(other), chain_expected)

    def test_empty_chain_compaction_is_value_error(self):
        with self.assertRaises(ValueError):
            cp.compact_chain_memory(cp.MemoryChain())


class DirectoryCompactionTests(unittest.TestCase):
    def test_directory_compaction_layout_and_bitwise_equality(self):
        with tempfile.TemporaryDirectory() as td:
            seq = _drive_chain(td)
            expected = _full_bytes(seq)
            before = sorted(os.listdir(td))
            self.assertIn("head", before)
            self.assertEqual(len(before), 6)  # head + basis + four deltas

            compact_chain(td)
            files = sorted(os.listdir(td))
            self.assertEqual(
                files, ["head", "seg-0000000000.g0000000001.seqd"]
            )
            with open(os.path.join(td, "head"), "rb") as fh:
                self.assertEqual(json.loads(fh.read()), {"g": 1, "h": 0})
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), expected)

            # Idempotent.
            compact_chain(td)
            self.assertEqual(
                sorted(os.listdir(td)),
                ["head", "seg-0000000000.g0000000001.seqd"],
            )
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), expected)

            # No lock/control file is left inside the chain directory.
            self.assertFalse(
                any(f.startswith(".") for f in os.listdir(td))
            )

    def test_partial_merge_then_continue_on_disk(self):
        with tempfile.TemporaryDirectory() as td:
            seq = _drive_chain(td)
            expected = _full_bytes(seq)
            compact_chain(td, keep_tail=2)
            self.assertEqual(
                sorted(f for f in os.listdir(td) if f != "head"),
                [
                    "seg-0000000000.g0000000001.seqd",
                    "seg-0000000001.g0000000001.seqd",
                ],
            )
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), expected)

            model, _ = _stack()
            model.load(td)
            out, hidden = model.forward(Tensor(_SEG3))
            model.backward(_total(out))
            model.adam_step(_ADAM_LR)
            continued = _full_bytes(model)
            model.save(td)  # appends onto generation 1
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), continued)

    def test_kill_before_head_moves_leaves_old_complete_chain(self):
        with tempfile.TemporaryDirectory() as td:
            seq = _drive_chain(td)
            expected = _full_bytes(seq)
            original = sorted(os.listdir(td))
            real_atomic = cp._atomic_write

            def fail_at_head(directory, final_name, raw):
                if final_name == cp._HEAD_NAME:
                    raise OSError(errno.EIO, "simulated process kill")
                return real_atomic(directory, final_name, raw)

            with mock.patch.object(cp, "_atomic_write", fail_at_head):
                with self.assertRaises(OSError):
                    compact_chain(td)
            # The rollback removed the staged new segments: same layout.
            self.assertEqual(sorted(os.listdir(td)), original)
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), expected)

            # The directory recovers and compacts normally afterwards.
            compact_chain(td)
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), expected)

    def test_kill_while_writing_segments_rolls_back(self):
        with tempfile.TemporaryDirectory() as td:
            seq = _drive_chain(td)
            expected = _full_bytes(seq)
            original = sorted(os.listdir(td))
            real_atomic = cp._atomic_write
            state = {"calls": 0}

            def fail_on_second_segment(directory, final_name, raw):
                if final_name != cp._HEAD_NAME:
                    state["calls"] += 1
                    if state["calls"] == 2:
                        raise OSError(errno.EIO, "simulated process kill")
                return real_atomic(directory, final_name, raw)

            with mock.patch.object(
                cp, "_atomic_write", fail_on_second_segment
            ):
                with self.assertRaises(OSError):
                    compact_chain(td, keep_tail=3)
            self.assertEqual(sorted(os.listdir(td)), original)
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), expected)

    def test_orphan_new_generation_files_with_old_head_load_old_chain(self):
        # Simulate SIGKILL after the new segments hit disk but before the
        # head advanced: the directory holds BOTH generations while head
        # still names the old one.  Loading must return the old complete
        # chain and ignore the orphan files.
        with tempfile.TemporaryDirectory() as td:
            seq = _drive_chain(td)
            expected = _full_bytes(seq)
            old_files = set(os.listdir(td))

            # Plant a would-be compacted generation-1 basis + an empty tail
            # delta without touching the head.
            doc = cp.load_chain(td)
            with open(
                os.path.join(td, cp._segment_name(0, 1)), "wb"
            ) as fh:
                fh.write(cp.build_bytes(doc))
            # An unrelated torn new-generation file that the crash left
            # behind; the old head never references it.
            with open(
                os.path.join(td, cp._segment_name(1, 1)), "wb"
            ) as fh:
                fh.write(b"partial and torn")
            # Head still says generation 0 / segment 4.
            with open(os.path.join(td, "head"), "rb") as fh:
                self.assertEqual(fh.read(), b"4")

            # Load follows the old generation and never opens the orphans.
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), expected)

            # A subsequent successful compaction supersedes the orphans: it
            # atomically replaces the planted files with valid ones and
            # sweeps every old-generation segment.
            compact_chain(td, keep_tail=2)
            files = set(os.listdir(td))
            self.assertNotIn(cp._segment_name(0), files)
            with open(os.path.join(td, cp._segment_name(1, 1)), "rb") as fh:
                self.assertNotEqual(fh.read(), b"partial and torn")
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), expected)
            self.assertEqual(
                sorted(f for f in os.listdir(td) if f != "head"),
                [
                    "seg-0000000000.g0000000001.seqd",
                    "seg-0000000001.g0000000001.seqd",
                ],
            )
            _ = old_files

    def test_kill_during_sweep_keeps_new_chain_reachable(self):
        with tempfile.TemporaryDirectory() as td:
            seq = _drive_chain(td)
            expected = _full_bytes(seq)
            real_unlink = os.unlink
            removed = {"count": 0}

            def fail_mid_sweep(path, *a, **k):
                if path.endswith(cp._SEG_SUFFIX):
                    removed["count"] += 1
                    if removed["count"] == 2:
                        raise OSError(errno.EIO, "simulated process kill")
                return real_unlink(path, *a, **k)

            with mock.patch.object(os, "unlink", fail_mid_sweep):
                with self.assertRaises(OSError):
                    compact_chain(td)
            # The head already points at the new complete chain; it loads.
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), expected)
            files = sorted(os.listdir(td))
            self.assertIn("seg-0000000000.g0000000001.seqd", files)
            # Appending onto the survived chain still works.
            model, _ = _stack()
            model.load(td)
            model.adam_step(_ADAM_LR)
            continued = _full_bytes(model)
            model.save(td)
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), continued)


class CompactionErrorTests(unittest.TestCase):
    def test_missing_directory_is_filenotfound(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                compact_chain(os.path.join(td, "no-chain"))

    def test_empty_directory_is_value_error(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                compact_chain(td)

    def test_bad_arguments(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(TypeError):
                compact_chain(123)
            with self.assertRaises(TypeError):
                cp.compact_chain_memory(object())

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root")
    def test_unwritable_directory_is_oserror(self):
        with tempfile.TemporaryDirectory() as td:
            # A multi-segment chain forces compaction to write a new basis,
            # so the unwritable directory is hit during the commit.
            _drive_chain(os.path.join(td, "chain"))
            chain_dir = os.path.join(td, "chain")
            os.chmod(chain_dir, 0o555)
            try:
                with self.assertRaises(OSError):
                    compact_chain(chain_dir)
            finally:
                os.chmod(chain_dir, 0o755)

    def test_disk_full_is_oserror_and_leaves_old_chain(self):
        with tempfile.TemporaryDirectory() as td:
            seq = _drive_chain(td)
            expected = _full_bytes(seq)
            original = sorted(os.listdir(td))

            def enospc(fd):
                raise OSError(errno.ENOSPC, "simulated disk full")

            with mock.patch("os.fsync", side_effect=enospc):
                with self.assertRaises(OSError) as ctx:
                    compact_chain(td)
                self.assertEqual(ctx.exception.errno, errno.ENOSPC)
            self.assertEqual(sorted(os.listdir(td)), original)
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), expected)

    def _truncate_segment(self, td, index):
        path = os.path.join(td, cp._segment_name(index))
        with open(path, "rb") as fh:
            good = fh.read()
        with open(path, "wb") as fh:
            fh.write(good[: len(good) // 2])
        return good

    def test_truncated_segment_rejects_whole_compaction(self):
        with tempfile.TemporaryDirectory() as td:
            _drive_chain(td)
            original = sorted(os.listdir(td))
            good = self._truncate_segment(td, 3)
            with self.assertRaises(ValueError):
                compact_chain(td)
            self.assertEqual(sorted(os.listdir(td)), original)
            with open(os.path.join(td, cp._segment_name(3)), "wb") as fh:
                fh.write(good)
            compact_chain(td)  # whole again: compacts fine

    def test_out_of_order_segment_number_rejects_compaction(self):
        with tempfile.TemporaryDirectory() as td:
            _drive_chain(td)
            original = set(os.listdir(td))
            # Rename segment 2 to segment 4 so the walk finds a gap (a
            # referenced segment is absent); the whole compaction is refused.
            os.rename(
                os.path.join(td, cp._segment_name(2)),
                os.path.join(td, cp._segment_name(4)),
            )
            with self.assertRaises((ValueError, FileNotFoundError)):
                compact_chain(td)
            expected = (original - {cp._segment_name(2)}) | {
                cp._segment_name(4)
            }
            self.assertEqual(set(os.listdir(td)), expected)


class V2ChainCompactionTests(unittest.TestCase):
    @staticmethod
    def _v2_delta(number, hc, entries):
        payload, header_entries = [], []
        for index, shape, tree in entries:
            cp._freeze_tree(tree, shape, payload)
            header_entries.append({"i": index, "s": shape})
        header = {
            "v": 2,
            "b": cp._segment_name(0),
            "n": number,
            "hc": hc,
            "changed": header_entries,
            "pending": False,
        }
        return cp._frame(
            cp.DELTA_MAGIC, cp.DELTA_END_MAGIC, header, payload
        )

    def test_genuine_v2_chain_compacts_to_native_basis(self):
        seq, _ = _stack()
        basis_raw = bytearray()
        seq.save(basis_raw)
        from tests.test_v3_features import _downgrade_to_v2

        param_count = len(seq.parameters())
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, cp._segment_name(0)), "wb") as fh:
                fh.write(_downgrade_to_v2(bytes(basis_raw)))
            with open(os.path.join(td, "head"), "wb") as fh:
                fh.write(b"0")
            with open(os.path.join(td, cp._segment_name(1)), "wb") as fh:
                fh.write(self._v2_delta(1, None, []))

            out, hidden = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out))
            entries = [
                (2 * param_count + s, t.shape, t.tolist())
                for s, t in enumerate(hidden)
            ]
            with open(os.path.join(td, cp._segment_name(2)), "wb") as fh:
                fh.write(self._v2_delta(2, 2, entries))
            with open(os.path.join(td, "head"), "wb") as fh:
                fh.write(b"2")

            expected = cp.build_bytes(cp.load_chain(td))
            compact_chain(td)
            self.assertEqual(cp.build_bytes(cp.load_chain(td)), expected)
            with open(
                os.path.join(td, cp._segment_name(0, 1)), "rb"
            ) as fh:
                compacted = fh.read()
            self.assertEqual(
                struct.unpack("<I", compacted[8:12])[0], 3
            )
            doc = cp.load_chain(td)
            self.assertEqual(doc["optim"]["t"], 0)
            self.assertEqual(
                [e["v"] for e in doc["hidden"]],
                [t.tolist() for t in hidden],
            )


class OneShotRNN(_RNNStep):
    """A layer whose backward consumes the forward cache exactly once."""

    def backward(self, upstream):
        if self._cache is None:
            raise RuntimeError("backward cache was already consumed")
        result = super().backward(upstream)
        self._cache = None
        return result


class BoomOnceOneShot(OneShotRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._boom = True

    def backward(self, upstream):
        if self._boom:
            self._boom = False
            raise RuntimeError("transient")
        return super().backward(upstream)


class BackwardCacheRecoveryTests(unittest.TestCase):
    def _layers(self, boom_type):
        weights = _base_weights()
        first = boom_type(2, 3, weights["wxh1"], weights["whh1"], weights["b1"])
        second = OneShotRNN(
            3, 2, weights["wxh2"], weights["whh2"], weights["b2"]
        )
        return first, second, weights

    def test_retry_rebuilds_consumed_caches_and_matches_clean_run(self):
        for recompute in (False, True):
            with self.subTest(recompute=recompute):
                boom, quiet, weights = self._layers(BoomOnceOneShot)
                seq = Sequential([boom, quiet])
                seq.set_recompute(recompute)
                seq.forward(Tensor(_SEG1))
                with self.assertRaises(RuntimeError):
                    seq.backward(1.0)
                # Partial gradients rolled back, caches rebuilt.
                self.assertIsNone(quiet.wxh.grad)
                self.assertIsNotNone(quiet._cache)
                seq.backward(1.0)

                ref = Sequential(
                    [
                        OneShotRNN(
                            2, 3,
                            weights["wxh1"], weights["whh1"], weights["b1"],
                        ),
                        OneShotRNN(
                            3, 2,
                            weights["wxh2"], weights["whh2"], weights["b2"],
                        ),
                    ]
                )
                ref.forward(Tensor(_SEG1))
                ref.backward(1.0)
                self.assertEqual(
                    [p.grad.tolist() for p in seq.parameters()],
                    [p.grad.tolist() for p in ref.parameters()],
                )
                # The successful retry consumed the pass: no free extra call.
                with self.assertRaises(RuntimeError):
                    seq.backward(1.0)

    def test_failure_does_not_change_pending_status_semantics(self):
        # Even with caches consumed and rebuilt, another *forward* is not
        # required -- the retry rides the original forward -- but a clean
        # forward after the failure is also still allowed.
        boom, quiet, _ = self._layers(BoomOnceOneShot)
        seq = Sequential([boom, quiet])
        seq.forward(Tensor(_SEG1))
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)
        # A fresh forward replaces the pending pass; the retry window is now
        # the new forward.
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        for param in seq.parameters():
            self.assertIsNotNone(param.grad)


class CompactConcurrencyTests(unittest.TestCase):
    class _IntLayer:
        checkpoint_kind = "IntLayer"

        def __init__(self, w, g):
            self.w = Tensor(w)
            self._g = g
            self._cache = None

        def parameters(self):
            return [self.w]

        def forward(self, x, hidden):
            xs = x.tolist()
            ws = self.w.tolist()
            y = [
                [xs[r][j] + ws[j] for j in range(len(xs[0]))]
                for r in range(len(xs))
            ]
            self._cache = xs
            return Tensor(y), Tensor(y)

        def backward(self, upstream):
            dy = (
                upstream.tolist()
                if isinstance(upstream, Tensor)
                else [[upstream, upstream] for _ in self._cache]
            )
            self.w.grad = Tensor(list(self._g))
            return Tensor(dy)

    def _model(self):
        return Sequential(
            [
                self._IntLayer([1000, 2000], [3, 7]),
                self._IntLayer([5000, 9000], [1, 2]),
            ]
        )

    def test_saves_and_compactions_interleave_into_coherent_chains(self):
        errors = []
        with tempfile.TemporaryDirectory() as td:
            seq = self._model()
            x = Tensor([[1, 1]])
            seq.forward(x)
            seq.backward(1)
            seq.save(td)

            def saver():
                try:
                    for _ in range(150):
                        seq.forward(x)
                        seq.backward(1)
                        seq.update(1)
                        seq.save(td)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def compactor():
                try:
                    for _ in range(150):
                        try:
                            compact_chain(td, keep_tail=2)
                        except ValueError:
                            pass
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def reader():
                try:
                    for _ in range(150):
                        doc = cp.load_chain(td)
                        values = []
                        for entry in doc["params"]:
                            tree = entry["v"]
                            flat = tree if isinstance(tree[0], float) else [
                                v for row in tree for v in row
                            ]
                            values.extend(flat)
                        base = [1000, 2000, 5000, 9000]
                        grad = [3, 7, 1, 2]
                        ks = set()
                        for j, value in enumerate(values):
                            delta = base[j] - int(value)
                            self.assertEqual(delta % grad[j], 0)
                            ks.add(delta // grad[j])
                        self.assertEqual(len(ks), 1)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=saver),
                threading.Thread(target=compactor),
                threading.Thread(target=reader),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            # The final chain is one complete, loadable state.
            doc = cp.load_chain(td)
            self.assertIsNotNone(doc)


if __name__ == "__main__":
    unittest.main()
