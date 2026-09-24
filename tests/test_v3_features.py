"""Tests for the version-3 capabilities:

* ``adam_step`` with fixed coefficients and checkpointed moments,
* bounded-memory recompute mode (``set_recompute``),
* version-3 checkpoints carrying the optimizer state,
* version-2 full-file and incremental-chain migration,
* concurrency around ``adam_step`` / ``save`` / ``load``.
"""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
import threading
import zlib
import unittest

from sequence_engine import Sequential, Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _RNNStep,
    _base_weights,
    _build,
    _SEG1,
    _SEG2,
    _SEG3,
    _total,
)

_B1, _B2, _EPS = 0.9, 0.999, 1e-8
_LR = 0.05


def _stack(weights=None):
    return _build(weights if weights is not None else _base_weights())


def _downgrade_to_v2(raw):
    """Turn native v3 snapshot bytes into a faithful v2 file."""
    hlen = struct.unpack("<Q", raw[12:20])[0]
    header = json.loads(raw[20 : 20 + hlen])
    end = raw.rfind(cp.END_MAGIC)
    payload = raw[20 + hlen : end]
    param_leaves = sum(math.prod(shape) for shape in header["params"])
    payload_v2 = payload[: -(2 * param_leaves * 9)]
    header.pop("optim")
    header["v"] = 2
    new_header = json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    leaf_count = len(payload_v2) // 9
    body = (
        raw[:8]
        + struct.pack("<I", 2)
        + struct.pack("<Q", len(new_header))
        + new_header
        + payload_v2
    )
    crc = zlib.crc32(body[20:])
    return body + cp.END_MAGIC + struct.pack("<QI", leaf_count, crc)


def _reference_adam(theta, grads, m, v, t, lr):
    """Exact scalar reference for one adam step."""
    m_next = _B1 * m + 0.1 * grads
    v_next = _B2 * v + 0.001 * (grads * grads)
    m_hat = m_next / (1.0 - _B1**t)
    v_hat = v_next / (1.0 - _B2**t)
    theta_next = theta - lr * m_hat / (math.sqrt(v_hat) + _EPS)
    return theta_next, m_next, v_next


def _walk(tree, fn):
    if isinstance(tree, list):
        return [_walk(item, fn) for item in tree]
    return fn(tree)


def _flatten(tree):
    if isinstance(tree, list):
        out = []
        for item in tree:
            out.extend(_flatten(item))
        return out
    return [tree]


class AdamStepTests(unittest.TestCase):
    def test_first_step_matches_the_specified_formula(self):
        seq, _ = _stack()
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        before = [p.tolist() for p in seq.parameters()]
        grads = [p.grad.tolist() for p in seq.parameters()]

        def expected(theta, grads):
            if isinstance(theta, list):
                return [expected(a, b) for a, b in zip(theta, grads)]
            t, m, v = _reference_adam(theta, grads, 0.0, 0.0, 1, _LR)
            return t

        seq.adam_step(_LR)
        for param, theta0, grad in zip(seq.parameters(), before, grads):
            self.assertEqual(param.tolist(), expected(theta0, grad))
        self.assertEqual(seq._adam_t, 1)

    def test_moments_follow_the_fixed_updates(self):
        seq, _ = _stack()
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        grads = [p.grad.tolist() for p in seq.parameters()]
        seq.adam_step(_LR)
        for m_tree, g_tree in zip(seq._adam_m, grads):
            for m_value, g_value in zip(_flatten(m_tree), _flatten(g_tree)):
                self.assertEqual(m_value, 0.1 * g_value)
        for v_tree, g_tree in zip(seq._adam_v, grads):
            for v_value, g_value in zip(_flatten(v_tree), _flatten(g_tree)):
                self.assertEqual(v_value, 0.001 * (g_value * g_value))

        # Second step: t=2 and moments compound with the carried values.
        m_before = [_flatten(m) for m in seq._adam_m]
        seq.adam_step(_LR)
        self.assertEqual(seq._adam_t, 2)
        for m_tree, m0_flat, g_tree in zip(seq._adam_m, m_before, grads):
            for m_value, m0, g_value in zip(
                _flatten(m_tree), m0_flat, _flatten(g_tree)
            ):
                self.assertEqual(m_value, 0.9 * m0 + 0.1 * g_value)

    def test_gradients_are_not_cleared(self):
        seq, _ = _stack()
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        grads = [p.grad.tolist() for p in seq.parameters()]
        seq.adam_step(_LR)
        seq.adam_step(_LR)
        self.assertEqual(
            [p.grad.tolist() for p in seq.parameters()], grads
        )

    def test_zero_grad_then_adam_uses_zero_gradients(self):
        # Gradient accumulation/clear semantics are untouched: zero_grad
        # pins zeros, and an adam step on them leaves parameters in place.
        seq, _ = _stack()
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        seq.zero_grad()
        before = [p.tolist() for p in seq.parameters()]
        seq.adam_step(_LR)
        self.assertEqual([p.tolist() for p in seq.parameters()], before)
        self.assertEqual(seq._adam_t, 1)
        self.assertTrue(
            all(
                v == 0.0
                for tree in seq._adam_m
                for v in _flatten(tree)
            )
        )

    def test_parameters_without_gradients_untouched_but_t_advances(self):
        seq, _ = _stack()
        before = [p.tolist() for p in seq.parameters()]
        seq.adam_step(_LR)
        self.assertEqual([p.tolist() for p in seq.parameters()], before)
        self.assertEqual(seq._adam_t, 1)

    def test_non_numeric_or_non_finite_lr_is_value_error(self):
        seq, _ = _stack()
        seq.zero_grad()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                seq.adam_step(bad)
        with self.assertRaises(ValueError):
            seq.adam_step("0.05")
        with self.assertRaises(ValueError):
            seq.adam_step(True)
        self.assertEqual(seq._adam_t, 0)

    def test_plain_update_does_not_touch_adam_state(self):
        seq, _ = _stack()
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        seq.adam_step(_LR)
        t_before = seq._adam_t
        m_before = seq._adam_m
        v_before = seq._adam_v
        params_before = [p.tolist() for p in seq.parameters()]
        seq.update(0.1)
        self.assertEqual(seq._adam_t, t_before)
        self.assertIs(seq._adam_m, m_before)
        self.assertIs(seq._adam_v, v_before)
        self.assertNotEqual(
            [p.tolist() for p in seq.parameters()], params_before
        )

    def test_coefficients_are_fixed(self):
        import inspect

        signature = inspect.signature(Sequential.adam_step)
        self.assertEqual(tuple(signature.parameters), ("self", "learning_rate"))


class AdamCheckpointTests(unittest.TestCase):
    def _trained(self, steps=2):
        seq, _ = _stack()
        out1, h1 = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out1))
        out2, h2 = seq.forward(Tensor(_SEG2), h1)
        seq.backward(_total(out2))
        for _ in range(steps):
            seq.adam_step(_LR)
        return seq

    def test_optimizer_state_round_trips_full_snapshot(self):
        seq = self._trained(steps=3)
        state = (
            seq._adam_t,
            [_walk(m, lambda x: x) for m in seq._adam_m],
            [_walk(v, lambda x: x) for v in seq._adam_v],
            [p.tolist() for p in seq.parameters()],
        )
        buf = bytearray()
        seq.save(buf)
        self.assertEqual(struct.unpack("<I", bytes(buf)[8:12])[0], 3)
        restored, _ = _stack()
        restored.load(bytes(buf))
        self.assertEqual(restored.loaded_from_version, 3)
        self.assertEqual(restored._adam_t, state[0])
        self.assertEqual(restored._adam_m, state[1])
        self.assertEqual(restored._adam_v, state[2])
        self.assertEqual(
            [p.tolist() for p in restored.parameters()], state[3]
        )

    def test_resume_is_bitwise_identical_to_uninterrupted_run(self):
        seg3 = _SEG3

        def run(checkpoint_target=None):
            seq, _ = _stack()
            out1, hidden = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out1))
            if checkpoint_target is not None:
                seq.save(checkpoint_target)
                seq, _ = _stack()
                hidden = seq.load(checkpoint_target)
            out2, hidden2 = seq.forward(Tensor(_SEG2), hidden)
            seq.backward(_total(out2))
            seq.adam_step(_LR)
            if checkpoint_target is not None:
                seq.save(checkpoint_target)
                seq, _ = _stack()
                hidden2 = seq.load(checkpoint_target)
            carry = [Tensor(slot.tolist()) for slot in hidden2]
            out3, _ = seq.forward(Tensor(seg3), carry)
            seq.backward(_total(out3))
            seq.adam_step(_LR)
            return (
                [p.tolist() for p in seq.parameters()],
                [p.grad.tolist() for p in seq.parameters()],
                seq._adam_t,
                [m for m in seq._adam_m],
            )

        expected = run()
        self.assertEqual(run(bytearray()), expected)
        self.assertEqual(run(cp.MemoryChain()), expected)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.ckp")
            self.assertEqual(run(path), expected)

    def test_chain_reassembly_matches_full_snapshot(self):
        seq = self._trained(steps=2)
        full = bytearray()
        seq.save(full)
        chain = cp.MemoryChain()
        seq.save(chain)
        seq.adam_step(_LR)
        seq.adam_step(_LR)
        full2 = bytearray()
        seq.save(full2)
        seq.save(chain)
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(chain)), bytes(full2)
        )
        victim, _ = _stack()
        victim.load(chain)
        self.assertEqual(victim._adam_t, seq._adam_t)
        self.assertEqual(victim._adam_m, seq._adam_m)
        self.assertEqual(victim._adam_v, seq._adam_v)

    def test_unstepped_model_saves_zero_moments_and_t_zero(self):
        seq, _ = _stack()
        buf = bytearray()
        seq.save(buf)
        doc = cp.parse_bytes(bytes(buf))
        self.assertEqual(doc["optim"]["t"], 0)
        self.assertTrue(
            all(
                v == 0.0
                for entry in doc["optim"]["m"]
                for v in _flatten(entry["v"])
            )
        )
        restored, _ = _stack()
        restored.load(bytes(buf))
        self.assertEqual(restored._adam_t, 0)

    def test_oversized_step_count_or_nonfinite_moment_rejected_on_save(self):
        seq, _ = _stack()
        seq.zero_grad()
        params = seq.parameters()

        def doc(t, m_value=0.0, v_value=0.0):
            return {
                "params": [{"s": p.shape, "v": p.tolist()} for p in params],
                "grads": [
                    {"s": p.shape, "v": p.grad.tolist()} for p in params
                ],
                "hidden": None,
                "optim": {
                    "t": t,
                    "m": [
                        {
                            "s": p.shape,
                            "v": _walk(p.grad.tolist(), lambda x: m_value),
                        }
                        for p in params
                    ],
                    "v": [
                        {
                            "s": p.shape,
                            "v": _walk(p.grad.tolist(), lambda x: v_value),
                        }
                        for p in params
                    ],
                },
                "layers": [
                    {"kind": "_RNNStep",
                     "shapes": [[2, 3], [3, 3], [3]]},
                    {"kind": "_RNNStep",
                     "shapes": [[3, 2], [2, 2], [2]]},
                ],
                "pending": False,
            }

        with self.assertRaises(ValueError):
            cp.build_bytes(doc(2**63))
        with self.assertRaises(ValueError):
            cp.build_bytes(doc(10**40))
        with self.assertRaises(ValueError):
            cp.build_bytes(doc(1, m_value=float("inf")))
        with self.assertRaises(ValueError):
            cp.build_bytes(doc(1, v_value=float("nan")))

    def test_nonfinite_model_state_rejected_on_save(self):
        seq, _ = _stack()
        seq.zero_grad()
        seq.adam_step(0.0)  # populate zero moments
        seq._adam_m[0][0][0] = float("inf")
        with self.assertRaises(ValueError):
            seq.save(bytearray())

    def _repack(self, raw, mutate, version=3):
        hlen = struct.unpack("<Q", raw[12:20])[0]
        header = json.loads(raw[20 : 20 + hlen])
        mutate(header)
        new_header = json.dumps(
            header, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        end = raw.rfind(cp.END_MAGIC)
        payload = raw[20 + hlen : end]
        body = (
            raw[:8]
            + struct.pack("<I", version)
            + struct.pack("<Q", len(new_header))
            + new_header
            + payload
        )
        crc = zlib.crc32(body[20:])
        leaf_count = len(payload) // 9
        return body + cp.END_MAGIC + struct.pack("<QI", leaf_count, crc)

    def test_checkpoint_with_bad_optim_state_rejected_wholesale(self):
        seq = self._trained(steps=1)
        raw = bytearray()
        seq.save(raw)
        raw = bytes(raw)

        victim, _ = _stack()
        with self.assertRaises(ValueError):
            victim.load(self._repack(raw, lambda h: h.pop("optim")))
        self.assertEqual(victim._adam_t, 0)
        self.assertTrue(all(p.grad is None for p in victim.parameters()))

        victim2, _ = _stack()
        with self.assertRaises(ValueError):
            victim2.load(
                self._repack(raw, lambda h: h["optim"].__setitem__("t", -1))
            )
        self.assertEqual(victim2._adam_t, 0)

        victim3, _ = _stack()
        with self.assertRaises(ValueError):
            victim3.load(
                self._repack(
                    raw, lambda h: h["optim"]["m"].__setitem__(0, [9, 9])
                )
            )
        self.assertEqual(victim3._adam_t, 0)

    def test_v2_full_file_migrates_to_zero_optim_state_and_resaves_v3(self):
        seq, _ = _stack()
        out, h1 = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        raw = bytearray()
        seq.save(raw)
        v2 = _downgrade_to_v2(bytes(raw))
        self.assertEqual(struct.unpack("<I", v2[8:12])[0], 2)

        migrated, _ = _stack()
        restored = migrated.load(v2)
        self.assertEqual(migrated.loaded_from_version, 2)
        self.assertEqual(migrated._adam_t, 0)
        self.assertTrue(
            all(
                v == 0.0
                for m_tree in migrated._adam_m
                for v in _flatten(m_tree)
            )
        )
        self.assertEqual(
            [s.tolist() for s in restored], [s.tolist() for s in h1]
        )
        rebuf = bytearray()
        migrated.save(rebuf)
        self.assertEqual(struct.unpack("<I", bytes(rebuf)[8:12])[0], 3)

        # Continuing the migrated state equals a native optimizer that just
        # started from zeros (exactly what migration produced).
        native, _ = _stack()
        native.load(bytes(raw))
        migrated.adam_step(_LR)
        native.adam_step(_LR)
        self.assertEqual(
            [p.tolist() for p in migrated.parameters()],
            [p.tolist() for p in native.parameters()],
        )


class V2ChainMigrationTests(unittest.TestCase):
    def _v2_delta(self, number, hc, entries):
        payload = []
        header_entries = []
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

    def test_genuine_v2_chain_loads_and_reassembles(self):
        seq, _ = _stack()
        basis_raw = bytearray()
        seq.save(basis_raw)
        basis_v2 = _downgrade_to_v2(bytes(basis_raw))
        param_count = len(seq.parameters())

        chain = cp.MemoryChain()
        chain.write_segment(cp._segment_name(0), basis_v2)
        chain.write_segment(cp._segment_name(1), self._v2_delta(1, None, []))

        out, hidden = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        hidden_entries = [
            (2 * param_count + slot, slot_tensor.shape, slot_tensor.tolist())
            for slot, slot_tensor in enumerate(hidden)
        ]
        chain.write_segment(
            cp._segment_name(2), self._v2_delta(2, 2, hidden_entries)
        )
        chain.write_head(b"2")

        doc = cp.load_chain_memory(chain)
        self.assertEqual(doc["optim"]["t"], 0)
        self.assertTrue(
            all(
                v == 0.0
                for entry in doc["optim"]["m"]
                for v in _flatten(entry["v"])
            )
        )
        self.assertEqual(
            [entry["v"] for entry in doc["hidden"]],
            [slot.tolist() for slot in hidden],
        )

        victim, _ = _stack()
        restored = victim.load(chain)
        self.assertEqual(
            [s.tolist() for s in restored], [s.tolist() for s in hidden]
        )
        self.assertEqual(victim._adam_t, 0)

        # Continue the migrated chain with native v3 deltas.
        victim.adam_step(_LR)
        full = bytearray()
        victim.save(full)
        victim.save(chain)
        self.assertEqual(
            cp.build_bytes(cp.load_chain_memory(chain)), bytes(full)
        )

    def test_corrupt_v2_chain_is_rejected(self):
        seq, _ = _stack()
        basis_raw = bytearray()
        seq.save(basis_raw)
        chain = cp.MemoryChain()
        chain.write_segment(
            cp._segment_name(0), _downgrade_to_v2(bytes(basis_raw))
        )
        bad_delta = self._v2_delta(1, 2, [])  # claims hidden, carries none
        chain.write_segment(cp._segment_name(1), bad_delta)
        chain.write_head(b"1")
        with self.assertRaises(ValueError):
            cp.load_chain_memory(chain)


class RecomputeTests(unittest.TestCase):
    def _train_segment(self, seq, segment, hidden=None):
        out, new_hidden = seq.forward(Tensor(segment), hidden)
        seq.backward(_total(out))
        return out, new_hidden

    def test_single_segment_outputs_and_gradients_bitwise_identical(self):
        plain, _ = _stack()
        tuned, _ = _stack()
        tuned.set_recompute(True)
        out_p, h_p = plain.forward(Tensor(_SEG1))
        out_t, h_t = tuned.forward(Tensor(_SEG1))
        self.assertEqual(out_t.tolist(), out_p.tolist())
        self.assertEqual(
            [s.tolist() for s in h_t], [s.tolist() for s in h_p]
        )
        plain.backward(_total(out_p))
        tuned.backward(_total(out_t))
        self.assertEqual(
            [p.grad.tolist() for p in tuned.parameters()],
            [p.grad.tolist() for p in plain.parameters()],
        )

    def test_multi_segment_run_and_adam_trajectory_identical(self):
        plain, _ = _stack()
        tuned, _ = _stack()
        tuned.set_recompute(True)
        hidden_p = hidden_t = None
        for segment in (_SEG1, _SEG2, _SEG3):
            out_p, hidden_p = plain.forward(Tensor(segment), hidden_p)
            out_t, hidden_t = tuned.forward(Tensor(segment), hidden_t)
            self.assertEqual(out_t.tolist(), out_p.tolist())
            plain.backward(_total(out_p))
            tuned.backward(_total(out_t))
            plain.adam_step(_LR)
            tuned.adam_step(_LR)
            self.assertEqual(
                [p.tolist() for p in tuned.parameters()],
                [p.tolist() for p in plain.parameters()],
            )
            self.assertEqual(tuned._adam_t, plain._adam_t)
            # Detached values carry across the boundary in both modes.
            hidden_p = [Tensor(s.tolist()) for s in hidden_p]
            hidden_t = [Tensor(s.tolist()) for s in hidden_t]

    def test_tuple_loss_seed_works_in_recompute_mode(self):
        plain, _ = _stack()
        tuned, _ = _stack()
        tuned.set_recompute(True)
        out_p, _ = plain.forward(Tensor(_SEG1))
        out_t, _ = tuned.forward(Tensor(_SEG1))
        plain.backward((out_p, 1.5))
        tuned.backward((out_t, 1.5))
        self.assertEqual(
            [p.grad.tolist() for p in tuned.parameters()],
            [p.grad.tolist() for p in plain.parameters()],
        )

    def test_backward_contract_still_holds(self):
        tuned, _ = _stack()
        tuned.set_recompute(True)
        with self.assertRaises(RuntimeError):
            tuned.backward(1.0)
        out, _ = tuned.forward(Tensor(_SEG1))
        tuned.backward(_total(out))
        with self.assertRaises(RuntimeError):
            tuned.backward(1.0)

    def test_only_anchors_are_retained(self):
        tuned, layers = _stack()
        tuned.set_recompute(True)
        tuned.forward(Tensor(_SEG1))
        batch_values, hidden_values = tuned._anchors
        self.assertEqual(batch_values, _SEG1)
        self.assertEqual(hidden_values, [None, None])
        # Layers may drop their caches; backward recomputes from anchors.
        for layer in layers:
            layer._cache = None
        out, _ = tuned.forward(Tensor(_SEG1))
        for layer in layers:
            layer._cache = None
        tuned.backward(_total(out))
        ref, _ = _stack()
        ref_out, _ = ref.forward(Tensor(_SEG1))
        ref.backward(_total(ref_out))
        self.assertEqual(
            [p.grad.tolist() for p in tuned.parameters()],
            [p.grad.tolist() for p in ref.parameters()],
        )
        self.assertIsNone(tuned._anchors)

    def test_retained_state_does_not_grow_with_sequence_length(self):
        tuned, _ = _stack()
        tuned.set_recompute(True)
        hidden = None
        sizes = set()
        for segment in (_SEG1, _SEG2, _SEG3, _SEG1, _SEG2, _SEG3):
            out, hidden = tuned.forward(Tensor(segment), hidden)
            # Snapshot the retained anchors' size while the segment is open.
            batch_values, hidden_values = tuned._anchors
            sizes.add(
                (
                    len(batch_values),
                    sum(h is not None for h in hidden_values),
                )
            )
            tuned.backward(_total(out))
            hidden = [Tensor(s.tolist()) for s in hidden]
        # Every open segment keeps exactly one batch (2 rows) and at most
        # one hidden slot per layer, regardless of total sequence length.
        self.assertEqual(sizes, {(2, 0), (2, 2)})

    def test_parameter_rewritten_mid_segment_raises_and_leaves_nothing(self):
        tuned, _ = _stack()
        tuned.set_recompute(True)
        out, _ = tuned.forward(Tensor(_SEG1))
        target = tuned.parameters()[0]
        values = target.tolist()
        values[0][0] += 1.0
        target._set_values(values)
        with self.assertRaises(RuntimeError):
            tuned.backward(_total(out))
        self.assertTrue(all(p.grad is None for p in tuned.parameters()))
        # The rejected pass consumes nothing: the next segment trains.
        recover_out, _ = tuned.forward(Tensor(_SEG1))
        tuned.backward(_total(recover_out))
        self.assertTrue(all(p.grad is not None for p in tuned.parameters()))

    def test_recompute_divergence_detected(self):
        # Even if a parameter is changed and changed back bit-identically
        # the fingerprint passes; instead simulate nondeterministic layers
        # by flipping the recorded output after the anchors are fixed.
        class Flaky(_RNNStep):
            def forward(self, x, hidden):
                result = super().forward(x, hidden)
                return result

        weights = _base_weights()
        seq = Sequential(
            [
                Flaky(2, 3, weights["wxh1"], weights["whh1"], weights["b1"]),
                Flaky(3, 2, weights["wxh2"], weights["whh2"], weights["b2"]),
            ]
        )
        seq.set_recompute(True)
        out, _ = seq.forward(Tensor(_SEG1))
        out._set_values([[0.0, 0.0]])  # tamper with the recorded boundary
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)
        self.assertTrue(all(p.grad is None for p in seq.parameters()))

    def test_switching_mode_between_segments_is_allowed(self):
        seq, _ = _stack()
        seq.set_recompute(True)
        out, h = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        seq.set_recompute(False)  # at a boundary: fine
        out2, _ = seq.forward(Tensor(_SEG2), h)
        seq.backward(_total(out2))
        seq.set_recompute(True)  # at a boundary: fine
        self.assertTrue(seq._recompute)

    def test_switching_mode_mid_segment_raises_runtime_error(self):
        on, _ = _stack()
        on.set_recompute(True)
        on.forward(Tensor(_SEG1))
        with self.assertRaises(RuntimeError):
            on.set_recompute(False)
        # State untouched: the pending backward still completes normally.
        out, _ = on.forward(Tensor(_SEG1))
        on.backward(_total(out))

        off, _ = _stack()
        off.forward(Tensor(_SEG1))
        with self.assertRaises(RuntimeError):
            off.set_recompute(True)
        # Setting it to the same value mid-segment is harmless.
        off.set_recompute(False)

    def test_default_mode_retains_layer_caches(self):
        # Without recompute the engine keeps the existing behavior: layers
        # use whatever they cached at forward time.
        seq, layers = _stack()
        out, _ = seq.forward(Tensor(_SEG1))
        self.assertIsNotNone(layers[0]._cache)
        seq.backward(_total(out))

    def test_saved_boundary_in_recompute_mode_round_trips(self):
        seq, _ = _stack()
        seq.set_recompute(True)
        out, hidden = seq.forward(Tensor(_SEG1))
        buf = bytearray()
        seq.save(buf)  # boundary save before backward
        victim, _ = _stack()
        restored = victim.load(bytes(buf))
        self.assertEqual(
            [s.tolist() for s in restored], [s.tolist() for s in hidden]
        )
        # The pending backward of the saving model is untouched.
        seq.backward(_total(out))

    def test_failed_backward_is_retriable_in_recompute_mode(self):
        class BoomOnce(_RNNStep):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._boom = True

            def backward(self, upstream):
                if self._boom:
                    self._boom = False
                    raise RuntimeError("transient")
                return super().backward(upstream)

        weights = _base_weights()
        boom = BoomOnce(2, 3, weights["wxh1"], weights["whh1"], weights["b1"])
        quiet = _RNNStep(3, 2, weights["wxh2"], weights["whh2"], weights["b2"])
        seq = Sequential([boom, quiet])
        seq.set_recompute(True)
        out, _ = seq.forward(Tensor(_SEG1))
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)
        self.assertIsNone(quiet.wxh.grad)
        seq.backward(1.0)  # retry recomputes activations again
        ref, _ = _stack()
        ref_out, _ = ref.forward(Tensor(_SEG1))
        ref.backward(1.0)
        self.assertEqual(
            [p.grad.tolist() for p in seq.parameters()],
            [p.grad.tolist() for p in ref.parameters()],
        )
        with self.assertRaises(RuntimeError):
            seq.backward(1.0)


class AdamConcurrencyTests(unittest.TestCase):
    def _model(self):
        from tests.test_v2_chains import ConcurrencyTests

        return ConcurrencyTests()._model()

    def test_every_snapshot_is_one_serial_adam_state(self):
        # Build an oracle of serial states 0..N under adam with constant
        # gradients, then race adam_step against save/load: every observed
        # snapshot must be exactly one oracle state (t picks the state).
        seq, a, b = self._model()
        seq.forward(Tensor([[1, 1]]))
        seq.backward(1)
        steps = 120
        oracle_steps = steps * 3  # three stepper threads share one model
        oracle = {}
        probe, _, _ = self._model()
        probe.forward(Tensor([[1, 1]]))
        probe.backward(1)
        probe._ensure_optim_state(len(probe.parameters()))
        for t in range(oracle_steps + 1):
            oracle[t] = (
                [p.tolist() for p in probe.parameters()],
                [_walk(m, lambda x: x) for m in probe._adam_m],
                [_walk(v, lambda x: x) for v in probe._adam_v],
            )
            if t < oracle_steps:
                probe.adam_step(_LR)

        errors = []

        def stepper():
            try:
                for _ in range(steps):
                    seq.adam_step(_LR)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def check(model):
            t = model._adam_t
            self.assertIn(t, oracle)
            theta, m, v = oracle[t]
            self.assertEqual([p.tolist() for p in model.parameters()], theta)
            self.assertEqual([list(x) for x in model._adam_m], m)
            self.assertEqual([list(x) for x in model._adam_v], v)

        def buffer_saver():
            try:
                for _ in range(steps):
                    buf = bytearray()
                    seq.save(buf)
                    model, _, _ = self._model()
                    model.load(bytes(buf))
                    check(model)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        chain = cp.MemoryChain()

        def chain_saver():
            try:
                for _ in range(steps):
                    seq.save(chain)
                    model, _, _ = self._model()
                    model.load(chain)
                    check(model)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        base = bytearray()
        seq.save(base)

        def loader():
            try:
                for _ in range(steps):
                    seq.load(bytes(base))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=stepper) for _ in range(3)
        ] + [
            threading.Thread(target=buffer_saver),
            threading.Thread(target=chain_saver),
            threading.Thread(target=loader),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])

    def test_two_threads_saving_same_path_land_one_full_checkpoint(self):
        model_a, _, _ = self._model()
        model_b, _, _ = self._model()
        x = Tensor([[1, 1]])
        errors = []
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "shared.ckp")
            model_a.save(path)

            def drive(model):
                try:
                    for _ in range(60):
                        model.forward(x)
                        model.backward(1)
                        model.adam_step(_LR)
                        model.save(path)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            t1 = threading.Thread(target=drive, args=(model_a,))
            t2 = threading.Thread(target=drive, args=(model_b,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            probe, _, _ = self._model()
            probe.load(path)  # must be one complete, loadable checkpoint
            a_buf, b_buf = bytearray(), bytearray()
            model_a.save(a_buf)
            model_b.save(b_buf)
            with open(path, "rb") as fh:
                disk = fh.read()
            self.assertIn(disk, (bytes(a_buf), bytes(b_buf)))
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
