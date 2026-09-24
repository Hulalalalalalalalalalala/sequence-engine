"""Tests for bounded-memory recompute, Adam steps and v3 optimizer state."""

from __future__ import annotations

import copy
import json
import math
import os
import struct
import tempfile
import threading
import unittest
import zlib

from sequence_engine import Sequential, Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _RNNStep,
    _base_weights,
    _build,
    _N_H1,
    _N_H2,
    _N_IN,
    _SEG1,
    _SEG2,
    _SEG3,
    _total,
)

_LR = 0.1
_ADAM_LR = 0.01


def _stack(weights=None):
    return _build(weights if weights is not None else _base_weights())


def _zeros_shape(shape):
    if not shape:
        return 0.0
    return [_zeros_shape(shape[1:]) for _ in range(shape[0])]


def _flatten(tree):
    if isinstance(tree, list):
        out = []
        for item in tree:
            out.extend(_flatten(item))
        return out
    return [tree]


def _adam_reference(p, g, m, v, t, lr):
    if isinstance(p, list):
        ps, ms, vs = [], [], []
        for a, b, c, d in zip(p, g, m, v):
            np_, nm, nv = _adam_reference(a, b, c, d, t, lr)
            ps.append(np_)
            ms.append(nm)
            vs.append(nv)
        return ps, ms, vs
    m_new = 0.9 * m + 0.1 * g
    v_new = 0.999 * v + 0.001 * (g * g)
    m_hat = m_new / (1.0 - 0.9**t)
    v_hat = v_new / (1.0 - 0.999**t)
    return p - lr * m_hat / (math.sqrt(v_hat) + 1e-8), m_new, v_new


class RecomputeParityTests(unittest.TestCase):
    def test_outputs_and_gradients_bitwise_equal_across_segments(self):
        weights = _base_weights()
        plain, _ = _stack(weights)
        replay, replay_layers = _stack(weights)
        replay.set_recompute(True)
        carried_p = carried_r = None
        for seg in (_SEG1, _SEG2, _SEG3):
            op, hp = plain.forward(Tensor(seg), carried_p)
            oor, hr = replay.forward(Tensor(seg), carried_r)
            self.assertEqual(op.tolist(), oor.tolist())
            self.assertEqual(
                [s.tolist() for s in hp], [s.tolist() for s in hr]
            )
            # Activations are released at the forward boundary.
            self.assertTrue(all(layer._cache is None for layer in replay_layers))
            plain.backward((op, _total(op)))
            replay.backward((oor, _total(oor)))
            self.assertEqual(
                [p.grad.tolist() for p in plain.parameters()],
                [p.grad.tolist() for p in replay.parameters()],
            )
            carried_p = [Tensor(s.tolist()) for s in hp]
            carried_r = [Tensor(s.tolist()) for s in hr]

    def test_layer_without_release_hook_still_matches(self):
        # A layer that does not know how to drop its cache is simply left
        # alone; parity must not depend on the hook existing.
        class CachingScale(_RNNStep):
            pass

        weights = _base_weights()
        layers_a = [
            CachingScale(_N_IN, _N_H1, weights["wxh1"], weights["whh1"], weights["b1"]),
            CachingScale(_N_H1, _N_H2, weights["wxh2"], weights["whh2"], weights["b2"]),
        ]
        layers_b = [
            CachingScale(_N_IN, _N_H1, weights["wxh1"], weights["whh1"], weights["b1"]),
            CachingScale(_N_H1, _N_H2, weights["wxh2"], weights["whh2"], weights["b2"]),
        ]
        plain = Sequential(layers_a)
        replay = Sequential(layers_b)
        replay.set_recompute(True)
        op, _ = plain.forward(Tensor(_SEG1))
        oor, _ = replay.forward(Tensor(_SEG1))
        plain.backward(_total(op))
        replay.backward(_total(oor))
        self.assertEqual(
            [p.grad.tolist() for p in plain.parameters()],
            [p.grad.tolist() for p in replay.parameters()],
        )

    def test_default_switch_is_off_and_requires_bool(self):
        seq, _ = _stack()
        self.assertFalse(seq._recompute)
        with self.assertRaises(ValueError):
            seq.set_recompute(1)


class RecomputeSwitchAndStalenessTests(unittest.TestCase):
    def test_toggle_mid_slice_is_runtime_error_in_both_directions(self):
        seq, _ = _stack()
        seq.set_recompute(True)
        out, _ = seq.forward(Tensor(_SEG1))
        with self.assertRaises(RuntimeError):
            seq.set_recompute(False)
        self.assertTrue(seq._recompute)  # unchanged
        seq.backward(_total(out))  # slice still usable
        seq.set_recompute(False)  # boundary: allowed

        seq2, _ = _stack()
        seq2.forward(Tensor(_SEG1))
        with self.assertRaises(RuntimeError):
            seq2.set_recompute(True)
        self.assertFalse(seq2._recompute)
        seq2.backward(1.0)

    def test_mutated_parameters_refuse_backward_without_half_state(self):
        weights = _base_weights()
        stale, layers = _stack(weights)
        stale.set_recompute(True)
        stale.forward(Tensor(_SEG1))
        original = layers[0].wxh.tolist()
        layers[0].wxh._set_values([[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]])
        with self.assertRaises(RuntimeError):
            stale.backward(1.0)
        self.assertTrue(stale._pending_backward)
        self.assertTrue(all(p.grad is None for p in stale.parameters()))
        # Restore the exact parameters: the same backward now runs.
        layers[0].wxh._set_values(original)
        stale.backward(1.0)
        ref, _ = _stack(weights)
        ref.forward(Tensor(_SEG1))
        ref.backward(1.0)
        self.assertEqual(
            [p.grad.tolist() for p in stale.parameters()],
            [p.grad.tolist() for p in ref.parameters()],
        )

    def test_failed_recompute_backward_rolls_back_and_retries(self):
        class BoomOnce(_RNNStep):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._boom = True

            def backward(self, upstream):
                if self._boom:
                    self._boom = False
                    raise RuntimeError("transient")
                return super().backward(upstream)

        weights = _base_weights()
        boom = BoomOnce(_N_IN, _N_H1, weights["wxh1"], weights["whh1"], weights["b1"])
        quiet = _RNNStep(_N_H1, _N_H2, weights["wxh2"], weights["whh2"], weights["b2"])
        broken = Sequential([boom, quiet])
        broken.set_recompute(True)
        broken.forward(Tensor(_SEG1))
        with self.assertRaises(RuntimeError):
            broken.backward(1.0)
        self.assertIsNone(quiet.wxh.grad)
        self.assertTrue(broken._pending_backward)
        broken.backward(1.0)
        ref, _ = _stack(weights)
        ref.forward(Tensor(_SEG1))
        ref.backward(1.0)
        self.assertEqual(
            [p.grad.tolist() for p in broken.parameters()],
            [p.grad.tolist() for p in ref.parameters()],
        )
        with self.assertRaises(RuntimeError):
            broken.backward(1.0)


class AdamStepTests(unittest.TestCase):
    def test_step_matches_fixed_coefficient_reference(self):
        seq, _ = _stack()
        seq.zero_grad()
        lr = _ADAM_LR
        ref_p = [p.tolist() for p in seq.parameters()]
        zero = [p.grad.tolist() for p in seq.parameters()]
        ref_m = copy.deepcopy(zero)
        ref_v = copy.deepcopy(zero)
        for t, seg in enumerate((_SEG1, _SEG2, _SEG3), start=1):
            out, _ = seq.forward(Tensor(seg))
            seq.backward(_total(out))
            grads = [p.grad.tolist() for p in seq.parameters()]
            np_, nm, nv = [], [], []
            for p, g, m, v in zip(ref_p, grads, ref_m, ref_v):
                a, b, c = _adam_reference(p, g, m, v, t, lr)
                np_.append(a)
                nm.append(b)
                nv.append(c)
            ref_p, ref_m, ref_v = np_, nm, nv
            seq.adam_step(lr)
            self.assertEqual(seq._adam_t, t)
            self.assertEqual([p.tolist() for p in seq.parameters()], ref_p)
            self.assertEqual([m.tolist() for m in seq._adam_m], ref_m)
            self.assertEqual([v.tolist() for v in seq._adam_v], ref_v)

    def test_lr_validation(self):
        seq, _ = _stack()
        seq.adam_step(_ADAM_LR)  # no gradients: still a valid step
        for bad in ("0.1", True, False, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                seq.adam_step(bad)

    def test_no_gradients_leaves_parameters_in_place_but_advances_t(self):
        seq, _ = _stack()
        before = [p.tolist() for p in seq.parameters()]
        seq.adam_step(_ADAM_LR)
        self.assertEqual([p.tolist() for p in seq.parameters()], before)
        self.assertEqual(seq._adam_t, 1)
        # Moments decayed toward zero from zero and remain zero.
        self.assertTrue(
            all(
                leaf == 0
                for m in seq._adam_m
                for leaf in _flatten(m.tolist())
            )
        )

    def test_gradients_are_not_cleared(self):
        seq, _ = _stack()
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        grads = [p.grad.tolist() for p in seq.parameters()]
        seq.adam_step(_ADAM_LR)
        self.assertEqual([p.grad.tolist() for p in seq.parameters()], grads)

    def test_plain_update_path_is_independent(self):
        seq, _ = _stack()
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        before = [p.tolist() for p in seq.parameters()]
        grads = [p.grad.tolist() for p in seq.parameters()]
        seq.update(_LR)
        # update never touches Adam state.
        self.assertEqual(seq._adam_t, 0)
        self.assertIsNone(seq._adam_m)
        self.assertIsNone(seq._adam_v)

        def sub(values, grads):
            if isinstance(values, list):
                return [sub(a, b) for a, b in zip(values, grads)]
            return values - _LR * grads

        self.assertEqual(
            [p.tolist() for p in seq.parameters()],
            [sub(a, b) for a, b in zip(before, grads)],
        )

    def test_zero_grad_does_not_reset_moments(self):
        seq, _ = _stack()
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        seq.adam_step(_ADAM_LR)
        moments = [m.tolist() for m in seq._adam_m]
        seq.zero_grad()
        self.assertEqual(seq._adam_t, 1)
        self.assertEqual([m.tolist() for m in seq._adam_m], moments)


def _trajectory(seq, schedule, lr, hidden=None):
    for seg, use_adam in schedule:
        out, hidden = seq.forward(Tensor(seg), hidden)
        seq.backward(_total(out))
        if use_adam:
            seq.adam_step(lr)
        else:
            seq.update(_LR)
        hidden = [Tensor(s.tolist()) for s in hidden]
    return (
        [p.tolist() for p in seq.parameters()],
        [p.grad.tolist() for p in seq.parameters()],
        seq._adam_t,
        [m.tolist() for m in seq._adam_m],
        [v.tolist() for v in seq._adam_v],
        [s.tolist() for s in hidden],
    )


class AdamCheckpointTests(unittest.TestCase):
    SCHEDULE = ((_SEG1, True), (_SEG2, False), (_SEG3, True), (_SEG1, True))

    def test_resume_matches_uninterrupted_bitwise_memory_and_path(self):
        lr = _ADAM_LR
        uninterrupted, _ = _stack()
        expected = _trajectory(uninterrupted, self.SCHEDULE, lr)

        for save_target in ("memory", "path"):
            runner, _ = _stack()
            prefix = _trajectory(runner, self.SCHEDULE[:2], lr)
            if save_target == "memory":
                buf = bytearray()
                runner.save(buf)
                source = bytes(buf)
            else:
                td = tempfile.TemporaryDirectory()
                path = os.path.join(td.name, "state.ckp")
                runner.save(path)
                source = path
            resumed, _ = _stack()
            restored = resumed.load(source)
            self.assertEqual(resumed.loaded_from_version, 3)
            self.assertEqual([s.tolist() for s in restored], prefix[5])
            got = _trajectory(resumed, self.SCHEDULE[2:], lr, restored)
            self.assertEqual(got, expected)
            if save_target == "path":
                td.cleanup()

    def test_chain_carries_optimizer_state_bitwise(self):
        seq, _ = _stack()
        chain = cp.MemoryChain()
        seq.save(chain)
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        seq.adam_step(_ADAM_LR)
        seq.save(chain)
        full = bytearray()
        seq.save(full)
        doc = cp.load_chain_memory(chain)
        self.assertEqual(cp.build_bytes(doc), bytes(full))
        self.assertEqual(doc["optim"]["t"], 1)
        target, _ = _stack()
        target.load(chain)
        self.assertEqual(target._adam_t, 1)
        self.assertEqual(
            [m.tolist() for m in target._adam_m],
            [m.tolist() for m in seq._adam_m],
        )
        self.assertEqual(
            [v.tolist() for v in target._adam_v],
            [v.tolist() for v in seq._adam_v],
        )

    def test_oversized_step_count_and_nonfinite_moments_rejected(self):
        seq, _ = _stack()
        seq.adam_step(_ADAM_LR)
        seq._adam_t = 2**63
        with self.assertRaises(ValueError):
            seq.save(bytearray())
        seq._adam_t = 1
        seq._adam_m[0]._set_values(
            [[float("inf"), 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        with self.assertRaises(ValueError):
            seq.save(bytearray())
        # The same non-finite state is refused on the incremental path; a
        # valid basis committed beforehand stays the reachable head.
        seq._adam_m[0]._set_values(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        chain = cp.MemoryChain()
        seq.save(chain)  # valid basis
        seq._adam_m[0]._set_values(
            [[float("inf"), 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        with self.assertRaises(ValueError):
            seq.save(chain)
        doc = cp.load_chain_memory(chain)  # prior head intact and finite
        self.assertEqual(doc["optim"]["t"], 1)

    def test_loaded_zero_state_starts_adam_from_scratch(self):
        # A checkpoint saved at t == 0, loaded, then stepped, must equal a
        # fresh model stepped the same way (state really starts at zero).
        fresh, _ = _stack()
        buf = bytearray()
        fresh.save(buf)
        loaded, _ = _stack()
        loaded.load(bytes(buf))

        def one_step(seq):
            out, _ = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out))
            seq.adam_step(_ADAM_LR)
            return [p.tolist() for p in seq.parameters()]

        self.assertEqual(one_step(loaded), one_step(fresh))


def _synthesize_v2(raw_v3):
    """Build genuine version-2 bytes from native v3 bytes (no optim group).

    Drops the two moment leaf groups from the payload and the ``optim``
    header field, re-framing with version 2. This stands in for a real
    pre-v3 file so v2 migration is exercised against the old wire shape.
    """
    raw = bytes(raw_v3)
    hlen = struct.unpack("<Q", raw[12:20])[0]
    header = json.loads(raw[20 : 20 + hlen])

    def size(shape):
        n = 1
        for d in shape:
            n *= d
        return n

    param_leaves = sum(size(s) for s in header["params"])
    end = raw.rfind(cp.END_MAGIC)
    payload = raw[20 + hlen : end]
    # v3 order: params, grads, m, v, hidden -> v2 keeps params, grads, hidden.
    groups = [param_leaves, param_leaves, param_leaves, param_leaves]
    groups.append(sum(size(s) for s in (header["hidden"] or [])))
    offsets = []
    pos = 0
    for n in groups:
        offsets.append((pos, pos + n * 9))
        pos += n * 9
    v2_payload = b"".join(
        (payload[offsets[0][0] : offsets[0][1]],
         payload[offsets[1][0] : offsets[1][1]],
         payload[offsets[4][0] : offsets[4][1]])
    )
    v2_header = {
        "v": 2,
        "params": header["params"],
        "grads": header["grads"],
        "hidden": header["hidden"],
        "layers": header["layers"],
        "pending": header["pending"],
    }
    new_header = json.dumps(
        v2_header, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    body = (
        raw[:8]
        + struct.pack("<I", 2)
        + struct.pack("<Q", len(new_header))
        + new_header
        + v2_payload
    )
    crc = zlib.crc32(body[20:])
    leaf_count = len(v2_payload) // 9
    return body + cp.END_MAGIC + struct.pack("<QI", leaf_count, crc)


class Version2MigrationTests(unittest.TestCase):
    def test_genuine_v2_wire_shape_migrates_item_by_item(self):
        seq, _ = _stack()
        out, h1 = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        raw = bytearray()
        seq.save(raw)
        v2 = _synthesize_v2(raw)
        self.assertEqual(struct.unpack("<I", v2[8:12])[0], 2)

        migrated, _ = _stack()
        restored = migrated.load(bytes(v2))
        self.assertEqual(migrated.loaded_from_version, 2)
        self.assertEqual(migrated._adam_t, 0)
        self.assertTrue(
            all(leaf == 0 for m in migrated._adam_m for leaf in _flatten(m.tolist()))
        )
        self.assertTrue(
            all(leaf == 0 for v in migrated._adam_v for leaf in _flatten(v.tolist()))
        )
        self.assertEqual(
            [p.tolist() for p in migrated.parameters()],
            [p.tolist() for p in seq.parameters()],
        )
        self.assertEqual(
            [s.tolist() for s in restored], [s.tolist() for s in h1]
        )
        # Re-save is native v3.
        rebuf = bytearray()
        migrated.save(rebuf)
        self.assertEqual(struct.unpack("<I", bytes(rebuf)[8:12])[0], 3)

    def test_v2_then_adam_matches_fresh_start(self):
        seq, _ = _stack()
        raw = bytearray()
        seq.save(raw)
        v2 = _synthesize_v2(raw)
        migrated, _ = _stack()
        migrated.load(bytes(v2))
        fresh, _ = _stack()

        def step(seq):
            o, _ = seq.forward(Tensor(_SEG1))
            seq.backward(_total(o))
            seq.adam_step(_ADAM_LR)
            return [p.tolist() for p in seq.parameters()], seq._adam_t

        self.assertEqual(step(migrated), step(fresh))


class OptimizerStateRejectionTests(unittest.TestCase):
    def _repack(self, raw, mutate_header, payload_transform=None):
        raw = bytes(raw)
        hlen = struct.unpack("<Q", raw[12:20])[0]
        header = json.loads(raw[20 : 20 + hlen])
        end = raw.rfind(cp.END_MAGIC)
        payload = raw[20 + hlen : end]
        mutate_header(header)
        if payload_transform is not None:
            payload = payload_transform(header, payload)
        new_header = json.dumps(
            header, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        body = (
            raw[:8]
            + struct.pack("<I", 3)
            + struct.pack("<Q", len(new_header))
            + new_header
            + payload
        )
        crc = zlib.crc32(body[20:])
        leaf_count = len(payload) // 9
        return body + cp.END_MAGIC + struct.pack("<QI", leaf_count, crc)

    def test_missing_optim_field_rejected(self):
        seq, _ = _stack()
        seq.zero_grad()
        buf = bytearray()
        seq.save(buf)
        forged = self._repack(bytes(buf), lambda h: h.pop("optim"))
        victim, _ = _stack()
        with self.assertRaises(ValueError):
            victim.load(forged)

    def test_t_zero_with_nonzero_moments_rejected(self):
        seq, _ = _stack()
        seq.zero_grad()
        buf = bytearray()
        seq.save(buf)

        # Flip the first first-moment leaf (it immediately follows the
        # params+grads groups) to 1.0 while keeping t == 0.
        def transform(header, payload):
            pcount = 0
            for s in header["params"]:
                n = 1
                for d in s:
                    n *= d
                pcount += n
            offset = (2 * pcount) * 9 + 1
            payload = bytearray(payload)
            payload[offset : offset + 8] = struct.pack("<d", 1.0)
            return bytes(payload)

        forged = self._repack(bytes(buf), lambda h: None, transform)
        victim, _ = _stack()
        with self.assertRaises(ValueError):
            victim.load(forged)

    def test_moment_shape_mismatch_rejected(self):
        seq, _ = _stack()
        seq.zero_grad()
        buf = bytearray()
        seq.save(buf)

        def mangle(header):
            header["optim"]["m"][0] = [9, 9]

        forged = self._repack(bytes(buf), mangle)
        victim, _ = _stack()
        with self.assertRaises(ValueError):
            victim.load(forged)


class AdamConcurrencyTests(unittest.TestCase):
    def test_concurrent_adam_update_save_load_always_coherent(self):
        seq, _ = _stack()
        errors = []

        def train(use_adam):
            try:
                for _ in range(100):
                    out, _ = seq.forward(Tensor(_SEG1))
                    seq.backward(_total(out))
                    if use_adam:
                        seq.adam_step(_ADAM_LR)
                    else:
                        seq.update(_LR)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def saver():
            try:
                for _ in range(100):
                    buf = bytearray()
                    seq.save(buf)
                    probe, _ = _stack()
                    probe.load(bytes(buf))
                    self.assertGreaterEqual(probe._adam_t, 0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=train, args=(True,)),
            threading.Thread(target=train, args=(False,)),
            threading.Thread(target=saver),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
