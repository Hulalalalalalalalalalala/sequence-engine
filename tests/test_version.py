"""Tests for the version-2 format and version-1 read migration."""

import json
import struct
import zlib

import unittest

from sequence_engine import Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _build,
    _base_weights,
    _fresh_stack,
    _SEG1,
    _total,
)


def _trained_bytes():
    seq, _ = _build(_base_weights())
    out, _ = seq.forward(Tensor(_SEG1))
    seq.backward(_total(out))
    buf = bytearray()
    seq.save(buf)
    return seq, bytes(buf)


def _forge_v1(document):
    header, payload = cp._freeze(document, version=1, source_version=None)
    return cp._frame(header, payload, 1)


def _repack(raw, mutate, version=None):
    """Rewrite a checkpoint with a mutated JSON header and a valid CRC."""
    if version is None:
        version = struct.unpack("<I", raw[8:12])[0]
    (hlen,) = struct.unpack("<Q", raw[12:20])
    header = json.loads(raw[20 : 20 + hlen])
    payload = raw[20 + hlen : raw.rfind(cp.END_MAGIC)]
    mutate(header)
    new_header = json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
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


class FormatVersionTests(unittest.TestCase):
    def test_native_files_are_version_2_with_source_2(self):
        seq, raw = _trained_bytes()
        self.assertEqual(struct.unpack("<I", raw[8:12])[0], 2)
        doc = cp.parse_bytes(raw)
        self.assertEqual(doc["src"], 2)
        header_len = struct.unpack("<Q", raw[12:20])[0]
        header = json.loads(raw[20 : 20 + header_len])
        self.assertEqual(header["v"], 2)
        self.assertEqual(header["src"], 2)

    def test_version1_loads_and_records_source(self):
        _, raw_v2 = _trained_bytes()
        raw_v1 = _forge_v1(cp.parse_bytes(raw_v2))
        self.assertEqual(struct.unpack("<I", raw_v1[8:12])[0], 1)

        restored, _ = _fresh_stack()
        hidden = restored.load(raw_v1)
        self.assertEqual(restored._loaded_src, 1)
        self.assertIsNotNone(hidden)

        trained, _ = _fresh_stack()
        out, expected_hidden = trained.forward(Tensor(_SEG1))
        trained.backward(_total(out))
        self.assertEqual(
            [p.tolist() for p in restored.parameters()],
            [p.tolist() for p in trained.parameters()],
        )
        self.assertEqual(
            [s.tolist() for s in hidden], [s.tolist() for s in expected_hidden]
        )

    def test_resaving_migrated_checkpoint_writes_version_2(self):
        _, raw_v2 = _trained_bytes()
        raw_v1 = _forge_v1(cp.parse_bytes(raw_v2))
        restored, _ = _fresh_stack()
        restored.load(raw_v1)
        buf = bytearray()
        restored.save(buf)
        self.assertEqual(struct.unpack("<I", bytes(buf)[8:12])[0], 2)
        self.assertEqual(cp.parse_bytes(bytes(buf))["src"], 2)

    def test_migration_failure_rejects_whole_file(self):
        _, raw_v2 = _trained_bytes()
        raw_v1 = _forge_v1(cp.parse_bytes(raw_v2))

        cases = {
            "parameter/gradient shape disagreement":
                _repack(raw_v1, lambda h: h["params"].__setitem__(0, [9])),
            "v1 missing pending":
                _repack(raw_v1, lambda h: h.pop("pending")),
            "v1 with an added field":
                _repack(raw_v1, lambda h: h.update(unexpected=1)),
            "v2 missing src":
                _repack(raw_v2, lambda h: h.pop("src"), version=2),
            "v2 unknown field":
                _repack(raw_v2, lambda h: h.update(unexpected=1), version=2),
        }
        for message, raw in cases.items():
            with self.subTest(message):
                with self.assertRaises(ValueError):
                    cp.parse_bytes(raw)
                victim, _ = _fresh_stack()
                before = [p.tolist() for p in victim.parameters()]
                with self.assertRaises(ValueError):
                    victim.load(raw)
                self.assertEqual(
                    [p.tolist() for p in victim.parameters()], before
                )

    def test_unknown_version_rejected(self):
        _, raw = _trained_bytes()
        forged = raw[:8] + struct.pack("<I", 999) + raw[12:]
        with self.assertRaises(ValueError):
            cp.parse_bytes(forged)

    def test_non_finite_v1_leaf_rejected_item_by_item(self):
        _, raw_v2 = _trained_bytes()
        raw_v1 = bytearray(_forge_v1(cp.parse_bytes(raw_v2)))
        hlen = struct.unpack("<Q", bytes(raw_v1[12:20]))[0]
        start, end = 20 + hlen, raw_v1.rfind(cp.END_MAGIC)
        planted = False
        for pos in range(start, end, 9):
            if raw_v1[pos] == ord("f"):
                raw_v1[pos + 1 : pos + 9] = struct.pack("<d", float("inf"))
                planted = True
                break
        self.assertTrue(planted)
        raw_v1[-4:] = struct.pack("<I", zlib.crc32(bytes(raw_v1[20:end])))
        with self.assertRaises(ValueError):
            cp.parse_bytes(bytes(raw_v1))

    def test_leaf_count_disagrees_with_header_rejected(self):
        # Claim a leaf count larger than the header's shapes while keeping
        # a matching CRC over the actual payload: framing passes, shape
        # cross-check must reject with ValueError, not an index error.
        _, raw = _trained_bytes()
        (hlen,) = struct.unpack("<Q", raw[12:20])
        payload = raw[20 + hlen : raw.rfind(cp.END_MAGIC)]
        header = json.loads(raw[20 : 20 + hlen])
        new_header = json.dumps(
            header, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        body = (
            raw[:8]
            + struct.pack("<I", 2)
            + struct.pack("<Q", len(new_header))
            + new_header
            + payload
        )
        crc = zlib.crc32(body[20:])
        forged = body + cp.END_MAGIC + struct.pack(
            "<QI", len(payload) // 9 + 1, crc
        )
        with self.assertRaises(ValueError):
            cp.parse_bytes(forged)

    def test_hidden_slot_count_checked_on_fresh_model(self):
        _, raw = _trained_bytes()
        doc = cp.parse_bytes(raw)
        # The wire format cannot cross-check hidden slots against the
        # layer count, so this document parses; the model must reject it
        # on the spot even though it has never run a forward.
        doc["hidden"].append(doc["hidden"][0])
        forged = cp.build_bytes(doc)
        victim, _ = _fresh_stack()
        before = [p.tolist() for p in victim.parameters()]
        with self.assertRaises(ValueError):
            victim.load(forged)
        # Whole-document rejection: nothing applied.
        self.assertEqual(
            [p.tolist() for p in victim.parameters()], before
        )
        self.assertTrue(all(p.grad is None for p in victim.parameters()))

    def test_valid_different_batch_hidden_loads_on_fresh_model(self):
        # A different but well-formed batch size is NOT a mismatch: a
        # fresh model accepts it and the next forward must be shaped by it.
        seq, _ = _fresh_stack()
        big_batch = Tensor([[0.5, -0.25], [0.125, 0.75], [-0.4, 0.3]])
        out, hidden = seq.forward(big_batch)
        seq.backward(_total(out))
        buf = bytearray()
        seq.save(buf)

        victim, _ = _fresh_stack()
        restored = victim.load(bytes(buf))
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].shape[0], 3)

    def test_malformed_hidden_slot_rejected_even_without_forward(self):
        # A brand-new model validates hidden slots on the spot during
        # load(), including entries missing their values.
        victim, _ = _fresh_stack()
        doc = cp.parse_bytes(_trained_bytes()[1])
        doc["hidden"][0] = {"s": [2, 3]}  # entry present, values missing
        with self.assertRaises(ValueError):
            victim._validate_against_model(doc)


if __name__ == "__main__":
    unittest.main()
