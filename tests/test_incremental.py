"""Tests for incremental checkpoint chains."""

import json
import os
import struct
import tempfile
import zlib

import unittest

from sequence_engine import Sequential, Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _RNNStep,
    _StatelessScale,
    _base_weights,
    _fresh_stack,
    _N_H1,
    _N_IN,
    _SEG1,
    _SEG2,
    _SEG3,
    _total,
)


def _full_bytes(model):
    buf = bytearray()
    model.save(buf)
    return bytes(buf)


class IncrementalChainTests(unittest.TestCase):
    def _walk(self, seq, chain_dir, segments):
        for index, segment in enumerate(segments):
            out, hidden = seq.forward(Tensor(segment))
            seq.backward(_total(out))
            if index % 2 == 1:
                seq.update(0.1)
            seq.save_incremental(chain_dir)
        return hidden

    def test_first_entry_writes_base_delta_manifest(self):
        seq, _ = _fresh_stack()
        with tempfile.TemporaryDirectory() as td:
            chain = os.path.join(td, "chain")
            self.assertEqual(seq.save_incremental(chain), 1)
            self.assertEqual(
                sorted(os.listdir(chain)),
                [".seqckp.lock", "00000001.delta", "base.ckp", "manifest.json"],
            )
            manifest = json.load(
                open(os.path.join(chain, "manifest.json"))
            )
            self.assertEqual(manifest["seq"], 1)
            self.assertEqual(manifest["v"], 2)

    def test_reassembly_is_bitwise_identical_to_full_save(self):
        seq, _ = _fresh_stack()
        with tempfile.TemporaryDirectory() as td:
            chain = os.path.join(td, "chain")
            self._walk(seq, chain, (_SEG1, _SEG2, _SEG3, _SEG1))
            expected = _full_bytes(seq)

            reader, _ = _fresh_stack()
            hidden = reader.load_incremental(chain)
            self.assertEqual(_full_bytes(reader), expected)

            # Also true through the low-level API.
            document = cp.load_chain(chain)
            self.assertEqual(cp.build_bytes(document), expected)
            self.assertIsNotNone(hidden)

    def test_each_intermediate_step_round_trips(self):
        seq, _ = _fresh_stack()
        with tempfile.TemporaryDirectory() as td:
            chain = os.path.join(td, "chain")
            seq.save_incremental(chain)
            for segment in (_SEG1, _SEG2, _SEG3):
                out, _ = seq.forward(Tensor(segment))
                seq.backward(_total(out))
                seq.update(0.05)
                seq.save_incremental(chain)
                reader, _ = _fresh_stack()
                reader.load_incremental(chain)
                self.assertEqual(_full_bytes(reader), _full_bytes(seq))

    def test_unchanged_snapshot_writes_no_layer_bodies(self):
        seq, _ = _fresh_stack()
        with tempfile.TemporaryDirectory() as td:
            chain = os.path.join(td, "chain")
            seq.save_incremental(chain)
            seq.save_incremental(chain)  # identical state
            raw = open(os.path.join(chain, "00000002.delta"), "rb").read()
            header_len = struct.unpack("<Q", raw[13:21])[0]
            header = json.loads(raw[21 : 21 + header_len])
            self.assertEqual(header["changed"], [])
            reader, _ = _fresh_stack()
            reader.load_incremental(chain)
            self.assertEqual(_full_bytes(reader), _full_bytes(seq))

    def test_only_changed_layers_are_written(self):
        seq, layers = _fresh_stack()
        with tempfile.TemporaryDirectory() as td:
            chain = os.path.join(td, "chain")
            seq.save_incremental(chain)
            out, _ = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out))
            seq.save_incremental(chain)
            raw = open(os.path.join(chain, "00000002.delta"), "rb").read()
            header_len = struct.unpack("<Q", raw[13:21])[0]
            header = json.loads(raw[21 : 21 + header_len])
            # Every layer accumulated gradients, so both layers changed.
            self.assertEqual(sorted(header["changed"]), [0, 1])

            # Mutate ONLY layer 0's parameters (same shape) and save:
            # delta 3 must carry exactly one layer body and be smaller
            # than the full base checkpoint.
            values = layers[0].wxh.tolist()
            values[0][0] += 0.25
            layers[0].wxh._set_values(values)
            seq.save_incremental(chain)
            raw3 = open(os.path.join(chain, "00000003.delta"), "rb").read()
            hlen3 = struct.unpack("<Q", raw3[13:21])[0]
            header3 = json.loads(raw3[21 : 21 + hlen3])
            self.assertEqual(header3["changed"], [0])
            self.assertLess(
                os.path.getsize(os.path.join(chain, "00000003.delta")),
                os.path.getsize(os.path.join(chain, "base.ckp")),
            )
            reader, _ = _fresh_stack()
            reader.load_incremental(chain)
            self.assertEqual(_full_bytes(reader), _full_bytes(seq))

    def test_empty_parameter_layer_is_deterministic(self):
        weights = _base_weights()

        def build():
            return Sequential(
                [
                    _RNNStep(
                        _N_IN, _N_H1,
                        weights["wxh1"], weights["whh1"], weights["b1"],
                    ),
                    _StatelessScale(),
                ]
            )

        seq = build()
        with tempfile.TemporaryDirectory() as td:
            chain = os.path.join(td, "chain")
            seq.save_incremental(chain)
            out, _ = seq.forward(Tensor(_SEG1))
            seq.backward(_total(out))
            seq.update(0.1)
            seq.save_incremental(chain)

            reader = build()
            reader.load_incremental(chain)
            self.assertEqual(_full_bytes(reader), _full_bytes(seq))

            # A second identical save keeps deterministic behavior.
            seq.save_incremental(chain)
            reader2 = build()
            reader2.load_incremental(chain)
            self.assertEqual(_full_bytes(reader2), _full_bytes(seq))

    def test_missing_chain_is_file_not_found(self):
        reader, _ = _fresh_stack()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                reader.load_incremental(os.path.join(td, "no-chain"))

    def test_non_directory_target_is_oserror_or_valueerror_deterministically(self):
        # Saving a chain under a path blocked by an ordinary file raises OSError.
        seq, _ = _fresh_stack()
        with tempfile.TemporaryDirectory() as td:
            blocker = os.path.join(td, "file")
            with open(blocker, "wb") as fh:
                fh.write(b"x")
            with self.assertRaises(OSError):
                seq.save_incremental(os.path.join(blocker, "chain"))


class IncrementalCorruptionTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.td = self._td.name
        self.chain = os.path.join(self.td, "chain")
        seq, _ = _fresh_stack()
        seq.save_incremental(self.chain)
        for segment in (_SEG1, _SEG2, _SEG3):
            out, _ = seq.forward(Tensor(segment))
            seq.backward(_total(out))
            seq.update(0.05)
            seq.save_incremental(self.chain)
        self.tip = _full_bytes(seq)

    def tearDown(self):
        self._td.cleanup()

    def _assert_rejected(self):
        reader, _ = _fresh_stack()
        with self.assertRaises(ValueError):
            reader.load_incremental(self.chain)

    def test_truncated_delta_rejected(self):
        path = os.path.join(self.chain, "00000002.delta")
        raw = open(path, "rb").read()
        with open(path, "wb") as fh:
            fh.write(raw[: len(raw) // 2])
        self._assert_rejected()

    def test_flipped_delta_byte_rejected(self):
        path = os.path.join(self.chain, "00000002.delta")
        raw = bytearray(open(path, "rb").read())
        raw[len(raw) // 2] ^= 0xFF
        with open(path, "wb") as fh:
            fh.write(bytes(raw))
        self._assert_rejected()

    def test_missing_delta_rejected(self):
        os.unlink(os.path.join(self.chain, "00000002.delta"))
        self._assert_rejected()

    def test_delta_missing_field_rejected(self):
        path = os.path.join(self.chain, "00000002.delta")
        raw = open(path, "rb").read()
        hlen = struct.unpack("<Q", raw[13:21])[0]
        header = json.loads(raw[21 : 21 + hlen])
        del header["prev"]
        self._repack_delta(path, raw, header)
        self._assert_rejected()

    def test_broken_predecessor_crc_rejected(self):
        path = os.path.join(self.chain, "00000003.delta")
        raw = open(path, "rb").read()
        hlen = struct.unpack("<Q", raw[13:21])[0]
        header = json.loads(raw[21 : 21 + hlen])
        header["prev"] = (header["prev"] + 1) & 0xFFFFFFFF
        self._repack_delta(path, raw, header)
        self._assert_rejected()

    def test_manifest_missing_field_rejected(self):
        path = os.path.join(self.chain, "manifest.json")
        manifest = json.load(open(path))
        del manifest["base_digest"]
        with open(path, "w") as fh:
            json.dump(manifest, fh)
        self._assert_rejected()

    def test_manifest_garbage_rejected(self):
        with open(os.path.join(self.chain, "manifest.json"), "wb") as fh:
            fh.write(b"not json")
        self._assert_rejected()

    def test_truncated_base_rejected(self):
        path = os.path.join(self.chain, "base.ckp")
        raw = open(path, "rb").read()
        with open(path, "wb") as fh:
            fh.write(raw[:-10])
        self._assert_rejected()

    def test_tampered_base_digest_rejected(self):
        # A base whose bytes no longer match the manifest is rejected.
        path = os.path.join(self.chain, "base.ckp")
        raw = bytearray(open(path, "rb").read())
        # Mutating a byte breaks the full-checkpoint CRC first; instead
        # mutate the manifest digest to a wrong value.
        manifest_path = os.path.join(self.chain, "manifest.json")
        manifest = json.load(open(manifest_path))
        manifest["base_digest"] = (manifest["base_digest"] + 1) & 0xFFFFFFFF
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh)
        self._assert_rejected()

    def test_orphan_unreferenced_delta_is_ignored(self):
        # Simulate an interrupted append: next delta exists, but the
        # manifest still points at the previous entry.
        orphan = os.path.join(self.chain, "00000005.delta")
        with open(orphan, "wb") as fh:
            fh.write(b"SEQDELTA1" + b"\x00" * 16)
        reader, _ = _fresh_stack()
        reader.load_incremental(self.chain)
        self.assertEqual(_full_bytes(reader), self.tip)

    def test_chain_recovers_after_restoring_files(self):
        path = os.path.join(self.chain, "00000002.delta")
        good = open(path, "rb").read()
        with open(path, "wb") as fh:
            fh.write(good[: len(good) // 2])
        self._assert_rejected()
        with open(path, "wb") as fh:
            fh.write(good)
        reader, _ = _fresh_stack()
        reader.load_incremental(self.chain)
        self.assertEqual(_full_bytes(reader), self.tip)

    def test_non_finite_state_refused_before_io(self):
        seq, _ = _fresh_stack()
        seq.parameters()[0]._set_values(
            [[float("inf"), 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        with self.assertRaises(ValueError):
            seq.save_incremental(os.path.join(self.td, "poison"))
        self.assertFalse(os.path.exists(os.path.join(self.td, "poison")))

    def _repack_delta(self, path, raw, header):
        end = raw.rfind(b"SEQDELTA1END")
        old_hlen = struct.unpack("<Q", raw[13:21])[0]
        payload = raw[21 + old_hlen : end]
        new_header = json.dumps(
            header, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        body = raw[:13] + struct.pack("<Q", len(new_header)) + new_header + payload
        crc = zlib.crc32(body[21:])
        with open(path, "wb") as fh:
            fh.write(body + b"SEQDELTA1END" + struct.pack("<I", crc))


if __name__ == "__main__":
    unittest.main()
