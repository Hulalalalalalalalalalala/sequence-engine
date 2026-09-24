"""Checkpoint container: fixed binary snapshots and incremental chains.

A *full* checkpoint fixes, in one self-describing document:

* every parameter tensor,
* every accumulated gradient (zeros when a parameter has none yet),
* the slice-boundary hidden state of the latest completed segment,
* the layer order and per-layer parameter shapes,
* the format version, the source version it was migrated from (if any)
  and the engine's pending-backward flag.

Wire format (all integers little-endian)::

    magic        8 bytes  b"SEQECKP1"
    version      uint32   format version (2)
    header_len   uint64   length of the JSON header in bytes
    header       header_len bytes of UTF-8 JSON (shapes, layer order, flags)
    payload      one leaf per tensor element, each a 1-byte tag
                 ("i" | "f") followed by int64 or float64
    end_magic    12 bytes b"SEQECKP1END"
    leaf_count   uint64   number of encoded leaves
    payload_crc  uint32   CRC-32 of the header bytes followed by the payload

Version 1 files use the same framing; they are read and migrated item by
item into the version 2 document shape, with ``src`` recording the
version the bytes were written in.  A migration that fails part-way
rejects the whole file -- nothing is silently filled in or dropped.

An *incremental chain* lives in a directory::

    base.ckp        full checkpoint of the first snapshot in the chain
    00000001.delta  per-layer delta (unchanged layers stored as references)
    00000002.delta  ...
    manifest.json   atomically published index: sequence id and base digest

Deltas only contain layers whose parameter *or* gradient leaves changed
relative to the previous chain entry; unchanged layers are referenced by
their 64-bit FNV-1a digest.  Reassembling a chain reproduces the full
state bit for bit, including the sign of negative zero.  Every delta
chains the CRC of its predecessor, so a torn, truncated or missing link
is detected and the whole chain is rejected with ValueError; a missing
chain path still raises FileNotFoundError and unwritable directories /
full disks still raise OSError.

Float leaves are emitted as raw IEEE-754 bytes, so values (including the
sign of negative zero) round-trip bit for bit.  Non-finite floats and
integers that do not fit in int64 are refused up front.
"""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
import threading
import zlib

MAGIC = b"SEQECKP1"
END_MAGIC = b"SEQECKP1END"

FORMAT_VERSION = 2
_PREVIOUS_FORMAT_VERSION = 1
_SUPPORTED_READ_VERSIONS = (1, 2)

# Incremental-chain file names.
BASE_NAME = "base.ckp"
MANIFEST_NAME = "manifest.json"
_DELTA_SUFFIX = ".delta"

_TAG_INT = ord("i")
_TAG_FLOAT = ord("f")

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

_TMP_PREFIX = ".seqckp.tmp-"

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_FNV_MASK = 0xFFFFFFFFFFFFFFFF

_UINT32_MAX = 0xFFFFFFFF
_UINT64_MAX = 0xFFFFFFFFFFFFFFFF


class CheckpointError(ValueError):
    """A checkpoint is truncated, corrupt, or structurally invalid."""


# ---------------------------------------------------------------------------
# Shape / tree helpers
# ---------------------------------------------------------------------------


def _check_shape(shape):
    if not isinstance(shape, list):
        raise CheckpointError("tensor shape must be a list of ints")
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise CheckpointError("tensor shape dimensions must be positive ints")


def _shape_matches(tree, shape):
    node = tree
    for dim in shape:
        if not isinstance(node, list) or len(node) != dim:
            return False
        node = node[0] if node else None
    return not isinstance(node, list)


def _shape_size(shape):
    size = 1
    for dim in shape:
        size *= dim
    return size


def _check_leaf(value):
    if isinstance(value, bool):
        raise CheckpointError("checkpoint values must be numbers, not booleans")
    if isinstance(value, int):
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise CheckpointError(
                "checkpoint contains an integer too large to represent (int64)"
            )
        return _TAG_INT
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError("checkpoint contains a non-finite floating value")
        return _TAG_FLOAT
    raise CheckpointError(
        f"checkpoint values must be numbers, got {type(value).__name__}"
    )


def _iter_leaves(tree, shape):
    if shape:
        if not isinstance(tree, list) or len(tree) != shape[0]:
            raise CheckpointError("tensor tree does not match its declared shape")
        for item in tree:
            yield from _iter_leaves(item, shape[1:])
    else:
        if isinstance(tree, list):
            raise CheckpointError("tensor tree does not match its declared shape")
        yield tree


def _pack_leaf(value):
    tag = _check_leaf(value)
    if tag == _TAG_INT:
        return struct.pack("<bq", _TAG_INT, value)
    return struct.pack("<bd", _TAG_FLOAT, value)


# ---------------------------------------------------------------------------
# Freezing: snapshot document -> (JSON header, binary payload)
# ---------------------------------------------------------------------------


def _freeze_entry(entry):
    if not isinstance(entry, dict) or set(entry) != {"s", "v"}:
        raise CheckpointError("tensor entry must be an object with 's' and 'v'")
    shape = entry["s"]
    _check_shape(shape)
    tree = entry["v"]
    if not _shape_matches(tree, shape):
        raise CheckpointError("tensor tree does not match its declared shape")
    chunks = [_pack_leaf(leaf) for leaf in _iter_leaves(tree, shape)]
    return list(shape), chunks


def _freeze_layer(layer):
    if not isinstance(layer, dict) or set(layer) != {"kind", "shapes"}:
        raise CheckpointError("layer descriptor must be an object with kind/shapes")
    kind = layer["kind"]
    if not isinstance(kind, str) or not kind:
        raise CheckpointError("layer kind must be a non-empty string")
    shapes = layer["shapes"]
    if not isinstance(shapes, list):
        raise CheckpointError("layer parameter shapes must be a list")
    frozen_shapes = []
    for shape in shapes:
        _check_shape(shape)
        frozen_shapes.append(list(shape))
    return {"kind": kind, "shapes": frozen_shapes}


def _freeze(document, version=FORMAT_VERSION, source_version=None):
    if not isinstance(document, dict):
        raise CheckpointError("checkpoint document must be an object")
    payload = bytearray()
    header = {"v": version}
    if version >= 2:
        header["src"] = source_version

    entries = document.get("params")
    if not isinstance(entries, list):
        raise CheckpointError("checkpoint must record a parameter tensor list")
    params_header = []
    for entry in entries:
        shape, chunks = _freeze_entry(entry)
        params_header.append(shape)
        for chunk in chunks:
            payload += chunk
    header["params"] = params_header

    grads = document.get("grads")
    if not isinstance(grads, list) or len(grads) != len(entries):
        raise CheckpointError("checkpoint must record one gradient slot per parameter")
    grads_header = []
    for entry in grads:
        shape, chunks = _freeze_entry(entry)
        grads_header.append(shape)
        for chunk in chunks:
            payload += chunk
    header["grads"] = grads_header

    hidden = document.get("hidden")
    if hidden is None:
        header["hidden"] = None
    else:
        if not isinstance(hidden, list) or not hidden:
            raise CheckpointError("hidden state must be a non-empty list or null")
        hidden_header = []
        for entry in hidden:
            shape, chunks = _freeze_entry(entry)
            hidden_header.append(shape)
            for chunk in chunks:
                payload += chunk
        header["hidden"] = hidden_header

    layers = document.get("layers")
    if not isinstance(layers, list) or not layers:
        raise CheckpointError("checkpoint must record the layer order")
    header["layers"] = [_freeze_layer(layer) for layer in layers]
    if (
        sum(len(layer["shapes"]) for layer in header["layers"])
        != len(header["params"])
    ):
        raise CheckpointError("layer parameter counts do not match the parameter list")

    pending = document.get("pending")
    if not isinstance(pending, bool):
        raise CheckpointError("pending flag must be a boolean")
    header["pending"] = pending
    return header, payload


def build_bytes(document):
    """Serialize a snapshot document to the fixed binary representation."""
    return _build_with_source(document, FORMAT_VERSION, FORMAT_VERSION)


def _build_with_source(document, version, source_version):
    header, payload = _freeze(document, version, source_version)
    return _frame(header, payload, version)


def _frame(header, payload, version):
    header_bytes = json.dumps(
        header, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    body = bytearray()
    body += MAGIC
    body += struct.pack("<I", version)
    body += struct.pack("<Q", len(header_bytes))
    body += header_bytes
    crc = zlib.crc32(header_bytes)
    leaf_count = len(payload) // 9
    crc = zlib.crc32(payload, crc)
    body += payload
    body += END_MAGIC
    body += struct.pack("<Q", leaf_count)
    body += struct.pack("<I", crc & _UINT32_MAX)
    return bytes(body)


# ---------------------------------------------------------------------------
# Parsing: raw bytes -> validated snapshot document
# ---------------------------------------------------------------------------


_PREFIX_LEN = len(MAGIC) + 4 + 8


def _read_frame(raw):
    """Validate framing/CRC and return ``(version, header_dict, payload)``."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise CheckpointError("checkpoint must be bytes")
    data = bytes(raw)

    if len(data) < _PREFIX_LEN:
        raise CheckpointError("checkpoint is truncated while reading its header")
    if data[: len(MAGIC)] != MAGIC:
        raise CheckpointError("not a sequence-engine checkpoint (bad magic)")
    (version,) = struct.unpack("<I", data[len(MAGIC) : len(MAGIC) + 4])
    if version not in _SUPPORTED_READ_VERSIONS:
        raise CheckpointError(
            f"unsupported checkpoint version {version}; this engine reads "
            f"versions {_PREVIOUS_FORMAT_VERSION} and {FORMAT_VERSION}"
        )
    (header_len,) = struct.unpack(
        "<Q", data[len(MAGIC) + 4 : len(MAGIC) + 4 + 8]
    )
    if header_len <= 0:
        raise CheckpointError("checkpoint header length is invalid")

    # The fixed trailer is always the last 24 bytes; locate it via end_magic
    # so a torn write (missing/partial trailer) cannot be mistaken for data.
    trailer_pos = data.rfind(END_MAGIC)
    if trailer_pos < 0 or len(data) - trailer_pos != len(END_MAGIC) + 8 + 4:
        raise CheckpointError("checkpoint trailer is missing or the file is torn")
    header_start = _PREFIX_LEN
    payload_end = trailer_pos
    if header_len > payload_end - header_start:
        raise CheckpointError("checkpoint header overruns the payload area")
    header_bytes = data[header_start : header_start + header_len]

    (leaf_count, stored_crc) = struct.unpack(
        "<QI", data[trailer_pos + len(END_MAGIC) :]
    )
    # CRC covers exactly the header followed by the payload.
    actual_crc = zlib.crc32(data[header_start:payload_end])
    if actual_crc != stored_crc:
        raise CheckpointError("checkpoint CRC mismatch: the file is corrupt or torn")

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"checkpoint header is invalid: {exc}") from exc
    if not isinstance(header, dict):
        raise CheckpointError("checkpoint header must be a JSON object")
    if header.get("v") != version:
        raise CheckpointError("checkpoint header version does not match the file")

    if leaf_count > _UINT64_MAX or not isinstance(leaf_count, int):
        raise CheckpointError("checkpoint leaf count is invalid")

    payload = data[header_start + header_len : payload_end]
    if len(payload) != leaf_count * 9:
        raise CheckpointError("checkpoint payload has the wrong size")
    return version, header, payload


def parse_bytes(raw):
    """Validate and decode a checkpoint; return the snapshot document.

    Version 1 files are migrated item by item into the version 2 document
    shape.  The returned document carries a ``src`` field recording the
    version the bytes were actually written in (1 for migrated files, 2
    for native files); every value is range-checked during migration, so a
    half-failed migration rejects the whole file.
    """
    version, header, payload = _read_frame(raw)
    if version == _PREVIOUS_FORMAT_VERSION:
        return _migrate_v1(header, payload)
    return _document_from_header(header, payload, FORMAT_VERSION)


def _declared_leaf_count(header):
    """Number of leaves the header's tensor shapes demand."""
    total = sum(_shape_size(shape) for shape in header["params"])
    total += sum(_shape_size(shape) for shape in header["grads"])
    if header["hidden"] is not None:
        total += sum(_shape_size(shape) for shape in header["hidden"])
    return total


def _document_from_header(header, payload, version):
    params_shapes = _parse_header(header, version)
    if len(payload) // 9 != _declared_leaf_count(header):
        raise CheckpointError("checkpoint leaf count does not match the header")
    reader = _LeafReader(payload)
    params = [{"s": shape, "v": reader.tree(shape)} for shape in params_shapes]
    grads = [{"s": shape, "v": reader.tree(shape)} for shape in header["grads"]]
    hidden = (
        None
        if header["hidden"] is None
        else [
            {"s": shape, "v": reader.tree(shape)} for shape in header["hidden"]
        ]
    )
    if reader.remaining() != 0:
        raise CheckpointError("checkpoint payload has trailing bytes")
    return {
        "params": params,
        "grads": grads,
        "hidden": hidden,
        "layers": header["layers"],
        "pending": header["pending"],
        "src": header.get("src", version),
    }


def _migrate_v1(header, payload):
    """Move a version 1 document into the version 2 shape, item by item.

    Each field is read and range-checked independently; any failure
    raises before a document is handed back, so a checkpoint that can
    only be half-migrated is refused wholesale.
    """
    try:
        params_shapes = _parse_header(header, _PREVIOUS_FORMAT_VERSION)
    except CheckpointError:
        raise
    except Exception as exc:  # defensive: never half-fill a migration
        raise CheckpointError(f"version 1 migration failed: {exc}") from exc
    if len(payload) // 9 != _declared_leaf_count(header):
        raise CheckpointError(
            "version 1 migration failed: leaf count does not match the header"
        )
    reader = _LeafReader(payload)

    def entries(shapes):
        out = []
        for shape in shapes:
            tree = reader.tree(shape)
            _validate_migrated_tree(tree, shape)
            out.append({"s": list(shape), "v": tree})
        return out

    try:
        params = entries(params_shapes)
        grads = entries(header["grads"])
        if header["hidden"] is None:
            hidden = None
        else:
            hidden = entries(header["hidden"])
        layers = _migrate_v1_layers(header["layers"])
        pending = header["pending"]
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"version 1 migration failed: {exc}") from exc
    if reader.remaining() != 0:
        raise CheckpointError("version 1 migration failed: payload has trailing bytes")
    return {
        "params": params,
        "grads": grads,
        "hidden": hidden,
        "layers": layers,
        "pending": pending,
        "src": _PREVIOUS_FORMAT_VERSION,
    }


def _migrate_v1_layers(layers):
    out = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != {"kind", "shapes"}:
            raise CheckpointError(f"version 1 layer {index} is malformed")
        kind = layer["kind"]
        if not isinstance(kind, str) or not kind:
            raise CheckpointError(f"version 1 layer {index} kind is malformed")
        shapes = []
        for shape in layer["shapes"]:
            _check_shape(shape)
            shapes.append(list(shape))
        out.append({"kind": kind, "shapes": shapes})
    return out


def _validate_migrated_tree(tree, shape):
    if shape:
        if not isinstance(tree, list) or len(tree) != shape[0]:
            raise CheckpointError("version 1 tensor values do not match their shape")
        for item in tree:
            _validate_migrated_tree(item, shape[1:])
        return
    if isinstance(tree, bool) or not isinstance(tree, (int, float)):
        raise CheckpointError("version 1 tensor leaves must be numbers")
    if isinstance(tree, int) and not _INT64_MIN <= tree <= _INT64_MAX:
        raise CheckpointError("version 1 integer leaf is outside int64 range")
    if isinstance(tree, float) and not math.isfinite(tree):
        raise CheckpointError("version 1 float leaf is non-finite")


def _parse_header(header, version):
    required = {"v", "params", "grads", "hidden", "layers", "pending"}
    if version >= 2:
        required.add("src")
    if not isinstance(header, dict) or set(header) != required:
        if not isinstance(header, dict):
            raise CheckpointError("checkpoint header must be a JSON object")
        missing = required - set(header)
        unknown = set(header) - required
        details = []
        if missing:
            details.append("missing field(s): " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown field(s): " + ", ".join(sorted(unknown)))
        raise CheckpointError("checkpoint header schema mismatch: " + "; ".join(details))
    if not isinstance(header["v"], int) or isinstance(header["v"], bool):
        raise CheckpointError("checkpoint version must be an integer")
    if version >= 2:
        src = header["src"]
        if not isinstance(src, int) or isinstance(src, bool) or src not in (
            _PREVIOUS_FORMAT_VERSION,
            FORMAT_VERSION,
        ):
            raise CheckpointError("checkpoint source version is not supported")
    params = _parse_shape_list(header["params"], "params", allow_empty=True)
    grads = _parse_shape_list(header["grads"], "grads", allow_empty=True)
    if len(params) != len(grads) or params != grads:
        raise CheckpointError("parameter and gradient shape lists must match")
    if header["hidden"] is not None:
        _parse_shape_list(header["hidden"], "hidden")
    layers = header["layers"]
    if not isinstance(layers, list) or not layers:
        raise CheckpointError("layer order must be a non-empty list")
    layer_shapes = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != {"kind", "shapes"}:
            raise CheckpointError(f"layer {index} descriptor must have kind/shapes")
        if not isinstance(layer["kind"], str) or not layer["kind"]:
            raise CheckpointError(f"layer {index} kind must be a non-empty string")
        shapes = _parse_shape_list(
            layer["shapes"], f"layer {index} shapes", allow_empty=True
        )
        layer_shapes.extend(shapes)
    if layer_shapes != params:
        raise CheckpointError(
            "layer order/parameter shapes do not match the parameter tensor list"
        )
    if not isinstance(header["pending"], bool):
        raise CheckpointError("pending flag must be a boolean")
    return params


def _parse_shape_list(value, what, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise CheckpointError(f"{what} must be a list of shapes")
    shapes = []
    for shape in value:
        _check_shape(shape)
        shapes.append(list(shape))
    return shapes


class _LeafReader:
    def __init__(self, payload):
        self._payload = payload
        self._pos = 0

    def _leaf(self):
        tag = self._payload[self._pos]
        chunk = self._payload[self._pos + 1 : self._pos + 9]
        self._pos += 9
        if tag == _TAG_INT:
            value = struct.unpack("<q", chunk)[0]
        elif tag == _TAG_FLOAT:
            value = struct.unpack("<d", chunk)[0]
        else:
            raise CheckpointError(f"unknown leaf tag {tag!r} in payload")
        # Range-check every migrated/loaded leaf: oversized ints or
        # non-finite floats are refused here even though the raw bytes
        # could be unpacked.
        _check_leaf(value)
        return value

    def tree(self, shape):
        if not shape:
            return self._leaf()
        return [self.tree(shape[1:]) for _ in range(shape[0])]

    def remaining(self):
        return (len(self._payload) - self._pos) // 9


# ---------------------------------------------------------------------------
# Atomic file / memory IO
# ---------------------------------------------------------------------------


def save_bytes(document, target):
    """Write *document* to a path (str/os.PathLike) or a bytearray target.

    File writes go through a temporary file in the destination directory
    and are atomically renamed into place, overwriting any existing file
    without leaving a backup.  A torn temporary file is never renamed, so
    a crashed process can never leave a half-written checkpoint at the
    final path.
    """
    raw = build_bytes(document)
    return _write_raw(raw, target)


def _write_raw(raw, target):
    if isinstance(target, bytearray):
        target.clear()
        target.extend(raw)
        return None
    if isinstance(target, (str, os.PathLike)):
        return _write_path(raw, os.fspath(target))
    raise TypeError("save target must be a path or bytearray")


def _write_path(raw, path):
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=directory)
    committed = False
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        committed = True
        _fsync_directory(directory)
    except BaseException:
        if not committed:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
        raise
    return None


def _fsync_directory(directory):
    """Best-effort durability for the rename; harmless where unsupported."""
    if not hasattr(os, "O_RDONLY") or not hasattr(os, "fsync"):
        return
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def load_bytes(source):
    """Read and validate a checkpoint from a path or from bytes/bytearray."""
    if isinstance(source, (str, os.PathLike)):
        with open(os.fspath(source), "rb") as fh:
            raw = fh.read()
        return parse_bytes(raw)
    if isinstance(source, (bytes, bytearray, memoryview)):
        return parse_bytes(source)
    raise TypeError("load source must be a path or bytes")


# ---------------------------------------------------------------------------
# Layer slicing: documents <-> per-layer pieces
# ---------------------------------------------------------------------------


def _slice_layers(document):
    """Split a full document into one descriptor per layer.

    Layer pieces own their parameter/gradient entries (in registration
    order) but share the hidden state, which lives outside any single
    layer.
    """
    params = document["params"]
    grads = document["grads"]
    layers = document["layers"]
    pieces = []
    offset = 0
    for layer in layers:
        count = len(layer["shapes"])
        pieces.append(
            {
                "kind": layer["kind"],
                "shapes": [list(shape) for shape in layer["shapes"]],
                "params": [
                    {"s": list(params[offset + i]["s"]),
                     "v": params[offset + i]["v"]}
                    for i in range(count)
                ],
                "grads": [
                    {"s": list(grads[offset + i]["s"]),
                     "v": grads[offset + i]["v"]}
                    for i in range(count)
                ],
            }
        )
        offset += count
    if offset != len(params):
        raise CheckpointError("layer parameter counts do not slice the parameters")
    return pieces


def _join_layers(pieces, hidden, pending):
    params = []
    grads = []
    layers = []
    for piece in pieces:
        layers.append({"kind": piece["kind"], "shapes": piece["shapes"]})
        params.extend(piece["params"])
        grads.extend(piece["grads"])
    return {
        "params": params,
        "grads": grads,
        "hidden": hidden,
        "layers": layers,
        "pending": pending,
    }



# ---------------------------------------------------------------------------
# Content digests
# ---------------------------------------------------------------------------


def _fnv64_continue(h, data):
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME) & _FNV_MASK
    return h


# ---------------------------------------------------------------------------
# Generic tensor-group frames (used inside deltas for layers and hidden)
# ---------------------------------------------------------------------------

_GROUP_MAGIC = b"SEQGRP1"
_GROUP_END_MAGIC = b"SEQGRP1END"


def _build_group(header, leaves):
    """Frame a JSON *header* and an iterable of packed 9-byte leaves."""
    header_bytes = json.dumps(
        header, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    payload = bytearray()
    count = 0
    for chunk in leaves:
        payload += chunk
        count += 1
    crc = zlib.crc32(header_bytes)
    crc = zlib.crc32(payload, crc)
    framed = bytearray()
    framed += _GROUP_MAGIC
    framed += struct.pack("<I", FORMAT_VERSION)
    framed += struct.pack("<Q", len(header_bytes))
    framed += header_bytes
    framed += payload
    framed += _GROUP_END_MAGIC
    framed += struct.pack("<Q", count)
    framed += struct.pack("<I", crc & _UINT32_MAX)
    return bytes(framed)


def _parse_group(raw):
    """Validate a group frame; return ``(header, leaf_reader)``."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise CheckpointError("group frame must be bytes")
    data = bytes(raw)
    prefix = len(_GROUP_MAGIC) + 4 + 8
    if len(data) < prefix:
        raise CheckpointError("group frame is truncated")
    if data[: len(_GROUP_MAGIC)] != _GROUP_MAGIC:
        raise CheckpointError("not a sequence-engine group frame (bad magic)")
    (version,) = struct.unpack(
        "<I", data[len(_GROUP_MAGIC) : len(_GROUP_MAGIC) + 4]
    )
    if version != FORMAT_VERSION:
        raise CheckpointError(f"unsupported group frame version {version}")
    (header_len,) = struct.unpack(
        "<Q", data[len(_GROUP_MAGIC) + 4 : len(_GROUP_MAGIC) + 4 + 8]
    )
    if header_len <= 0:
        raise CheckpointError("group frame header length is invalid")
    end_pos = data.rfind(_GROUP_END_MAGIC)
    if end_pos < 0 or len(data) - end_pos != len(_GROUP_END_MAGIC) + 8 + 4:
        raise CheckpointError("group frame trailer is missing or the file is torn")
    header_start = prefix
    if header_len > end_pos - header_start:
        raise CheckpointError("group frame header overruns its payload area")
    header_bytes = data[header_start : header_start + header_len]
    leaf_count, stored_crc = struct.unpack(
        "<QI", data[end_pos + len(_GROUP_END_MAGIC) :]
    )
    actual_crc = zlib.crc32(data[header_start:end_pos]) & _UINT32_MAX
    if actual_crc != stored_crc:
        raise CheckpointError("group frame CRC mismatch: the chain is corrupt or torn")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"group frame header is invalid: {exc}") from exc
    payload = data[header_start + header_len : end_pos]
    if len(payload) != leaf_count * 9:
        raise CheckpointError("group frame payload has the wrong size")
    return header, _LeafReader(payload)


def _entry_leaves(entry):
    shape = entry["s"]
    return (_pack_leaf(leaf) for leaf in _iter_leaves(entry["v"], shape))


def _build_layer_body(piece):
    """Serialize one layer piece (zero-parameter layers allowed)."""
    header = {
        "kind": piece["kind"],
        "shapes": [list(s) for s in piece["shapes"]],
        "params": [list(e["s"]) for e in piece["params"]],
        "grads": [list(e["s"]) for e in piece["grads"]],
    }

    def leaves():
        for entry in piece["params"]:
            yield from _entry_leaves(entry)
        for entry in piece["grads"]:
            yield from _entry_leaves(entry)

    return _build_group(header, leaves())


def _parse_layer_body(raw):
    header, reader = _parse_group(raw)
    if not isinstance(header, dict) or set(header) != {"kind", "shapes", "params", "grads"}:
        raise CheckpointError("layer frame header has the wrong fields")
    kind = header["kind"]
    if not isinstance(kind, str) or not kind:
        raise CheckpointError("layer frame kind must be a non-empty string")
    shapes = _parse_shape_list(header["shapes"], "layer shapes", allow_empty=True)
    param_shapes = _parse_shape_list(
        header["params"], "layer parameter shapes", allow_empty=True
    )
    grad_shapes = _parse_shape_list(
        header["grads"], "layer gradient shapes", allow_empty=True
    )
    if param_shapes != shapes or grad_shapes != shapes:
        raise CheckpointError("layer frame parameter/gradient shapes disagree")
    params = [{"s": s, "v": reader.tree(s)} for s in param_shapes]
    grads = [{"s": s, "v": reader.tree(s)} for s in grad_shapes]
    if reader.remaining() != 0:
        raise CheckpointError("layer frame payload has trailing bytes")
    return {
        "kind": kind,
        "shapes": shapes,
        "params": params,
        "grads": grads,
    }


def _build_hidden_body(hidden):
    header = {"hidden": [list(e["s"]) for e in hidden]}

    def leaves():
        for entry in hidden:
            yield from _entry_leaves(entry)

    return _build_group(header, leaves())


def _parse_hidden_body(raw):
    header, reader = _parse_group(raw)
    if not isinstance(header, dict) or set(header) != {"hidden"}:
        raise CheckpointError("hidden frame header has the wrong fields")
    shapes = _parse_shape_list(header["hidden"], "hidden shapes")
    entries = [{"s": s, "v": reader.tree(s)} for s in shapes]
    if reader.remaining() != 0:
        raise CheckpointError("hidden frame payload has trailing bytes")
    return entries


def _layer_digest(piece):
    """FNV-1a digest of the canonical serialization of one layer piece."""
    return _fnv64_continue(_FNV_OFFSET, _build_layer_body(piece)) & _FNV_MASK


# ---------------------------------------------------------------------------
# Incremental chains
# ---------------------------------------------------------------------------

_DELTA_SUFFIX = ".delta"
_DELTA_MAGIC = b"SEQDELTA1"
_DELTA_END_MAGIC = b"SEQDELTA1END"
_LOCK_NAME = ".seqckp.lock"


def _delta_name(seq):
    return f"{seq:08d}{_DELTA_SUFFIX}"


def _read_chain_member(path):
    """Read a chain member; a missing member is chain corruption."""
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except FileNotFoundError:
        raise CheckpointError(
            f"incremental chain is missing a member: {os.path.basename(path)}"
        ) from None


def save_chain(document, directory):
    """Append *document* to the incremental chain at *directory*.

    The first call writes ``base.ckp``, delta ``00000001.delta`` and the
    manifest; later calls append one numbered delta holding only the
    layers whose content changed since the previous chain entry.
    Unchanged layers are stored as content digests and copied from the
    earlier entry during reassembly.  The manifest is published
    atomically last, so a crash (or a failed save) leaves the chain at
    the previous complete entry.  Concurrent writers to the same
    directory are serialized with an advisory lock; the bytes on disk
    always constitute one complete checkpoint, never a mix of two saves.

    Returns the new sequence id (1 for the first snapshot).
    """
    if not isinstance(directory, (str, os.PathLike)):
        raise TypeError("incremental save target must be a directory path")
    path = os.fspath(directory)

    # Validate the document fully before any IO: build_bytes freezes every
    # leaf and rejects oversized integers / non-finite floats deterministically.
    base_raw = build_bytes(document)
    pieces = _slice_layers(document)
    digests = [_layer_digest(piece) for piece in pieces]

    os.makedirs(path, exist_ok=True)
    lock_fd = _chain_lock(path)
    try:
        manifest_path = os.path.join(path, MANIFEST_NAME)
        base_path = os.path.join(path, BASE_NAME)
        if not os.path.exists(manifest_path):
            seq = 0
            expected_base_digest = None
            prev_pieces = None
            prev_digests = None
            prev_crc = 0
            _write_atomic_path(base_path, base_raw)
        else:
            manifest = _read_manifest(manifest_path)
            seq = manifest["seq"]
            expected_base_digest = manifest["base_digest"]
            base_doc = parse_bytes(_read_chain_member(base_path))
            if _document_digest(base_doc) != expected_base_digest:
                raise CheckpointError("incremental chain base does not match its manifest")
            prev_doc = _assemble_chain(path, manifest, base_doc)
            prev_pieces = _slice_layers(prev_doc)
            prev_digests = [_layer_digest(p) for p in prev_pieces]
            prev_crc = manifest["prev_crc"]

        if seq == 0:
            refs = [{"i": i, "d": digest} for i, digest in enumerate(digests)]
            changed = []
        else:
            if len(pieces) != len(prev_pieces):
                raise CheckpointError("incremental save changes the number of layers")
            refs = [{"i": i, "d": digest} for i, digest in enumerate(digests)]
            changed = [
                i for i, (digest, old) in enumerate(zip(digests, prev_digests))
                if digest != old
            ]
        header = {
            "seq": seq + 1,
            "n": len(pieces),
            "layers": refs,
            "changed": changed,
            "pending": bool(document["pending"]),
            "prev": prev_crc,
        }
        delta_raw = _build_delta(header, pieces, document["hidden"])
        new_seq = seq + 1
        _write_atomic_path(os.path.join(path, _delta_name(new_seq)), delta_raw)
        new_crc = zlib.crc32(delta_raw) & _UINT32_MAX

        manifest = {
            "seq": new_seq,
            "v": FORMAT_VERSION,
            "base_digest": _document_digest_bytes(base_raw)
            if seq == 0
            else expected_base_digest,
            "prev_crc": new_crc,
        }
        _write_atomic_path(
            manifest_path,
            json.dumps(
                manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8"),
        )
        return new_seq
    finally:
        _chain_unlock(lock_fd)


def _build_delta(header, pieces, hidden):
    header_bytes = json.dumps(
        header, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    parts = []
    for index in header["changed"]:
        body = _build_layer_body(pieces[index])
        parts.append(struct.pack("<Q", len(body)))
        parts.append(body)
    if hidden is None:
        parts.append(struct.pack("<Q", 0))
    else:
        body = _build_hidden_body(hidden)
        parts.append(struct.pack("<Q", len(body)))
        parts.append(body)

    crc = zlib.crc32(header_bytes)
    framed = bytearray()
    framed += _DELTA_MAGIC
    framed += struct.pack("<I", FORMAT_VERSION)
    framed += struct.pack("<Q", len(header_bytes))
    framed += header_bytes
    for part in parts:
        framed += part
        crc = zlib.crc32(part, crc)
    framed += _DELTA_END_MAGIC
    framed += struct.pack("<I", crc & _UINT32_MAX)
    return bytes(framed)


def _read_delta(raw):
    """Validate a delta frame; return ``(header, changed_bodies, hidden)``."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise CheckpointError("delta must be bytes")
    data = bytes(raw)
    prefix = len(_DELTA_MAGIC) + 4 + 8
    if len(data) < prefix:
        raise CheckpointError("delta is truncated")
    if data[: len(_DELTA_MAGIC)] != _DELTA_MAGIC:
        raise CheckpointError("not a sequence-engine delta (bad magic)")
    (version,) = struct.unpack(
        "<I", data[len(_DELTA_MAGIC) : len(_DELTA_MAGIC) + 4]
    )
    if version != FORMAT_VERSION:
        raise CheckpointError(f"unsupported delta version {version}")
    (header_len,) = struct.unpack(
        "<Q", data[len(_DELTA_MAGIC) + 4 : len(_DELTA_MAGIC) + 4 + 8]
    )
    if header_len <= 0:
        raise CheckpointError("delta header length is invalid")
    end_pos = data.rfind(_DELTA_END_MAGIC)
    if end_pos < 0 or len(data) - end_pos != len(_DELTA_END_MAGIC) + 4:
        raise CheckpointError("delta trailer is missing or the file is torn")
    header_start = prefix
    if header_len > end_pos - header_start:
        raise CheckpointError("delta header overruns its payload area")
    header_bytes = data[header_start : header_start + header_len]
    (stored_crc,) = struct.unpack("<I", data[end_pos + len(_DELTA_END_MAGIC) :])
    actual_crc = zlib.crc32(data[header_start:end_pos]) & _UINT32_MAX
    if actual_crc != stored_crc:
        raise CheckpointError("delta CRC mismatch: the chain is corrupt or torn")

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"delta header is invalid: {exc}") from exc
    _validate_delta_header(header)

    bodies = {}
    pos = header_start + header_len
    for index in header["changed"]:
        if pos + 8 > end_pos:
            raise CheckpointError("delta is truncated while reading a layer body")
        (body_len,) = struct.unpack("<Q", data[pos : pos + 8])
        pos += 8
        if body_len <= 0 or pos + body_len > end_pos:
            raise CheckpointError("delta layer body has a bad length")
        bodies[index] = _parse_layer_body(data[pos : pos + body_len])
        pos += body_len

    if pos + 8 > end_pos:
        raise CheckpointError("delta is truncated before its hidden-state block")
    (hidden_len,) = struct.unpack("<Q", data[pos : pos + 8])
    pos += 8
    if hidden_len == 0:
        hidden = None
    else:
        if pos + hidden_len != end_pos:
            raise CheckpointError("delta hidden-state block has a bad length")
        hidden = _parse_hidden_body(data[pos : pos + hidden_len])
        pos += hidden_len
    if pos != end_pos:
        raise CheckpointError("delta payload has trailing bytes")
    return header, bodies, hidden


def _validate_delta_header(header):
    required = {"seq", "n", "layers", "changed", "pending", "prev"}
    if not isinstance(header, dict) or set(header) != required:
        raise CheckpointError("delta header has the wrong fields")
    seq = header["seq"]
    if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
        raise CheckpointError("delta sequence id must be a positive integer")
    n = header["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise CheckpointError("delta layer count must be a positive integer")
    layers = header["layers"]
    if not isinstance(layers, list) or len(layers) != n:
        raise CheckpointError("delta layer reference list has the wrong length")
    refs = {}
    for ref in layers:
        if not isinstance(ref, dict) or set(ref) != {"i", "d"}:
            raise CheckpointError("delta layer reference is malformed")
        i, digest = ref["i"], ref["d"]
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n:
            raise CheckpointError("delta layer index is out of range")
        if i in refs:
            raise CheckpointError("delta layer indices are not unique")
        if (
            isinstance(digest, bool)
            or not isinstance(digest, int)
            or not 0 <= digest <= _FNV_MASK
        ):
            raise CheckpointError("delta layer digest is invalid")
        refs[i] = digest
    changed = header["changed"]
    if not isinstance(changed, list):
        raise CheckpointError("delta changed-layer list is malformed")
    if any(
        isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n
        for i in changed
    ):
        raise CheckpointError("delta changed-layer index is out of range")
    if len(set(changed)) != len(changed):
        raise CheckpointError("delta changed-layer indices are not unique")
    prev = header["prev"]
    if (
        isinstance(prev, bool)
        or not isinstance(prev, int)
        or not 0 <= prev <= _UINT32_MAX
    ):
        raise CheckpointError("delta predecessor CRC is invalid")
    if not isinstance(header["pending"], bool):
        raise CheckpointError("delta pending flag must be a boolean")


def load_chain(directory):
    """Reassemble the incremental chain at *directory* into a full document.

    The result is bitwise identical to a full checkpoint taken at the
    latest chain entry.  Any truncation, corruption, missing field,
    missing chain member, digest mismatch or broken predecessor CRC
    rejects the whole chain with ValueError -- reassembly never returns a
    partially merged document.  Asking for a directory/path that does not
    exist raises FileNotFoundError; IO failures propagate as OSError.
    """
    if not isinstance(directory, (str, os.PathLike)):
        raise TypeError("incremental load source must be a directory path")
    path = os.fspath(directory)
    # Opening the user-requested manifest: "path not found" stays a
    # FileNotFoundError by contract.
    with open(os.path.join(path, MANIFEST_NAME), "rb") as fh:
        manifest_raw = fh.read()
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"chain manifest is invalid: {exc}") from exc
    manifest = _validate_manifest(manifest)
    base_doc = parse_bytes(_read_chain_member(os.path.join(path, BASE_NAME)))
    return _assemble_chain(path, manifest, base_doc)


def _read_manifest(manifest_path):
    with open(manifest_path, "rb") as fh:
        raw = fh.read()
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"chain manifest is invalid: {exc}") from exc
    return _validate_manifest(manifest)


def _validate_manifest(manifest):
    required = {"seq", "v", "base_digest", "prev_crc"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise CheckpointError("chain manifest has the wrong fields")
    seq = manifest["seq"]
    if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
        raise CheckpointError("chain manifest sequence id must be a positive int")
    if manifest["v"] != FORMAT_VERSION:
        raise CheckpointError("chain manifest version is unsupported")
    digest = manifest["base_digest"]
    if (
        isinstance(digest, bool)
        or not isinstance(digest, int)
        or not 0 <= digest <= _FNV_MASK
    ):
        raise CheckpointError("chain manifest base digest is invalid")
    prev_crc = manifest["prev_crc"]
    if (
        isinstance(prev_crc, bool)
        or not isinstance(prev_crc, int)
        or not 0 <= prev_crc <= _UINT32_MAX
    ):
        raise CheckpointError("chain manifest predecessor CRC is invalid")
    return manifest


def _assemble_chain(path, manifest, base_doc):
    """Validate every link and merge layers into the document at the tip."""
    if _document_digest(base_doc) != manifest["base_digest"]:
        raise CheckpointError("incremental chain base does not match its manifest")
    pieces = _slice_layers(base_doc)
    layer_count = len(pieces)
    digests = [_layer_digest(p) for p in pieces]
    hidden = base_doc["hidden"]
    pending = base_doc["pending"]
    expected_prev_crc = 0
    seq = manifest["seq"]

    for current in range(1, seq + 1):
        raw = _read_chain_member(os.path.join(path, _delta_name(current)))
        header, bodies, delta_hidden = _read_delta(raw)
        if header["seq"] != current:
            raise CheckpointError(
                f"delta {current} has an out-of-order sequence id {header['seq']}"
            )
        if header["n"] != layer_count:
            raise CheckpointError(f"delta {current} changes the number of layers")
        if header["prev"] != expected_prev_crc:
            raise CheckpointError(
                f"delta {current} predecessor CRC is broken: chain is not contiguous"
            )
        refs = header["layers"]
        if sorted(ref["i"] for ref in refs) != list(range(layer_count)):
            raise CheckpointError(f"delta {current} does not reference every layer")
        if sorted(bodies) != sorted(header["changed"]):
            raise CheckpointError(
                f"delta {current} layer bodies do not match its header"
            )
        ref_digests = {ref["i"]: ref["d"] for ref in refs}
        for index, piece in bodies.items():
            if _layer_digest(piece) != ref_digests[index]:
                raise CheckpointError(
                    f"delta {current} layer {index} digest does not match its body"
                )
            pieces[index] = piece
            digests[index] = _layer_digest(piece)
        for index, digest in ref_digests.items():
            if index in bodies:
                continue
            if digests[index] != digest:
                raise CheckpointError(
                    f"delta {current} layer {index} digest does not match the "
                    f"reassembled state"
                )
        hidden = delta_hidden
        pending = header["pending"]
        expected_prev_crc = zlib.crc32(raw) & _UINT32_MAX

    if expected_prev_crc != manifest["prev_crc"]:
        raise CheckpointError("chain manifest does not match the last delta")
    document = _join_layers(pieces, hidden, pending)
    document["src"] = FORMAT_VERSION
    return document


def _document_digest(document):
    return _document_digest_bytes(build_bytes(document))


def _document_digest_bytes(raw):
    return zlib.crc32(raw) & _UINT32_MAX


# ---------------------------------------------------------------------------
# Directory lock for concurrent incremental writers
# ---------------------------------------------------------------------------


def _chain_lock(directory):
    """Acquire an exclusive lock for a chain directory.

    Uses an advisory ``flock`` where available (POSIX), serializing both
    threads and processes; on platforms without ``fcntl`` it falls back
    to a per-directory in-process lock, which is sufficient for the
    engine's multi-thread contract. The returned handle is passed back
    to :func:`_chain_unlock`.
    """
    lock_path = os.path.join(directory, _LOCK_NAME)
    try:
        import fcntl
    except ImportError:
        fallback = _fallback_locks.setdefault(
            os.path.abspath(directory), threading.Lock()
        )
        fallback.acquire()
        return ("thread", fallback)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    except BaseException:
        os.close(fd)
        raise
    return ("fd", fd)


def _chain_unlock(handle):
    kind, value = handle
    if kind == "thread":
        value.release()
        return
    try:
        import fcntl

        fcntl.flock(value, fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass
    finally:
        os.close(value)


_fallback_locks = {}


def _write_atomic_path(path, raw):
    """Write *raw* to *path* via a temp file + atomic replace."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=directory)
    committed = False
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        committed = True
        _fsync_directory(directory)
    except BaseException:
        if not committed:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
        raise
