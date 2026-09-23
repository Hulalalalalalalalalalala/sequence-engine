"""Checkpoint container: fixed binary snapshots for Sequential state.

A checkpoint fixes, in one self-describing document:

* every parameter tensor,
* every accumulated gradient (zeros when a parameter has none yet),
* the slice-boundary hidden state of the latest completed segment,
* the layer order and per-layer parameter shapes,
* the format version and the engine's pending-backward flag.

Wire format (all integers little-endian)::

    magic        8 bytes  b"SEQECKP1"
    version      uint32   format version
    header_len   uint64   length of the JSON header in bytes
    header       header_len bytes of UTF-8 JSON (shapes, layer order, flags)
    payload      one leaf per tensor element, each a 1-byte tag
                 ("i" | "f") followed by int64 or float64
    end_magic    12 bytes b"SEQECKP1END"
    leaf_count   uint64   number of encoded leaves
    payload_crc  uint32   CRC-32 of the header bytes followed by the payload

Float leaves are emitted as raw IEEE-754 bytes, so values (including the
sign of negative zero) round-trip bit for bit.  Non-finite floats and
integers that do not fit in int64 are refused up front.

Loading validates the whole document before anything is handed back:
truncation, corruption, a version mismatch or any structural problem
rejects the entire checkpoint with ValueError; nothing is silently filled
in.
"""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
import zlib

MAGIC = b"SEQECKP1"
END_MAGIC = b"SEQECKP1END"
FORMAT_VERSION = 1

_TAG_INT = ord("i")
_TAG_FLOAT = ord("f")

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

_TMP_PREFIX = ".seqckp.tmp-"


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
        node = node[0]
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


# ---------------------------------------------------------------------------
# Freezing: snapshot document -> (JSON header, binary payload chunks)
# ---------------------------------------------------------------------------


def _freeze_tree(tree, shape, payload):
    if shape:
        if not isinstance(tree, list) or len(tree) != shape[0]:
            raise CheckpointError("tensor tree does not match its declared shape")
        for item in tree:
            _freeze_tree(item, shape[1:], payload)
    else:
        if isinstance(tree, list):
            raise CheckpointError("tensor tree does not match its declared shape")
        tag = _check_leaf(tree)
        payload.append(
            struct.pack("<bq", _TAG_INT, tree)
            if tag == _TAG_INT
            else struct.pack("<bd", _TAG_FLOAT, tree)
        )


def _freeze_entry(entry, payload):
    if not isinstance(entry, dict) or set(entry) != {"s", "v"}:
        raise CheckpointError("tensor entry must be an object with 's' and 'v'")
    shape = entry["s"]
    _check_shape(shape)
    tree = entry["v"]
    if not _shape_matches(tree, shape):
        raise CheckpointError("tensor tree does not match its declared shape")
    _freeze_tree(tree, shape, payload)
    return list(shape)


def _freeze_layer(layer):
    if not isinstance(layer, dict) or set(layer) != {"kind", "shapes"}:
        raise CheckpointError("layer descriptor must be an object with kind/shapes")
    kind = layer["kind"]
    if not isinstance(kind, str) or not kind:
        raise CheckpointError("layer kind must be a non-empty string")
    shapes = layer["shapes"]
    if not isinstance(shapes, list):
        raise CheckpointError("layer parameter shapes must be a list")
    for shape in shapes:
        _check_shape(shape)
    return {"kind": kind, "shapes": [list(shape) for shape in shapes]}


def _freeze(document):
    if not isinstance(document, dict):
        raise CheckpointError("checkpoint document must be an object")
    payload = []
    header = {"v": FORMAT_VERSION}

    entries = document.get("params")
    if not isinstance(entries, list):
        raise CheckpointError("checkpoint must record a parameter tensor list")
    header["params"] = [_freeze_entry(entry, payload) for entry in entries]

    grads = document.get("grads")
    if not isinstance(grads, list) or len(grads) != len(entries):
        raise CheckpointError("checkpoint must record one gradient slot per parameter")
    header["grads"] = [_freeze_entry(entry, payload) for entry in grads]

    hidden = document.get("hidden")
    if hidden is None:
        header["hidden"] = None
    else:
        if not isinstance(hidden, list) or not hidden:
            raise CheckpointError("hidden state must be a non-empty list or null")
        header["hidden"] = [_freeze_entry(entry, payload) for entry in hidden]

    layers = document.get("layers")
    if not isinstance(layers, list) or not layers:
        raise CheckpointError("checkpoint must record the layer order")
    header["layers"] = [_freeze_layer(layer) for layer in layers]
    if sum(len(layer["shapes"]) for layer in header["layers"]) != len(entries):
        raise CheckpointError("layer parameter counts do not match the parameter list")

    pending = document.get("pending")
    if not isinstance(pending, bool):
        raise CheckpointError("pending flag must be a boolean")
    header["pending"] = pending
    return header, payload


def build_bytes(document):
    """Serialize a snapshot document to the fixed binary representation."""
    header, payload = _freeze(document)
    header_bytes = json.dumps(
        header, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    body = bytearray()
    body += MAGIC
    body += struct.pack("<I", FORMAT_VERSION)
    body += struct.pack("<Q", len(header_bytes))
    body += header_bytes
    crc = zlib.crc32(header_bytes)
    leaf_count = 0
    for chunk in payload:
        body += chunk
        crc = zlib.crc32(chunk, crc)
        leaf_count += 1
    body += END_MAGIC
    body += struct.pack("<Q", leaf_count)
    body += struct.pack("<I", crc & 0xFFFFFFFF)
    return bytes(body)


# ---------------------------------------------------------------------------
# Parsing: raw bytes -> validated snapshot document
# ---------------------------------------------------------------------------


def parse_bytes(raw):
    """Validate and decode a checkpoint; return the snapshot document."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise CheckpointError("checkpoint must be bytes")
    data = bytes(raw)

    prefix_len = len(MAGIC) + 4 + 8
    if len(data) < prefix_len:
        raise CheckpointError("checkpoint is truncated while reading its header")
    if data[: len(MAGIC)] != MAGIC:
        raise CheckpointError("not a sequence-engine checkpoint (bad magic)")
    (version,) = struct.unpack("<I", data[len(MAGIC) : len(MAGIC) + 4])
    if version != FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint version {version}; expected {FORMAT_VERSION}"
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
    header_start = prefix_len
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
    if header.get("v") != FORMAT_VERSION:
        raise CheckpointError("checkpoint header version does not match")

    _parse_header(header)
    total_leaves = sum(
        _shape_size(shape)
        for group in (header["params"], header["grads"])
        for shape in group
    )
    if header["hidden"] is not None:
        total_leaves += sum(_shape_size(shape) for shape in header["hidden"])
    if total_leaves != leaf_count:
        raise CheckpointError("checkpoint leaf count does not match the header")

    payload = data[header_start + header_len : payload_end]
    if len(payload) != leaf_count * 9:
        raise CheckpointError("checkpoint payload has the wrong size")

    reader = _LeafReader(payload)
    params = [{"s": shape, "v": reader.tree(shape)} for shape in header["params"]]
    grads = [{"s": shape, "v": reader.tree(shape)} for shape in header["grads"]]
    hidden = (
        None
        if header["hidden"] is None
        else [{"s": shape, "v": reader.tree(shape)} for shape in header["hidden"]]
    )
    if reader.remaining() != 0:
        raise CheckpointError("checkpoint payload has trailing bytes")
    return {
        "params": params,
        "grads": grads,
        "hidden": hidden,
        "layers": header["layers"],
        "pending": header["pending"],
    }


def _parse_header(header):
    for key in ("params", "grads", "hidden", "layers", "pending"):
        if key not in header:
            raise CheckpointError(f"checkpoint header is missing field {key!r}")
    params = _parse_shape_list(header["params"], "params", allow_empty=True)
    grads = _parse_shape_list(header["grads"], "grads", allow_empty=True)
    if len(params) != len(grads) or params != grads:
        raise CheckpointError("parameter and gradient shape lists must match")
    if header["hidden"] is not None:
        hidden = _parse_shape_list(header["hidden"], "hidden")
    else:
        hidden = None
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
            return struct.unpack("<q", chunk)[0]
        if tag == _TAG_FLOAT:
            return struct.unpack("<d", chunk)[0]
        raise CheckpointError(f"unknown leaf tag {tag!r} in payload")

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
    if isinstance(target, bytearray):
        target.clear()
        target.extend(raw)
        return None
    if isinstance(target, (str, os.PathLike)):
        path = os.fspath(target)
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
    raise TypeError("save target must be a path or bytearray")


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
