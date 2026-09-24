"""Checkpoint container: versioned binary snapshots for Sequential state.

A full snapshot fixes, in one self-describing document:

* every parameter tensor,
* every accumulated gradient (zeros when a parameter has none yet),
* the Adam optimizer state: the step count ``t`` plus one first-moment
  tensor ``m`` and one second-moment tensor ``v`` per parameter (all
  zeros, ``t == 0``, before the first ``adam_step``),
* the slice-boundary hidden state of the latest boundary,
* the layer order and per-layer parameter shapes,
* the format version.

Two on-disk shapes share the same leaf encoding and trailer:

* **Full snapshots** (magic ``SEQECKP1``) -- one self-contained file, the
  format produced for file paths and in-memory buffers.
* **Incremental chains** -- a directory holding one full *basis* segment
  (``seg-00000000.seqd``) followed by delta segments that only encode the
  tensors that changed.  A small ``head`` pointer names the newest
  committed segment, so a crash between the segment write and the pointer
  update always recovers the previous complete state.

Full snapshot wire format (all integers little-endian)::

    magic        8 bytes  b"SEQECKP1"
    version      uint32   format version (3; versions 1 and 2 are migrated)
    header_len   uint64   length of the JSON header in bytes
    header       header_len bytes of UTF-8 JSON (shapes, layer order, flags)
    payload      one leaf per tensor element, each a 1-byte tag
                 ("i" | "f") followed by int64 or float64
    end_magic    12 bytes b"SEQECKP1END"
    leaf_count   uint64   number of encoded leaves
    payload_crc  uint32   CRC-32 of the header bytes followed by the payload

Version 3 extends version 2 with an ``optim`` header object
(``{"t", "m", "v"}`` -- the Adam step count and the two moment shape
lists); its leaves sit between the gradient leaves and the hidden
leaves, so payload order is params, grads, first moments, second
moments, hidden slots.

Delta segments use magic ``SEQDELTA`` / ``SEQDELTAEND`` with the same
framing; their header names the basis file, the segment number, the
Adam step count ``t``, the hidden-slot count (or null) and the changed
tensors as ``{"i", "s"}`` records, where *i* is the flat tensor index
(params, then grads, then first moments, then second moments, then
hidden slots).

Float leaves are emitted as raw IEEE-754 bytes, so values (including the
sign of negative zero) round-trip bit for bit.  Non-finite floats and
integers that do not fit in int64 are refused up front, and payloads that
merely *contain* such bit patterns are rejected on load as well.

Loading validates the whole document before anything is handed back:
truncation, corruption, a version that cannot be migrated, a missing or
unexpected field, or any structural problem rejects the entire checkpoint
with ValueError; nothing is silently filled in.
"""

from __future__ import annotations

import errno
import json
import math
import os
import re
import struct
import tempfile
import threading
import zlib

MAGIC = b"SEQECKP1"
END_MAGIC = b"SEQECKP1END"
DELTA_MAGIC = b"SEQDELTA"
DELTA_END_MAGIC = b"SEQDELTAEND"

FORMAT_VERSION = 3
SUPPORTED_READ_VERSIONS = (1, 2, 3)

_TAG_INT = ord("i")
_TAG_FLOAT = ord("f")

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

_TMP_PREFIX = ".seqckp.tmp-"
_SEG_PREFIX = "seg-"
_SEG_SUFFIX = ".seqd"
_SEG_WIDTH = 10
_HEAD_NAME = "head"
_BASIS_INDEX = 0

_FULL_HEADER_KEYS = frozenset(
    ("v", "params", "grads", "optim", "hidden", "layers", "pending")
)
_LEGACY_FULL_HEADER_KEYS = frozenset(
    ("v", "params", "grads", "hidden", "layers", "pending")
)
_DOCUMENT_KEYS = frozenset(
    ("params", "grads", "optim", "hidden", "layers", "pending")
)
_OPTIM_KEYS = frozenset(("t", "m", "v"))
_LAYER_KEYS = frozenset(("kind", "shapes"))
_DELTA_HEADER_KEYS = frozenset(
    ("v", "b", "n", "hc", "t", "changed", "pending")
)
# Wire records in a delta header carry just index + shape; the leaves live
# in the binary payload. The in-memory build input additionally carries "v".
_CHANGED_KEYS = frozenset(("i", "s"))
_CHANGED_INPUT_KEYS = frozenset(("i", "s", "v"))

# Same-process serialisation for writes into one chain directory.  The
# atomic segment-then-head commit gives the same guarantee across
# processes; the lock merely avoids two threads racing their temp files
# needlessly (and serialises snapshot diffing for one directory).
_chain_locks_guard = threading.Lock()
_chain_locks: dict[str, threading.Lock] = {}


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


def _exact_keys(obj, keys, what):
    if not isinstance(obj, dict) or set(obj) != keys:
        raise CheckpointError(
            f"{what} must be an object with exactly the fields {sorted(keys)}"
        )


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


def _zero_tree(shape):
    """An all-zero (float) tree with *shape*."""
    if shape:
        return [_zero_tree(shape[1:]) for _ in range(shape[0])]
    return 0.0


def _check_step_count(t):
    if isinstance(t, bool) or not isinstance(t, int):
        raise CheckpointError("optimizer step count must be an integer")
    if not _INT64_MIN <= t <= _INT64_MAX:
        raise CheckpointError(
            "optimizer step count is too large to represent (int64)"
        )
    return t


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


def _leaf_code(value):
    """Ordering identity for a leaf: type, value and float sign bit."""
    if isinstance(value, bool):
        raise CheckpointError("checkpoint values must be numbers, not booleans")
    if isinstance(value, int):
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise CheckpointError(
                "checkpoint contains an integer too large to represent (int64)"
            )
        return (0, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError("checkpoint contains a non-finite floating value")
        return (1, struct.pack("<d", value))
    raise CheckpointError(
        f"checkpoint values must be numbers, got {type(value).__name__}"
    )


def _all_zero_leaves(tree):
    if isinstance(tree, list):
        return all(_all_zero_leaves(item) for item in tree)
    return not isinstance(tree, bool) and tree == 0


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
    _exact_keys(entry, {"s", "v"}, "tensor entry")
    shape = entry["s"]
    _check_shape(shape)
    tree = entry["v"]
    if not _shape_matches(tree, shape):
        raise CheckpointError("tensor tree does not match its declared shape")
    _freeze_tree(tree, shape, payload)
    return list(shape)


def _freeze_layer(layer):
    _exact_keys(layer, _LAYER_KEYS, "layer descriptor")
    kind = layer["kind"]
    if not isinstance(kind, str) or not kind:
        raise CheckpointError("layer kind must be a non-empty string")
    shapes = layer["shapes"]
    if not isinstance(shapes, list):
        raise CheckpointError("layer parameter shapes must be a list")
    for shape in shapes:
        _check_shape(shape)
    return {"kind": kind, "shapes": [list(shape) for shape in shapes]}


def _freeze_optim(optim, param_shapes, payload):
    """Freeze the optimizer state; return its header fragment."""
    _exact_keys(optim, _OPTIM_KEYS, "optimizer state")
    t = _check_step_count(optim["t"])
    if t < 0:
        raise CheckpointError("optimizer step count must be non-negative")
    moments = optim["m"]
    velocities = optim["v"]
    if not isinstance(moments, list) or not isinstance(velocities, list):
        raise CheckpointError("optimizer moment lists must be lists")
    if len(moments) != len(param_shapes) or len(velocities) != len(param_shapes):
        raise CheckpointError("optimizer must keep one moment pair per parameter")
    m_shapes = [_freeze_entry(entry, payload) for entry in moments]
    v_shapes = [_freeze_entry(entry, payload) for entry in velocities]
    if m_shapes != param_shapes or v_shapes != param_shapes:
        raise CheckpointError("optimizer moment shapes must match parameter shapes")
    if t == 0:
        # Before the first step every moment must still be its start value.
        if not all(_all_zero_leaves(entry["v"]) for entry in moments) or not all(
            _all_zero_leaves(entry["v"]) for entry in velocities
        ):
            raise CheckpointError(
                "optimizer moments must be all zeros when the step count is 0"
            )
    return {"t": t, "m": m_shapes, "v": v_shapes}


def _freeze(document):
    if not isinstance(document, dict):
        raise CheckpointError("checkpoint document must be an object")
    _exact_keys(document, _DOCUMENT_KEYS, "checkpoint document")
    payload = []

    entries = document.get("params")
    if not isinstance(entries, list):
        raise CheckpointError("checkpoint must record a parameter tensor list")
    param_shapes = [_freeze_entry(entry, payload) for entry in entries]

    grads = document.get("grads")
    if not isinstance(grads, list) or len(grads) != len(entries):
        raise CheckpointError("checkpoint must record one gradient slot per parameter")
    grad_shapes = [_freeze_entry(entry, payload) for entry in grads]
    if grad_shapes != param_shapes:
        raise CheckpointError("parameter and gradient shapes must match")

    optim_header = _freeze_optim(document.get("optim"), param_shapes, payload)

    hidden = document.get("hidden")
    if hidden is None:
        hidden_shapes = None
    else:
        if not isinstance(hidden, list) or not hidden:
            raise CheckpointError("hidden state must be a non-empty list or null")
        hidden_shapes = [_freeze_entry(entry, payload) for entry in hidden]

    layers = document.get("layers")
    if not isinstance(layers, list) or not layers:
        raise CheckpointError("checkpoint must record the layer order")
    header_layers = [_freeze_layer(layer) for layer in layers]
    if sum(len(layer["shapes"]) for layer in header_layers) != len(entries):
        raise CheckpointError("layer parameter counts do not match the parameter list")

    pending = document.get("pending")
    if not isinstance(pending, bool):
        raise CheckpointError("pending flag must be a boolean")

    header = {
        "v": FORMAT_VERSION,
        "params": param_shapes,
        "grads": grad_shapes,
        "optim": optim_header,
        "hidden": hidden_shapes,
        "layers": header_layers,
        "pending": pending,
    }
    _exact_keys(header, _FULL_HEADER_KEYS, "checkpoint header")
    return header, payload


def build_bytes(document):
    """Serialize a full snapshot document to the fixed binary representation."""
    header, payload = _freeze(document)
    return _frame(MAGIC, END_MAGIC, header, payload)


def _frame(magic, end_magic, header, payload):
    header_bytes = json.dumps(
        header, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    body = bytearray()
    body += magic
    body += struct.pack("<I", header["v"])
    body += struct.pack("<Q", len(header_bytes))
    body += header_bytes
    crc = zlib.crc32(header_bytes)
    leaf_count = 0
    for chunk in payload:
        body += chunk
        crc = zlib.crc32(chunk, crc)
        leaf_count += 1
    body += end_magic
    body += struct.pack("<Q", leaf_count)
    body += struct.pack("<I", crc & 0xFFFFFFFF)
    return bytes(body)


# ---------------------------------------------------------------------------
# Parsing: raw bytes -> validated snapshot document
# ---------------------------------------------------------------------------


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
            return value
        if tag == _TAG_FLOAT:
            value = struct.unpack("<d", chunk)[0]
            if not math.isfinite(value):
                raise CheckpointError(
                    "checkpoint payload contains a non-finite floating value"
                )
            return value
        raise CheckpointError(f"unknown leaf tag {tag!r} in payload")

    def tree(self, shape):
        if not shape:
            return self._leaf()
        return [self.tree(shape[1:]) for _ in range(shape[0])]

    def remaining(self):
        return (len(self._payload) - self._pos) // 9


def _read_frame(data, magic, end_magic, what):
    prefix_len = len(magic) + 4 + 8
    if len(data) < prefix_len:
        raise CheckpointError(f"{what} is truncated while reading its header")
    if data[: len(magic)] != magic:
        raise CheckpointError(f"not a {what} (bad magic)")
    (version,) = struct.unpack("<I", data[len(magic) : len(magic) + 4])
    (header_len,) = struct.unpack(
        "<Q", data[len(magic) + 4 : len(magic) + 4 + 8]
    )
    if header_len <= 0:
        raise CheckpointError(f"{what} header length is invalid")

    trailer_pos = data.rfind(end_magic)
    trailer_len = len(end_magic) + 8 + 4
    if trailer_pos < 0 or len(data) - trailer_pos != trailer_len:
        raise CheckpointError(f"{what} trailer is missing or the file is torn")
    header_start = prefix_len
    payload_end = trailer_pos
    if header_len > payload_end - header_start:
        raise CheckpointError(f"{what} header overruns the payload area")
    header_bytes = data[header_start : header_start + header_len]

    (leaf_count, stored_crc) = struct.unpack(
        "<QI", data[trailer_pos + len(end_magic) :]
    )
    actual_crc = zlib.crc32(data[header_start:payload_end])
    if actual_crc != stored_crc:
        raise CheckpointError(f"{what} CRC mismatch: the file is corrupt or torn")

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"{what} header is invalid: {exc}") from exc
    if not isinstance(header, dict):
        raise CheckpointError(f"{what} header must be a JSON object")
    if header.get("v") != version:
        raise CheckpointError(f"{what} header version does not match its frame")

    payload = data[header_start + header_len : payload_end]
    if len(payload) != leaf_count * 9:
        raise CheckpointError(f"{what} payload has the wrong size")
    return version, header, payload, leaf_count


def _parse_shape_list(value, what, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise CheckpointError(f"{what} must be a list of shapes")
    shapes = []
    for shape in value:
        _check_shape(shape)
        shapes.append(list(shape))
    return shapes


def _parse_optim_header(header):
    _exact_keys(header, _OPTIM_KEYS, "optimizer state header")
    t = _check_step_count(header["t"])
    if t < 0:
        raise CheckpointError("optimizer step count must be non-negative")
    m_shapes = _parse_shape_list(header["m"], "optimizer first moments", allow_empty=True)
    v_shapes = _parse_shape_list(header["v"], "optimizer second moments", allow_empty=True)
    return t, m_shapes, v_shapes


def _parse_layer_header(layers):
    if not isinstance(layers, list) or not layers:
        raise CheckpointError("layer order must be a non-empty list")
    layer_shapes = []
    parsed_layers = []
    for index, layer in enumerate(layers):
        _exact_keys(layer, _LAYER_KEYS, f"layer {index} descriptor")
        if not isinstance(layer["kind"], str) or not layer["kind"]:
            raise CheckpointError(f"layer {index} kind must be a non-empty string")
        shapes = _parse_shape_list(
            layer["shapes"], f"layer {index} shapes", allow_empty=True
        )
        layer_shapes.extend(shapes)
        parsed_layers.append(
            {"kind": layer["kind"], "shapes": [list(s) for s in shapes]}
        )
    return parsed_layers, layer_shapes


def _parse_full_header(header, version):
    if not isinstance(header["v"], int) or isinstance(header["v"], bool):
        raise CheckpointError("checkpoint version must be an integer")
    if header["v"] != version:
        raise CheckpointError("checkpoint header version does not match")

    if version >= 3:
        _exact_keys(header, _FULL_HEADER_KEYS, "checkpoint header")
        params = _parse_shape_list(header["params"], "params", allow_empty=True)
        grads = _parse_shape_list(header["grads"], "grads", allow_empty=True)
        if len(params) != len(grads) or params != grads:
            raise CheckpointError("parameter and gradient shape lists must match")
        t, m_shapes, v_shapes = _parse_optim_header(header["optim"])
        if m_shapes != params or v_shapes != params:
            raise CheckpointError(
                "optimizer moment shape lists must match the parameter list"
            )
    else:
        _exact_keys(header, _LEGACY_FULL_HEADER_KEYS, "checkpoint header")
        params = _parse_shape_list(header["params"], "params", allow_empty=True)
        grads = _parse_shape_list(header["grads"], "grads", allow_empty=True)
        if len(params) != len(grads) or params != grads:
            raise CheckpointError("parameter and gradient shape lists must match")
        t = 0

    if header["hidden"] is not None:
        hidden = _parse_shape_list(header["hidden"], "hidden")
    else:
        hidden = None
    parsed_layers, layer_shapes = _parse_layer_header(header["layers"])
    if layer_shapes != params:
        raise CheckpointError(
            "layer order/parameter shapes do not match the parameter tensor list"
        )
    if not isinstance(header["pending"], bool):
        raise CheckpointError("pending flag must be a boolean")
    return params, hidden, t, parsed_layers


def parse_bytes(raw):
    """Validate and decode a full snapshot (v1, v2 or v3); return the document.

    Version-1 and version-2 files are migrated item by item into the
    current document shape: every tensor, shape and descriptor is copied
    and checked individually, and the missing optimizer state starts at
    ``t == 0`` with all-zero moments.  Any failure while migrating
    rejects the whole file -- nothing is partially returned or filled in.
    The document itself carries no source-version field; use
    :func:`load_bytes_with_source` when the originating format version is
    needed.
    """
    document, _source = _decode_full(raw)
    return document


def load_bytes_with_source(source):
    """Like :func:`load_bytes`, also returning the source format version."""
    if isinstance(source, MemoryChain):
        document = load_chain_memory(source)
        return document, FORMAT_VERSION
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if os.path.isdir(path):
            return load_chain(path), FORMAT_VERSION
        with open(path, "rb") as fh:
            raw = fh.read()
        return _decode_full(raw)
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _decode_full(source)
    raise TypeError("load source must be a path or bytes")


def _read_entries(reader, shapes):
    return [{"s": shape, "v": reader.tree(shape)} for shape in shapes]


def _zero_entries(shapes):
    return [{"s": list(shape), "v": _zero_tree(shape)} for shape in shapes]


def _decode_full(raw):
    """Return ``(document, source_version)`` for one full snapshot."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise CheckpointError("checkpoint must be bytes")
    data = bytes(raw)
    version, header, payload, leaf_count = _read_frame(
        data, MAGIC, END_MAGIC, "checkpoint"
    )
    if version not in SUPPORTED_READ_VERSIONS:
        raise CheckpointError(
            f"unsupported checkpoint version {version}; this build reads "
            f"versions {SUPPORTED_READ_VERSIONS}"
        )
    param_shapes, hidden_shapes, optim_t, parsed_layers = _parse_full_header(
        header, version
    )

    moment_groups = 2 if version >= 3 else 0
    total_leaves = sum(_shape_size(shape) for shape in param_shapes) * (
        2 + moment_groups
    )
    if hidden_shapes is not None:
        total_leaves += sum(_shape_size(shape) for shape in hidden_shapes)
    if total_leaves != leaf_count:
        raise CheckpointError("checkpoint leaf count does not match the header")

    reader = _LeafReader(payload)
    params = _read_entries(reader, param_shapes)
    grads = _read_entries(reader, param_shapes)
    if version >= 3:
        moments = _read_entries(reader, param_shapes)
        velocities = _read_entries(reader, param_shapes)
    else:
        moments = _zero_entries(param_shapes)
        velocities = _zero_entries(param_shapes)
    hidden = (
        None
        if hidden_shapes is None
        else _read_entries(reader, hidden_shapes)
    )
    if reader.remaining() != 0:
        raise CheckpointError("checkpoint payload has trailing bytes")

    document = {
        "params": params,
        "grads": grads,
        "optim": {"t": optim_t, "m": moments, "v": velocities},
        "hidden": hidden,
        "layers": parsed_layers,
        "pending": header["pending"],
    }
    if version == 1:
        document = _migrate_v1_document(document)
    elif version == 2:
        document = _migrate_v2_document(document)
    _validate_optim_invariants(document)
    return document, version


def _migrate_v2_document(document):
    """Add the v3 optimizer state to a decoded version-2 document.

    Version 2 shares v3's tensor semantics exactly; the only added field
    is the optimizer state, which starts at its beginning-of-training
    values.
    """
    return _migrate_legacy_document(document, "v2")


def _migrate_v1_document(document):
    """Migrate a decoded version-1 document into the v3 shape, item by item.

    The tensor semantics never changed across versions, so migration is a
    field-by-field re-validation rather than a guess: every tensor, shape
    and descriptor is copied and checked individually.  The optimizer
    state introduced in v3 starts at ``t == 0`` with all-zero moments.  A
    failure on any item raises and the caller rejects the whole file.
    """
    return _migrate_legacy_document(document, "v1")


def _migrate_legacy_document(document, label):
    if not isinstance(document, dict):
        raise CheckpointError(f"{label} checkpoint is not a valid document")
    try:
        params = document["params"]
        grads = document["grads"]
        hidden = document["hidden"]
        layers = document["layers"]
        pending = document["pending"]
    except KeyError as exc:
        raise CheckpointError(
            f"{label} checkpoint is missing field {exc.args[0]!r}; refusing to migrate"
        ) from None
    if not isinstance(params, list) or not isinstance(grads, list):
        raise CheckpointError(f"{label} checkpoint tensor lists are malformed")
    if len(params) != len(grads):
        raise CheckpointError(f"{label} checkpoint param/gradient counts disagree")
    m_params = []
    m_grads = []
    for index, (p_entry, g_entry) in enumerate(zip(params, grads)):
        m_params.append(_migrate_legacy_entry(p_entry, f"parameter {index}", label))
        m_grads.append(_migrate_legacy_entry(g_entry, f"gradient {index}", label))
        if m_params[-1]["s"] != m_grads[-1]["s"]:
            raise CheckpointError(
                f"{label} parameter {index} and its gradient disagree on shape"
            )
    if hidden is None:
        m_hidden = None
    else:
        if not isinstance(hidden, list) or not hidden:
            raise CheckpointError(f"{label} hidden state must be a non-empty list or null")
        m_hidden = [
            _migrate_legacy_entry(entry, f"hidden slot {index}", label)
            for index, entry in enumerate(hidden)
        ]
    if not isinstance(layers, list) or not layers:
        raise CheckpointError(f"{label} checkpoint must record the layer order")
    m_layers = []
    for index, layer in enumerate(layers):
        _exact_keys(layer, _LAYER_KEYS, f"{label} layer {index} descriptor")
        kind = layer["kind"]
        if not isinstance(kind, str) or not kind:
            raise CheckpointError(f"{label} layer {index} kind must be a non-empty string")
        shapes = layer["shapes"]
        if not isinstance(shapes, list):
            raise CheckpointError(f"{label} layer {index} shapes must be a list")
        m_shapes = []
        for shape in shapes:
            _check_shape(shape)
            m_shapes.append(list(shape))
        m_layers.append({"kind": kind, "shapes": m_shapes})
    if not isinstance(pending, bool):
        raise CheckpointError(f"{label} pending flag must be a boolean")
    if sum(len(layer["shapes"]) for layer in m_layers) != len(m_params):
        raise CheckpointError(
            f"{label} layer parameter counts do not match the parameter list"
        )
    return {
        "params": m_params,
        "grads": m_grads,
        # Older files predate the optimizer: start from t == 0, moments 0.
        "optim": {
            "t": 0,
            "m": _zero_entries([p["s"] for p in m_params]),
            "v": _zero_entries([p["s"] for p in m_params]),
        },
        "hidden": m_hidden,
        "layers": m_layers,
        "pending": pending,
    }


def _migrate_legacy_entry(entry, what, label):
    _exact_keys(entry, {"s", "v"}, f"{label} {what} entry")
    shape = entry["s"]
    _check_shape(shape)
    tree = entry["v"]
    if not _shape_matches(tree, shape):
        raise CheckpointError(f"{label} {what} values do not match their declared shape")
    # Re-walk the leaves so oversized integers / non-finite values smuggled
    # into an old payload are refused during migration, not after applying.
    _validate_migrated_tree(tree, shape, what, label)
    return {"s": list(shape), "v": _copy_tree(tree)}


def _validate_migrated_tree(tree, shape, what, label):
    if shape:
        if not isinstance(tree, list) or len(tree) != shape[0]:
            raise CheckpointError(
                f"{label} {what} values do not match their declared shape"
            )
        for item in tree:
            _validate_migrated_tree(item, shape[1:], what, label)
    else:
        if isinstance(tree, list):
            raise CheckpointError(
                f"{label} {what} values do not match their declared shape"
            )
        _check_leaf(tree)


def _validate_optim_invariants(document):
    """Structural optimizer checks shared by native loads and migrations."""
    optim = document.get("optim")
    if not isinstance(optim, dict):
        raise CheckpointError("checkpoint optimizer state is malformed")
    _exact_keys(optim, _OPTIM_KEYS, "optimizer state")
    t = _check_step_count(optim["t"])
    if t < 0:
        raise CheckpointError("optimizer step count must be non-negative")
    params = document["params"]
    moments, velocities = optim["m"], optim["v"]
    if (
        not isinstance(moments, list)
        or not isinstance(velocities, list)
        or len(moments) != len(params)
        or len(velocities) != len(params)
    ):
        raise CheckpointError("optimizer must keep one moment pair per parameter")
    for index, (m_entry, v_entry, p_entry) in enumerate(
        zip(moments, velocities, params)
    ):
        for name, entry in (("first moment", m_entry), ("second moment", v_entry)):
            _exact_keys(entry, {"s", "v"}, f"optimizer {name} {index} entry")
            _check_shape(entry["s"])
            if entry["s"] != p_entry["s"]:
                raise CheckpointError(
                    f"optimizer {name} {index} shape does not match its parameter"
                )
            if not _shape_matches(entry["v"], entry["s"]):
                raise CheckpointError(
                    f"optimizer {name} {index} values do not match its shape"
                )
    if t == 0:
        if not all(_all_zero_leaves(entry["v"]) for entry in moments) or not all(
            _all_zero_leaves(entry["v"]) for entry in velocities
        ):
            raise CheckpointError(
                "optimizer moments must be all zeros when the step count is 0"
            )


def _copy_tree(tree):
    if isinstance(tree, list):
        return [_copy_tree(item) for item in tree]
    return tree


# ---------------------------------------------------------------------------
# Delta segments
# ---------------------------------------------------------------------------


def _segment_name(index):
    return f"{_SEG_PREFIX}{index:0{_SEG_WIDTH}d}{_SEG_SUFFIX}"


def _segment_index(name):
    match = re.fullmatch(
        re.escape(_SEG_PREFIX) + rf"(\d{{{_SEG_WIDTH}}})" + re.escape(_SEG_SUFFIX),
        name,
    )
    if match is None:
        return None
    return int(match.group(1))


def _freeze_delta(document, schema_shapes):
    """Build delta framing from ``{b,n,hc,t,changed,pending}``.

    *schema_shapes* maps the flat tensor index of every tensor the
    post-delta state may contain to its shape: params, then grads, then
    first moments, then second moments, then the (newly introduced, if
    any) hidden slots.
    """
    if not isinstance(document, dict):
        raise CheckpointError("delta document must be an object")
    _exact_keys(document, _DELTA_HEADER_KEYS, "delta header")
    if document["v"] != FORMAT_VERSION:
        raise CheckpointError("delta version must be the current format version")
    basis_name = document["b"]
    if not isinstance(basis_name, str) or basis_name != _segment_name(_BASIS_INDEX):
        raise CheckpointError("delta must name its basis segment")
    number = document["n"]
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise CheckpointError("delta segment number must be a positive int")
    hc = document["hc"]
    if hc is not None and (isinstance(hc, bool) or not isinstance(hc, int) or hc < 0):
        raise CheckpointError("delta hidden count must be a non-negative int or null")
    t = _check_step_count(document["t"])
    if t < 0:
        raise CheckpointError("delta optimizer step count must be non-negative")
    changed = document["changed"]
    if not isinstance(changed, list):
        raise CheckpointError("delta changed list must be a list")
    if not isinstance(document["pending"], bool):
        raise CheckpointError("pending flag must be a boolean")

    payload = []
    seen = set()
    header_entries = []
    previous_index = -1
    for position, entry in enumerate(changed):
        _exact_keys(entry, _CHANGED_INPUT_KEYS, f"delta changed entry {position}")
        index = entry["i"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise CheckpointError(f"delta changed entry {position} has a bad index")
        if index in seen:
            raise CheckpointError(f"delta lists tensor {index} more than once")
        if index <= previous_index:
            raise CheckpointError("delta changed entries must be strictly ordered")
        previous_index = index
        seen.add(index)
        shape = entry["s"]
        _check_shape(shape)
        if index >= len(schema_shapes):
            raise CheckpointError(
                f"delta changed entry {position} names a tensor beyond the model"
            )
        if list(shape) != schema_shapes[index]:
            raise CheckpointError(
                f"delta changed entry {position} shape does not match the chain"
            )
        tree = entry["v"]
        if not _shape_matches(tree, shape):
            raise CheckpointError("delta tensor tree does not match its declared shape")
        _freeze_tree(tree, shape, payload)
        header_entries.append({"i": index, "s": list(shape)})

    header = {
        "v": FORMAT_VERSION,
        "b": basis_name,
        "n": number,
        "hc": hc,
        "t": t,
        "changed": header_entries,
        "pending": document["pending"],
    }
    return _frame(DELTA_MAGIC, DELTA_END_MAGIC, header, payload)


def _parse_delta(raw, segment_index_expected):
    """Validate one delta segment and return ``(number, hc, t, pending, items)``.

    *items* is a list of ``(flat_index, shape, tree)`` already decoded.
    """
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise CheckpointError("delta segment must be bytes")
    version, header, payload, leaf_count = _read_frame(
        bytes(raw), DELTA_MAGIC, DELTA_END_MAGIC, "delta segment"
    )
    if version != FORMAT_VERSION:
        raise CheckpointError(
            f"delta segments must be version {FORMAT_VERSION}, got {version}"
        )
    _exact_keys(header, _DELTA_HEADER_KEYS, "delta header")
    if header["v"] != FORMAT_VERSION:
        raise CheckpointError("delta header version does not match")
    if header["b"] != _segment_name(_BASIS_INDEX):
        raise CheckpointError("delta does not name the chain's basis segment")
    number = header["n"]
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise CheckpointError("delta segment number must be a positive int")
    if number != segment_index_expected:
        raise CheckpointError(
            f"delta segment number {number} does not match its position "
            f"{segment_index_expected} in the chain"
        )
    hc = header["hc"]
    if hc is not None and (isinstance(hc, bool) or not isinstance(hc, int) or hc < 0):
        raise CheckpointError("delta hidden count must be a non-negative int or null")
    t = _check_step_count(header["t"])
    if t < 0:
        raise CheckpointError("delta optimizer step count must be non-negative")
    changed = header["changed"]
    if not isinstance(changed, list):
        raise CheckpointError("delta changed list must be a list")
    if not isinstance(header["pending"], bool):
        raise CheckpointError("pending flag must be a boolean")

    total_leaves = 0
    seen = set()
    descriptors = []
    previous_index = -1
    for position, entry in enumerate(changed):
        _exact_keys(entry, _CHANGED_KEYS, f"delta changed entry {position}")
        index = entry["i"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise CheckpointError(f"delta changed entry {position} has a bad index")
        if index in seen or index <= previous_index:
            raise CheckpointError(
                f"delta changed entry {position} is duplicated or out of order"
            )
        seen.add(index)
        previous_index = index
        shape = entry["s"]
        _check_shape(shape)
        total_leaves += _shape_size(shape)
        descriptors.append((index, list(shape)))
    if total_leaves != leaf_count:
        raise CheckpointError("delta leaf count does not match its changed tensors")

    reader = _LeafReader(payload)
    items = [
        (index, shape, reader.tree(shape)) for index, shape in descriptors
    ]
    if reader.remaining() != 0:
        raise CheckpointError("delta payload has trailing bytes")
    return number, hc, t, header["pending"], items


# ---------------------------------------------------------------------------
# Chain assembly
# ---------------------------------------------------------------------------


def _tensors_from_document(document):
    """Return ``{flat_index: (shape, tree)}`` for one full state document.

    Index order: params, then grads, then first moments, then second
    moments, then hidden slots.
    """
    tensors = {}
    offset = 0
    for group in ("params", "grads"):
        for entry in document[group]:
            tensors[offset] = (entry["s"], entry["v"])
            offset += 1
    for entry in document["optim"]["m"]:
        tensors[offset] = (entry["s"], entry["v"])
        offset += 1
    for entry in document["optim"]["v"]:
        tensors[offset] = (entry["s"], entry["v"])
        offset += 1
    hidden = document["hidden"]
    if hidden is not None:
        for entry in hidden:
            tensors[offset] = (entry["s"], entry["v"])
            offset += 1
    return tensors


def _trees_differ(a, b, shape):
    if shape:
        if not isinstance(a, list) or not isinstance(b, list):
            return True
        if len(a) != shape[0] or len(b) != shape[0]:
            return True
        return any(
            _trees_differ(x, y, shape[1:]) for x, y in zip(a, b)
        )
    if isinstance(a, list) or isinstance(b, list):
        return True
    return _leaf_code(a) != _leaf_code(b)


def _assemble(basis_document, applied):
    """Build a full document from basis tensors plus per-index replacements.

    *applied* maps a flat tensor index (params, grads, first moments,
    second moments, hidden) to ``(shape, tree)``.  Hidden indices only
    exist once a delta introduced hidden state; a complete set must be
    present at that point.
    """
    param_count = len(basis_document["params"])
    layers = basis_document["layers"]
    hidden_count = (
        len(basis_document["hidden"]) if basis_document["hidden"] is not None else None
    )

    # The latest delta in the chain decides whether hidden state exists.
    final_hidden_count = applied.get("_hc", hidden_count)
    tensors = applied["tensors"]

    params = []
    grads = []
    moments = []
    velocities = []
    hidden = []
    total = 4 * param_count + (final_hidden_count or 0)
    for index in range(total):
        if index not in tensors:
            raise CheckpointError(
                f"chain is missing tensor {index}: the delta chain is incomplete"
            )
        shape, tree = tensors[index]
        entry = {"s": list(shape), "v": tree}
        if index < param_count:
            params.append(entry)
        elif index < 2 * param_count:
            grads.append(entry)
        elif index < 3 * param_count:
            moments.append(entry)
        elif index < 4 * param_count:
            velocities.append(entry)
        else:
            hidden.append(entry)

    if (
        len(params) != param_count
        or len(grads) != param_count
        or len(moments) != param_count
        or len(velocities) != param_count
    ):
        raise CheckpointError("chain did not preserve the parameter/moment count")
    document = {
        "params": params,
        "grads": grads,
        "optim": {"t": applied["t"], "m": moments, "v": velocities},
        "hidden": hidden if final_hidden_count is not None else None,
        "layers": layers,
        "pending": applied["pending"],
    }
    _validate_optim_invariants(document)
    return document


# ---------------------------------------------------------------------------
# Atomic file IO
# ---------------------------------------------------------------------------


def _atomic_write(directory, final_name, raw):
    path = os.path.join(directory, final_name)
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
    return path


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


def _chain_lock(directory):
    key = os.path.abspath(directory)
    with _chain_locks_guard:
        lock = _chain_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _chain_locks[key] = lock
    return lock


# ---------------------------------------------------------------------------
# Chain storage seam: one append/read protocol, two backends.
#
# * ``_DirectoryChainStore`` -- the production store; each object is written
#   via a temp file + ``os.replace`` (crash-atomic) inside a per-directory
#   lock.
# * ``MemoryChain`` -- a purely in-memory store following the exact same
#   segment-then-head commit ordering, used by the built-in self-checks so
#   the incremental path is exercised without touching the filesystem.
# ---------------------------------------------------------------------------


class _ChainStoreBase:
    def read_head(self):
        raise NotImplementedError

    def write_head(self, raw):
        raise NotImplementedError

    def read_segment(self, name):
        raise NotImplementedError

    def write_segment(self, name, raw):
        raise NotImplementedError


class MemoryChain(_ChainStoreBase):
    """An in-memory incremental checkpoint chain.

    Behaves exactly like a chain directory but keeps its segments in a dict
    and commits the segment mapping before the head pointer. Pass an
    instance to ``Sequential.save`` / ``Sequential.load`` just like a
    directory path. Safe for concurrent use by multiple threads.
    """

    def __init__(self):
        self._objects: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def read_head(self):
        with self._lock:
            try:
                return self._objects[_HEAD_NAME]
            except KeyError:
                raise FileNotFoundError(_HEAD_NAME) from None

    def write_head(self, raw):
        with self._lock:
            # Commit order: the named segment must already be present when
            # the head can point at it.
            self._objects[_HEAD_NAME] = bytes(raw)

    def read_segment(self, name):
        with self._lock:
            try:
                return self._objects[name]
            except KeyError:
                raise FileNotFoundError(name) from None

    def write_segment(self, name, raw):
        with self._lock:
            self._objects[name] = bytes(raw)

    def __len__(self):
        with self._lock:
            return sum(
                1 for name in self._objects
                if _segment_index(name) is not None
            )


class _DirectoryChainStore(_ChainStoreBase):
    def __init__(self, directory):
        self._directory = directory

    def read_head(self):
        with open(os.path.join(self._directory, _HEAD_NAME), "rb") as fh:
            return fh.read()

    def write_head(self, raw):
        _atomic_write(self._directory, _HEAD_NAME, raw)

    def read_segment(self, name):
        with open(os.path.join(self._directory, name), "rb") as fh:
            return fh.read()

    def write_segment(self, name, raw):
        _atomic_write(self._directory, name, raw)


# ---------------------------------------------------------------------------
# Chain public API
# ---------------------------------------------------------------------------


def save_chain(document, directory):
    """Append *document* to the incremental chain living in *directory*.

    The first save writes the full basis segment; later saves append a
    delta segment carrying only the tensors that changed (compared by
    encoded leaf identity, so float sign bits such as -0.0 count).  The
    segment file and then the ``head`` pointer are atomically committed,
    so a crash -- or two saves racing to the same directory -- always
    leaves exactly one complete chain state reachable from ``head``.
    """
    if not isinstance(directory, (str, os.PathLike)):
        raise TypeError("chain directory must be a path")
    directory = os.fspath(directory)
    if not os.path.isdir(directory):
        raise OSError(
            errno.ENOENT,
            f"incremental checkpoint directory does not exist: {directory!r}",
        )
    with _chain_lock(directory):
        _save_chain_store(document, _DirectoryChainStore(directory))
    return None


def save_chain_memory(document, store):
    """Append *document* to an in-memory chain (used by the self-checks)."""
    if not isinstance(store, MemoryChain):
        raise TypeError("store must be a MemoryChain")
    _save_chain_store(document, store)
    return None


def _save_chain_store(document, store):
    head_index = _read_head_optional(store)
    if head_index is None:
        store.write_segment(_segment_name(_BASIS_INDEX), build_bytes(document))
        _commit_head(store, _BASIS_INDEX)
        return
    previous = _load_chain_store(store, head_index)
    next_index = head_index + 1
    delta_bytes = _build_delta_between(previous, document, next_index)
    store.write_segment(_segment_name(next_index), delta_bytes)
    _commit_head(store, next_index)


def _commit_head(store, index):
    store.write_head(str(index).encode("ascii"))


def _read_head_optional(store):
    try:
        raw = store.read_head()
    except FileNotFoundError:
        return None
    text = raw.decode("ascii")
    if not re.fullmatch(r"\d+", text):
        raise CheckpointError("chain head pointer is corrupt")
    index = int(text)
    if _segment_index(_segment_name(index)) is None or index < 0:
        raise CheckpointError("chain head pointer names an invalid segment")
    return index


def _build_delta_between(previous, current, next_index):
    """Encode the tensors that differ between two full states as one delta."""
    param_count = len(current["params"])
    if len(previous["params"]) != param_count:
        raise CheckpointError(
            "an incremental chain cannot change the parameter count; save a "
            "full snapshot for the new model"
        )
    if len(previous["grads"]) != param_count or len(current["grads"]) != param_count:
        raise CheckpointError("gradient count changed inside the incremental chain")

    prev_optim = previous["optim"]
    cur_optim = current["optim"]
    if (
        len(prev_optim["m"]) != param_count
        or len(cur_optim["m"]) != param_count
        or len(prev_optim["v"]) != param_count
        or len(cur_optim["v"]) != param_count
    ):
        raise CheckpointError("optimizer moment count changed inside the chain")
    cur_t = _check_step_count(cur_optim["t"])
    if cur_t < 0:
        raise CheckpointError("optimizer step count must be non-negative")

    prev_layers = [
        (layer["kind"], [list(s) for s in layer["shapes"]])
        for layer in previous["layers"]
    ]
    cur_layers = [
        (layer["kind"], [list(s) for s in layer["shapes"]])
        for layer in current["layers"]
    ]
    if prev_layers != cur_layers:
        raise CheckpointError(
            "an incremental chain cannot change the layer order; save a full "
            "snapshot for the new model"
        )

    prev_tensors = _tensors_from_document(previous)
    cur_tensors = _tensors_from_document(current)
    prev_hc = len(previous["hidden"]) if previous["hidden"] is not None else None
    cur_hc = len(current["hidden"]) if current["hidden"] is not None else None

    if prev_hc is not None and cur_hc is not None and prev_hc != cur_hc:
        raise CheckpointError("hidden slot count changed inside the incremental chain")
    if prev_hc is not None and cur_hc is None:
        raise CheckpointError(
            "a delta cannot drop hidden state once the chain has fixed it"
        )
    if cur_hc is not None and cur_hc != len(current["layers"]):
        raise CheckpointError(
            "hidden state must provide exactly one slot per layer"
        )

    # Every tensor the post-state may name; needed both for the diff and
    # for shape validation while freezing.
    hidden_shapes_introduced = cur_hc is not None and prev_hc is None
    hidden_base = 4 * param_count
    schema = {}
    for index in range(hidden_base):
        schema[index] = cur_tensors[index][0]
    if cur_hc is not None:
        for slot in range(cur_hc):
            index = hidden_base + slot
            schema[index] = cur_tensors[index][0]

    changed_entries = []
    ordered = sorted(cur_tensors)
    for index in ordered:
        shape, tree = cur_tensors[index]
        prev = prev_tensors.get(index)
        if prev is not None and list(prev[0]) != list(shape):
            raise CheckpointError(
                f"tensor {index} changed shape inside the incremental chain; "
                "save a full snapshot for the new model"
            )
        if index >= hidden_base and hidden_shapes_introduced:
            changed_entries.append({"i": index, "s": shape, "v": tree})
            continue
        if prev is None or _trees_differ(prev[1], tree, shape):
            changed_entries.append({"i": index, "s": shape, "v": tree})

    delta_doc = {
        "v": FORMAT_VERSION,
        "b": _segment_name(_BASIS_INDEX),
        "n": next_index,
        "hc": cur_hc,
        "t": cur_t,
        "changed": changed_entries,
        "pending": current["pending"],
    }
    schema_shapes = [schema.get(i) for i in range(hidden_base + (cur_hc or 0))]
    return _freeze_delta(delta_doc, schema_shapes)


def load_chain(directory, up_to=None):
    """Reassemble a full state document from an incremental chain directory.

    Every segment from the basis through the ``head`` pointer (or segment
    *up_to*) is validated before the document is returned.  A missing
    directory raises FileNotFoundError; a missing referenced segment file
    raises FileNotFoundError as well; any truncation, corruption, missing
    field or structural inconsistency rejects the whole chain with
    ValueError.
    """
    if not isinstance(directory, (str, os.PathLike)):
        raise TypeError("chain directory must be a path")
    directory = os.fspath(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"incremental checkpoint directory not found: {directory!r}"
        )
    with _chain_lock(directory):
        return _load_chain_root(_DirectoryChainStore(directory), up_to)


def load_chain_memory(store, up_to=None):
    """Reassemble a full state document from an in-memory chain."""
    if not isinstance(store, MemoryChain):
        raise TypeError("store must be a MemoryChain")
    return _load_chain_root(store, up_to)


def _load_chain_root(store, up_to):
    head = _read_head_optional(store)
    if head is None:
        raise CheckpointError(
            "chain has no head pointer (no basis segment committed)"
        )
    if up_to is not None:
        if isinstance(up_to, bool) or not isinstance(up_to, int) or up_to < 0:
            raise CheckpointError("up_to must be a non-negative segment index")
        if up_to > head:
            raise CheckpointError(
                f"up_to segment {up_to} is beyond the chain head {head}"
            )
        head = up_to
    return _load_chain_store(store, head)


def _load_chain_store(store, head):
    basis_name = _segment_name(_BASIS_INDEX)
    basis_raw = store.read_segment(basis_name)
    basis = parse_bytes(basis_raw)
    param_count = len(basis["params"])
    basis_hc = len(basis["hidden"]) if basis["hidden"] is not None else None

    tensors = _tensors_from_document(basis)
    current_hc = basis_hc
    current_t = basis["optim"]["t"]
    pending = basis["pending"]
    hidden_base = 4 * param_count
    for index in range(1, head + 1):
        name = _segment_name(index)
        raw = store.read_segment(name)
        _number, delta_hc, delta_t, delta_pending, items = _parse_delta(raw, index)

        # Hidden-state transitions: absent -> present exactly once with a
        # full set of slots; afterwards the count is fixed.
        introducing = delta_hc is not None and current_hc is None
        if current_hc is not None and delta_hc != current_hc:
            raise CheckpointError(
                f"delta {index} changes the hidden slot count; chain rejected"
            )
        if not introducing and delta_hc is None and current_hc is None:
            expected_max = hidden_base - 1
        elif delta_hc is None:
            raise CheckpointError(
                f"delta {index} drops hidden state the chain already fixed"
            )
        else:
            expected_max = hidden_base + delta_hc - 1

        replaced_hidden = set()
        for flat_index, shape, tree in items:
            if flat_index > expected_max:
                raise CheckpointError(
                    f"delta {index} names tensor {flat_index} beyond its state"
                )
            if flat_index < hidden_base:
                if flat_index < param_count:
                    expected_shape = basis["params"][flat_index]["s"]
                elif flat_index < 2 * param_count:
                    expected_shape = basis["grads"][flat_index - param_count]["s"]
                elif flat_index < 3 * param_count:
                    expected_shape = basis["optim"]["m"][
                        flat_index - 2 * param_count
                    ]["s"]
                else:
                    expected_shape = basis["optim"]["v"][
                        flat_index - 3 * param_count
                    ]["s"]
                if list(shape) != list(expected_shape):
                    raise CheckpointError(
                        f"delta {index} tensor {flat_index} shape disagrees with basis"
                    )
            else:
                slot = flat_index - hidden_base
                if slot >= delta_hc:
                    raise CheckpointError(
                        f"delta {index} names hidden slot {slot} beyond its count"
                    )
                prior = tensors.get(flat_index)
                if prior is not None and list(prior[0]) != list(shape):
                    raise CheckpointError(
                        f"delta {index} hidden slot {slot} shape disagrees with "
                        "the slot introduced earlier in the chain"
                    )
                replaced_hidden.add(slot)
            tensors[flat_index] = (list(shape), tree)

        if introducing:
            layer_count = len(basis["layers"])
            if delta_hc != layer_count:
                raise CheckpointError(
                    f"delta {index} introduces {delta_hc} hidden slots for "
                    f"{layer_count} layers"
                )
            if replaced_hidden != set(range(delta_hc)):
                raise CheckpointError(
                    f"delta {index} introduces hidden state incompletely"
                )
        current_hc = delta_hc
        current_t = delta_t
        pending = delta_pending

    applied = {
        "tensors": tensors,
        "pending": pending,
        "_hc": current_hc,
        "t": current_t,
    }
    return _assemble(basis, applied)


# ---------------------------------------------------------------------------
# Full / chain dispatch and file / memory IO
# ---------------------------------------------------------------------------


def save_bytes(document, target):
    """Write *document* to a path, a chain directory, a MemoryChain or bytearray.

    File writes go through a temporary file in the destination directory
    and are atomically renamed into place, overwriting any existing file
    without leaving a backup.  An existing directory -- or a
    :class:`MemoryChain` -- is treated as an incremental checkpoint chain
    (see :func:`save_chain`).
    """
    if isinstance(target, MemoryChain):
        _save_chain_store(document, target)
        return None
    if isinstance(target, bytearray):
        raw = build_bytes(document)
        target.clear()
        target.extend(raw)
        return None
    if isinstance(target, (str, os.PathLike)):
        path = os.fspath(target)
        if os.path.isdir(path):
            return save_chain(document, path)
        directory = os.path.dirname(os.path.abspath(path))
        raw = build_bytes(document)
        _atomic_write(directory, os.path.basename(path), raw)
        return None
    raise TypeError(
        "save target must be a path, a chain directory, a MemoryChain or bytearray"
    )


def load_bytes(source):
    """Read and validate a full snapshot from a path, chain, or bytes.

    A directory path (or a :class:`MemoryChain`) loads the reassembled
    head of the incremental chain stored there.
    """
    if isinstance(source, MemoryChain):
        return load_chain_memory(source)
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if os.path.isdir(path):
            return load_chain(path)
        with open(path, "rb") as fh:
            raw = fh.read()
        return parse_bytes(raw)
    if isinstance(source, (bytes, bytearray, memoryview)):
        return parse_bytes(source)
    raise TypeError("load source must be a path or bytes")
