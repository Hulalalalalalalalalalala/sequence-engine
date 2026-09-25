"""Checkpoint container: versioned binary snapshots for Sequential state.

A full snapshot fixes, in one self-describing document:

* every parameter tensor,
* every accumulated gradient (zeros when a parameter has none yet),
* the optimizer state (first and second moment per parameter plus the
  step count; zeros and ``t = 0`` when ``adam_step`` has never run),
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
  update always recovers the previous complete state.  A chain can be
  **compacted** in place: the basis and a prefix of the deltas are folded
  into one new basis segment and the remaining deltas are renumbered.
  Compaction stages every new segment, records the new head in a marker
  and only then replaces the live segments, so a crash mid-compaction is
  rolled forward on the next open and the directory always holds exactly
  one complete chain -- the old head's or the new head's.

Full snapshot wire format (all integers little-endian)::

    magic        8 bytes  b"SEQECKP1"
    version      uint32   format version (3; versions 1 and 2 are read and
                          migrated, missing optimizer state starts at t=0
                          with zero moments)
    header_len   uint64   length of the JSON header in bytes
    header       header_len bytes of UTF-8 JSON (shapes, layer order, flags)
    payload      one leaf per tensor element, each a 1-byte tag
                 ("i" | "f") followed by int64 or float64
    end_magic    12 bytes b"SEQECKP1END"
    leaf_count   uint64   number of encoded leaves
    payload_crc  uint32   CRC-32 of the header bytes followed by the payload

Delta segments use magic ``SEQDELTA`` / ``SEQDELTAEND`` with the same
framing; their header names the basis file, the segment number, the
hidden-slot count (or null) and the changed tensors as ``{"i", "s"}``
records, where *i* is the flat tensor index (params, then grads, then the
optimizer first moments, second moments and step count, then hidden
slots).  Delta segments written by the previous format version (2), whose
flat space ends after the hidden slots, are read and migrated as well.

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

try:
    import fcntl
except ImportError:  # non-POSIX platforms: in-process locking only
    fcntl = None

MAGIC = b"SEQECKP1"
END_MAGIC = b"SEQECKP1END"
DELTA_MAGIC = b"SEQDELTA"
DELTA_END_MAGIC = b"SEQDELTAEND"

FORMAT_VERSION = 3
SUPPORTED_READ_VERSIONS = (1, 2, 3)
# Delta segments exist since version 2; both v2 and v3 segments are read.
SUPPORTED_DELTA_VERSIONS = (2, 3)

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
# Compaction commit protocol: new segments are staged under this prefix,
# then a marker file records the new head, then the staged files are
# copied over the live segment names and the head is advanced.  The
# marker is the last thing removed, so any crash is rolled forward by
# re-running the recovery step.
_STAGED_PREFIX = ".seqc-"
_COMPACT_MARKER = ".seqcompact"

_BASE_HEADER_KEYS = frozenset(("v", "params", "grads", "hidden", "layers", "pending"))
_FULL_HEADER_KEYS = frozenset(_BASE_HEADER_KEYS | {"optim"})
_OPTIM_KEYS = frozenset(("t", "m", "v"))
_DOCUMENT_KEYS = frozenset(("params", "grads", "hidden", "layers", "pending"))
_LAYER_KEYS = frozenset(("kind", "shapes"))
_DELTA_HEADER_KEYS = frozenset(("v", "b", "n", "hc", "changed", "pending"))
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


def _resolve_optim(document, param_shapes):
    """Return ``(t, m_entries, v_entries)`` for a document's optimizer state.

    A document without an ``optim`` field (or with ``None``) describes a
    model whose optimizer never stepped: ``t = 0`` and zero moments shaped
    like the parameters.
    """
    optim = document.get("optim")
    if optim is None:
        return 0, [
            {"s": list(shape), "v": _zeros_tree(shape)} for shape in param_shapes
        ], [
            {"s": list(shape), "v": _zeros_tree(shape)} for shape in param_shapes
        ]
    _exact_keys(optim, _OPTIM_KEYS, "checkpoint optimizer state")
    t = _check_step_count(optim["t"])
    m_entries = optim["m"]
    v_entries = optim["v"]
    if (
        not isinstance(m_entries, list)
        or not isinstance(v_entries, list)
        or len(m_entries) != len(param_shapes)
        or len(v_entries) != len(param_shapes)
    ):
        raise CheckpointError(
            "optimizer state must hold one first/second moment per parameter"
        )
    return t, m_entries, v_entries


def _freeze(document):
    if not isinstance(document, dict):
        raise CheckpointError("checkpoint document must be an object")
    missing = _DOCUMENT_KEYS - document.keys()
    extra = document.keys() - (_DOCUMENT_KEYS | {"optim"})
    if missing or extra:
        raise CheckpointError(
            f"checkpoint document must be an object with the fields "
            f"{sorted(_DOCUMENT_KEYS)} (plus an optional 'optim')"
        )
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

    hidden = document.get("hidden")
    if hidden is None:
        hidden_shapes = None
    else:
        if not isinstance(hidden, list) or not hidden:
            raise CheckpointError("hidden state must be a non-empty list or null")
        hidden_shapes = [_freeze_entry(entry, payload) for entry in hidden]

    optim_t, optim_m, optim_v = _resolve_optim(document, param_shapes)
    m_shapes = [_freeze_entry(entry, payload) for entry in optim_m]
    v_shapes = [_freeze_entry(entry, payload) for entry in optim_v]
    if m_shapes != param_shapes or v_shapes != param_shapes:
        raise CheckpointError("optimizer moment shapes must match the parameters")

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
        "hidden": hidden_shapes,
        "optim": {"t": optim_t, "m": m_shapes, "v": v_shapes},
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


def _parse_optim_header(optim, param_shapes):
    _exact_keys(optim, _OPTIM_KEYS, "checkpoint optimizer state")
    t = _check_step_count(optim["t"])
    m_shapes = _parse_shape_list(optim["m"], "optimizer first moments", allow_empty=True)
    v_shapes = _parse_shape_list(optim["v"], "optimizer second moments", allow_empty=True)
    if m_shapes != param_shapes or v_shapes != param_shapes:
        raise CheckpointError("optimizer moment shapes must match the parameters")
    return t


def _parse_full_header(header, version):
    expected = _FULL_HEADER_KEYS if version >= 3 else _BASE_HEADER_KEYS
    _exact_keys(header, expected, "checkpoint header")
    if not isinstance(header["v"], int) or isinstance(header["v"], bool):
        raise CheckpointError("checkpoint version must be an integer")
    if header["v"] != version:
        raise CheckpointError("checkpoint header version does not match")
    params = _parse_shape_list(header["params"], "params", allow_empty=True)
    grads = _parse_shape_list(header["grads"], "grads", allow_empty=True)
    if len(params) != len(grads) or params != grads:
        raise CheckpointError("parameter and gradient shape lists must match")
    if header["hidden"] is not None:
        hidden = _parse_shape_list(header["hidden"], "hidden")
    else:
        hidden = None
    optim_t = None
    if version >= 3:
        optim_t = _parse_optim_header(header["optim"], params)
    layers = header["layers"]
    if not isinstance(layers, list) or not layers:
        raise CheckpointError("layer order must be a non-empty list")
    layer_shapes = []
    for index, layer in enumerate(layers):
        _exact_keys(layer, _LAYER_KEYS, f"layer {index} descriptor")
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
    return params, hidden, optim_t


def parse_bytes(raw):
    """Validate and decode a full snapshot (v1, v2 or v3); return the document.

    A version-1 or version-2 file is migrated item by item into the current
    document shape (a missing optimizer state starts at ``t = 0`` with zero
    moments).  Any failure while migrating rejects the whole file -- nothing
    is partially returned or filled in.  The document itself carries no
    source-version field; use :func:`load_bytes_with_source` when the
    originating format version is needed.
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
    param_shapes, hidden_shapes, optim_t = _parse_full_header(header, version)

    param_leaves = sum(_shape_size(shape) for shape in param_shapes)
    total_leaves = param_leaves * 2
    if hidden_shapes is not None:
        total_leaves += sum(_shape_size(shape) for shape in hidden_shapes)
    if optim_t is not None:
        total_leaves += param_leaves * 2
    if total_leaves != leaf_count:
        raise CheckpointError("checkpoint leaf count does not match the header")

    reader = _LeafReader(payload)
    params = [{"s": shape, "v": reader.tree(shape)} for shape in param_shapes]
    grads = [{"s": shape, "v": reader.tree(shape)} for shape in param_shapes]
    hidden = (
        None
        if hidden_shapes is None
        else [{"s": shape, "v": reader.tree(shape)} for shape in hidden_shapes]
    )
    if optim_t is not None:
        optim = {
            "t": optim_t,
            "m": [{"s": shape, "v": reader.tree(shape)} for shape in param_shapes],
            "v": [{"s": shape, "v": reader.tree(shape)} for shape in param_shapes],
        }
    else:
        optim = None
    if reader.remaining() != 0:
        raise CheckpointError("checkpoint payload has trailing bytes")

    document = {
        "params": params,
        "grads": grads,
        "hidden": hidden,
        "optim": optim,
        "layers": [
            {"kind": layer["kind"], "shapes": [list(s) for s in layer["shapes"]]}
            for layer in header["layers"]
        ],
        "pending": header["pending"],
    }
    if version == 1:
        document = _migrate_v1_document(document)
    if version < FORMAT_VERSION:
        document = _migrate_add_optim(document)
    return document, version


def _migrate_v1_document(document):
    """Migrate a decoded version-1 document item by item.

    The tensor semantics are shared across versions, so migration is a
    field-by-field re-validation rather than a guess: every tensor, shape
    and descriptor is copied and checked individually.  A failure on any
    item raises and the caller rejects the whole file.  The optimizer
    state, absent before version 3, is filled in separately by
    :func:`_migrate_add_optim`.
    """
    if not isinstance(document, dict):
        raise CheckpointError("v1 checkpoint is not a valid document")
    migrated = {}
    try:
        params = document["params"]
        grads = document["grads"]
        hidden = document["hidden"]
        layers = document["layers"]
        pending = document["pending"]
    except KeyError as exc:
        raise CheckpointError(
            f"v1 checkpoint is missing field {exc.args[0]!r}; refusing to migrate"
        ) from None
    if not isinstance(params, list) or not isinstance(grads, list):
        raise CheckpointError("v1 checkpoint tensor lists are malformed")
    if len(params) != len(grads):
        raise CheckpointError("v1 checkpoint param/gradient counts disagree")
    m_params = []
    m_grads = []
    for index, (p_entry, g_entry) in enumerate(zip(params, grads)):
        m_params.append(_migrate_v1_entry(p_entry, f"parameter {index}"))
        m_grads.append(_migrate_v1_entry(g_entry, f"gradient {index}"))
        if m_params[-1]["s"] != m_grads[-1]["s"]:
            raise CheckpointError(
                f"v1 parameter {index} and its gradient disagree on shape"
            )
    migrated["params"] = m_params
    migrated["grads"] = m_grads
    if hidden is None:
        migrated["hidden"] = None
    else:
        if not isinstance(hidden, list) or not hidden:
            raise CheckpointError("v1 hidden state must be a non-empty list or null")
        migrated["hidden"] = [
            _migrate_v1_entry(entry, f"hidden slot {index}")
            for index, entry in enumerate(hidden)
        ]
    if not isinstance(layers, list) or not layers:
        raise CheckpointError("v1 checkpoint must record the layer order")
    m_layers = []
    for index, layer in enumerate(layers):
        _exact_keys(layer, _LAYER_KEYS, f"v1 layer {index} descriptor")
        kind = layer["kind"]
        if not isinstance(kind, str) or not kind:
            raise CheckpointError(f"v1 layer {index} kind must be a non-empty string")
        shapes = layer["shapes"]
        if not isinstance(shapes, list):
            raise CheckpointError(f"v1 layer {index} shapes must be a list")
        m_shapes = []
        for shape in shapes:
            _check_shape(shape)
            m_shapes.append(list(shape))
        m_layers.append({"kind": kind, "shapes": m_shapes})
    migrated["layers"] = m_layers
    if not isinstance(pending, bool):
        raise CheckpointError("v1 pending flag must be a boolean")
    migrated["pending"] = pending
    if sum(len(layer["shapes"]) for layer in m_layers) != len(m_params):
        raise CheckpointError("v1 layer parameter counts do not match the parameter list")
    return migrated


def _migrate_add_optim(document):
    """Bring a pre-v3 document up to the current shape, item by item.

    Versions 1 and 2 predate the optimizer state, so migration fixes the
    starting values exactly: step count ``t = 0`` and one zero first/second
    moment per parameter tensor.  The re-saved state is native version 3.
    """
    if not isinstance(document, dict):
        raise CheckpointError("checkpoint is not a valid document")
    if document.get("optim") is not None:
        return document
    params = document.get("params")
    if not isinstance(params, list):
        raise CheckpointError("checkpoint parameter list is malformed")
    m_entries = []
    v_entries = []
    for index, entry in enumerate(params):
        _exact_keys(entry, {"s", "v"}, f"parameter {index} entry")
        shape = entry["s"]
        _check_shape(shape)
        m_entries.append({"s": list(shape), "v": _zeros_tree(shape)})
        v_entries.append({"s": list(shape), "v": _zeros_tree(shape)})
    document["optim"] = {"t": 0, "m": m_entries, "v": v_entries}
    return document


def _migrate_v1_entry(entry, what):
    _exact_keys(entry, {"s", "v"}, f"v1 {what} entry")
    shape = entry["s"]
    _check_shape(shape)
    tree = entry["v"]
    if not _shape_matches(tree, shape):
        raise CheckpointError(f"v1 {what} values do not match their declared shape")
    # Re-walk the leaves so oversized integers / non-finite values smuggled
    # into a v1 payload are refused during migration, not after applying.
    _validate_migrated_tree(tree, shape, what)
    return {"s": list(shape), "v": _copy_tree(tree)}


def _validate_migrated_tree(tree, shape, what):
    if shape:
        if not isinstance(tree, list) or len(tree) != shape[0]:
            raise CheckpointError(f"v1 {what} values do not match their declared shape")
        for item in tree:
            _validate_migrated_tree(item, shape[1:], what)
    else:
        if isinstance(tree, list):
            raise CheckpointError(f"v1 {what} values do not match their declared shape")
        _check_leaf(tree)


def _copy_tree(tree):
    if isinstance(tree, list):
        return [_copy_tree(item) for item in tree]
    return tree


def _zeros_tree(shape):
    if not shape:
        return 0.0
    return [_zeros_tree(shape[1:]) for _ in range(shape[0])]


def _check_step_count(value, what="optimizer step count"):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointError(f"{what} must be a non-negative integer")
    if value > _INT64_MAX:
        raise CheckpointError(f"{what} is too large to represent (int64)")
    return value


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


def _staged_name(index):
    return _STAGED_PREFIX + _segment_name(index)


def _freeze_delta(document, schema_shapes):
    """Build delta framing from ``{b,n,hc,changed,pending}``.

    *schema_shapes* maps the flat tensor index of every tensor the
    post-delta state may contain to its shape: params then grads then the
    (newly introduced, if any) hidden slots.
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
        "changed": header_entries,
        "pending": document["pending"],
    }
    return _frame(DELTA_MAGIC, DELTA_END_MAGIC, header, payload)


def _parse_delta(raw, segment_index_expected):
    """Validate one delta segment and return ``(number, hc, pending, items)``.

    *items* is a list of ``(flat_index, shape, tree)`` already decoded.
    """
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise CheckpointError("delta segment must be bytes")
    version, header, payload, leaf_count = _read_frame(
        bytes(raw), DELTA_MAGIC, DELTA_END_MAGIC, "delta segment"
    )
    if version not in SUPPORTED_DELTA_VERSIONS:
        raise CheckpointError(
            f"delta segments must be version {SUPPORTED_DELTA_VERSIONS}, "
            f"got {version}"
        )
    _exact_keys(header, _DELTA_HEADER_KEYS, "delta header")
    if header["v"] != version:
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
    return number, hc, header["pending"], items


# ---------------------------------------------------------------------------
# Chain assembly
# ---------------------------------------------------------------------------


def _tensors_from_document(document):
    """Return ``{flat_index: (shape, tree)}`` for one full state document.

    The flat index space is: parameters, gradients, optimizer first
    moments, optimizer second moments, the optimizer step count (a scalar),
    then the hidden slots.  Hidden slots come last so introducing hidden
    state mid-chain only ever appends new indices.
    """
    tensors = {}
    offset = 0
    for entry in document["params"]:
        tensors[offset] = (entry["s"], entry["v"])
        offset += 1
    for entry in document["grads"]:
        tensors[offset] = (entry["s"], entry["v"])
        offset += 1
    optim = document["optim"]
    for entry in optim["m"]:
        tensors[offset] = (entry["s"], entry["v"])
        offset += 1
    for entry in optim["v"]:
        tensors[offset] = (entry["s"], entry["v"])
        offset += 1
    tensors[offset] = ([], optim["t"])
    offset += 1
    hidden = document["hidden"]
    if hidden is not None:
        for entry in hidden:
            tensors[offset] = (entry["s"], entry["v"])
            offset += 1
    return tensors


def _flat_hidden_base(param_count):
    """First flat index of the hidden slots for *param_count* parameters."""
    return 4 * param_count + 1


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

    *applied* maps a flat tensor index (params, grads, optimizer moments,
    step count, then hidden) to ``(shape, tree)``.  Hidden indices only
    exist once a delta introduced hidden state; a complete set must be
    present at that point.
    """
    param_count = len(basis_document["params"])
    layers = basis_document["layers"]
    hidden_count = len(basis_document["hidden"]) if basis_document["hidden"] is not None else None

    # The latest delta in the chain decides whether hidden state exists.
    final_hidden_count = applied.get("_hc", hidden_count)
    tensors = applied["tensors"]

    hidden_base = _flat_hidden_base(param_count)
    total = hidden_base + (final_hidden_count or 0)
    for index in range(total):
        if index not in tensors:
            raise CheckpointError(
                f"chain is missing tensor {index}: the delta chain is incomplete"
            )

    def entries(start, stop):
        return [
            {"s": list(tensors[i][0]), "v": tensors[i][1]}
            for i in range(start, stop)
        ]

    params = entries(0, param_count)
    grads = entries(param_count, 2 * param_count)
    optim_m = entries(2 * param_count, 3 * param_count)
    optim_v = entries(3 * param_count, 4 * param_count)
    step_shape, step_count = tensors[4 * param_count]
    if step_shape != []:
        raise CheckpointError("optimizer step count tensor must be a scalar")
    _check_step_count(step_count)
    hidden = entries(hidden_base, total)

    if len(params) != param_count or len(grads) != param_count:
        raise CheckpointError("chain did not preserve the parameter/gradient count")
    return {
        "params": params,
        "grads": grads,
        "hidden": hidden if final_hidden_count is not None else None,
        "optim": {"t": step_count, "m": optim_m, "v": optim_v},
        "layers": layers,
        "pending": applied["pending"],
    }


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


class _DirectoryChainLock:
    """Mutual exclusion for one chain directory, in-process and across
    processes.

    The per-process threading lock serialises this process; an advisory
    ``flock`` on the directory's own file descriptor serialises
    cooperating processes (no extra lock file is created, so a chain
    directory never gains visible entries).  Where the directory cannot
    be locked -- a platform without ``fcntl``, or a filesystem that
    refuses the lock -- the in-process lock alone is used.  Every
    directory-chain operation (save, load, compact) runs entirely under
    this lock, so a reader never observes a half-committed chain and a
    compaction never races a save.
    """

    def __init__(self, directory):
        self._directory = directory
        self._thread_lock = _chain_lock(directory)
        self._fd = None

    def __enter__(self):
        self._thread_lock.acquire()
        if fcntl is not None:
            try:
                fd = os.open(
                    self._directory,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
            except OSError:
                fd = None
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except OSError:
                    os.close(fd)
                else:
                    self._fd = fd
        return self

    def __exit__(self, *exc_info):
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self._fd)
            self._fd = None
        self._thread_lock.release()
        return False


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
        # Re-entrant so a whole compaction can run under the store lock
        # while calling the per-operation methods.
        self._lock = threading.RLock()

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
    with _DirectoryChainLock(directory):
        store = _DirectoryChainStore(directory)
        _recover_directory_chain(directory, store)
        _save_chain_store(document, store)
    return None


def save_chain_memory(document, store):
    """Append *document* to an in-memory chain (used by the self-checks)."""
    if not isinstance(store, MemoryChain):
        raise TypeError("store must be a MemoryChain")
    # The whole append runs under the store lock so a concurrent
    # compaction can never swap the chain between the diff and the commit.
    with store._lock:
        _save_chain_store(document, store)
    return None


def _save_chain_store(document, store):
    if isinstance(document, dict) and document.get("optim") is None:
        # Tolerate pre-v3 in-memory documents: the optimizer never stepped.
        document = _migrate_add_optim(dict(document))
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
    hidden_shapes_introduced = (
        cur_hc is not None and prev_hc is None
    )
    hidden_base = _flat_hidden_base(param_count)
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
    with _DirectoryChainLock(directory):
        store = _DirectoryChainStore(directory)
        _recover_directory_chain(directory, store)
        return _load_chain_root(store, up_to)


def load_chain_memory(store, up_to=None):
    """Reassemble a full state document from an in-memory chain."""
    if not isinstance(store, MemoryChain):
        raise TypeError("store must be a MemoryChain")
    with store._lock:
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
    pending = basis["pending"]
    hidden_base = _flat_hidden_base(param_count)
    for index in range(1, head + 1):
        name = _segment_name(index)
        raw = store.read_segment(name)
        _number, delta_hc, delta_pending, items = _parse_delta(raw, index)
        segment_version = struct.unpack("<I", raw[8:12])[0]

        # Hidden-state transitions: absent -> present exactly once with a
        # full set of slots; afterwards the count is fixed.
        introducing = delta_hc is not None and current_hc is None
        if current_hc is not None and delta_hc != current_hc:
            raise CheckpointError(
                f"delta {index} changes the hidden slot count; chain rejected"
            )
        # v2 segments index hidden slots right after the gradients; v3
        # segments place the optimizer state in between.
        seg_hidden_base = (
            hidden_base if segment_version >= 3 else 2 * param_count
        )
        if not introducing and delta_hc is None and current_hc is None:
            expected_max = seg_hidden_base - 1
        elif delta_hc is None:
            raise CheckpointError(
                f"delta {index} drops hidden state the chain already fixed"
            )
        else:
            expected_max = seg_hidden_base + delta_hc - 1

        replaced_hidden = set()
        for flat_index, shape, tree in items:
            if flat_index > expected_max:
                raise CheckpointError(
                    f"delta {index} names tensor {flat_index} beyond its state"
                )
            if segment_version < 3 and flat_index >= 2 * param_count:
                # Remap a v2 hidden index into the current flat layout.
                slot = flat_index - 2 * param_count
                if slot >= delta_hc:
                    raise CheckpointError(
                        f"delta {index} names hidden slot {slot} beyond its count"
                    )
                flat_index = hidden_base + slot
                prior = tensors.get(flat_index)
                if prior is not None and list(prior[0]) != list(shape):
                    raise CheckpointError(
                        f"delta {index} hidden slot {slot} shape disagrees with "
                        "the slot introduced earlier in the chain"
                    )
                replaced_hidden.add(slot)
                tensors[flat_index] = (list(shape), tree)
                continue
            if flat_index < 2 * param_count:
                expected_shape = (
                    basis["params"][flat_index]["s"]
                    if flat_index < param_count
                    else basis["grads"][flat_index - param_count]["s"]
                )
                if list(shape) != list(expected_shape):
                    raise CheckpointError(
                        f"delta {index} tensor {flat_index} shape disagrees with basis"
                    )
            elif flat_index < hidden_base:
                # Optimizer state: the moments mirror the parameter shapes
                # and the step count is a scalar.
                if flat_index < 3 * param_count:
                    expected_shape = basis["params"][flat_index - 2 * param_count]["s"]
                elif flat_index < 4 * param_count:
                    expected_shape = basis["params"][flat_index - 3 * param_count]["s"]
                else:
                    expected_shape = []
                    if (
                        isinstance(tree, bool)
                        or not isinstance(tree, int)
                        or tree < 0
                    ):
                        raise CheckpointError(
                            f"delta {index} carries an invalid optimizer step count"
                        )
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
        pending = delta_pending

    applied = {"tensors": tensors, "pending": pending, "_hc": current_hc}
    return _assemble(basis, applied)


# ---------------------------------------------------------------------------
# Chain compaction
# ---------------------------------------------------------------------------


def _plan_compaction(store, up_to):
    """Validate the chain and compute the compacted segment contents.

    Returns ``None`` when there is nothing to merge (a basis-only chain,
    or ``up_to=0``); otherwise ``(new_head, segments)`` where *segments*
    holds the full new content of every segment ``0..new_head``: a folded
    basis (segments ``0..up_to`` reassembled and re-frozen as one native
    current-version snapshot) followed by the remaining deltas, rebuilt
    deterministically against their new predecessors.  Every segment of
    the chain is validated (and old versions migrated) exactly as a load
    would, so any corruption or shape drift rejects the whole compaction
    before anything is written.
    """
    head = _read_head_optional(store)
    if head is None:
        raise CheckpointError(
            "chain has no head pointer (no basis segment committed)"
        )
    if up_to is None:
        up_to = head
    if isinstance(up_to, bool) or not isinstance(up_to, int) or up_to < 0:
        raise CheckpointError("up_to must be a non-negative segment index")
    if up_to > head:
        raise CheckpointError(
            f"up_to segment {up_to} is beyond the chain head {head}"
        )
    if up_to == 0 or head == 0:
        return None
    folded = _load_chain_store(store, up_to)
    segments = [build_bytes(folded)]
    previous = folded
    for index in range(up_to + 1, head + 1):
        document = _load_chain_store(store, index)
        segments.append(_build_delta_between(previous, document, index - up_to))
        previous = document
    return head - up_to, segments


def compact_chain(directory, up_to=None):
    """Fold the basis and the deltas through *up_to* into one new basis.

    The reassembled state -- parameters, gradients, optimizer moments and
    step count, hidden state -- is bit for bit identical before and after;
    only the segment count changes (deterministically, by exactly the
    merged range).  ``up_to=None`` folds everything through the current
    head, leaving a single basis segment.  A chain with nothing to merge
    (basis only, or ``up_to=0``) is left untouched.  Old-version segments
    participate exactly as on load and the compacted chain is rewritten
    in the current format version.

    The compaction commits through a stage-then-roll-forward protocol:
    a process killed at any point leaves either the old or the new head
    reachable, and the next open of the chain finishes the roll-forward.
    A missing directory raises FileNotFoundError; an unwritable
    directory or a full disk raises OSError; any corrupt, truncated or
    inconsistent segment rejects the whole compaction with ValueError
    and leaves the chain untouched.
    """
    if not isinstance(directory, (str, os.PathLike)):
        raise TypeError("chain directory must be a path")
    directory = os.fspath(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"incremental checkpoint directory not found: {directory!r}"
        )
    with _DirectoryChainLock(directory):
        store = _DirectoryChainStore(directory)
        _recover_directory_chain(directory, store)
        plan = _plan_compaction(store, up_to)
        if plan is not None:
            new_head, segments = plan
            _commit_compaction(directory, store, new_head, segments)
    return None


def compact_chain_memory(store, up_to=None):
    """Compact an in-memory chain; same semantics as :func:`compact_chain`."""
    if not isinstance(store, MemoryChain):
        raise TypeError("store must be a MemoryChain")
    with store._lock:
        plan = _plan_compaction(store, up_to)
        if plan is None:
            return None
        new_head, segments = plan
        for index, raw in enumerate(segments):
            store.write_segment(_segment_name(index), raw)
        for name in list(store._objects):
            index = _segment_index(name)
            if index is not None and index > new_head:
                del store._objects[name]
        _commit_head(store, new_head)
    return None


def _commit_compaction(directory, store, new_head, segments):
    """Atomically replace the live chain with the compacted *segments*.

    Phase one stages every new segment under a private name (each write
    is complete and durable on its own) and records the new head in the
    compaction marker.  Phase two is exactly the recovery routine: copy
    the staged segments over the live names, advance the head, clean up.
    A crash before the marker leaves the old chain untouched; a crash
    after it is rolled forward by the next open.
    """
    for index, raw in enumerate(segments):
        _atomic_write(directory, _staged_name(index), raw)
    _atomic_write(directory, _COMPACT_MARKER, str(new_head).encode("ascii"))
    _recover_directory_chain(directory, store)


def _recover_directory_chain(directory, store):
    """Roll an interrupted compaction forward to one complete chain state.

    Runs at the start of every directory-chain operation, under the
    directory lock.  With no marker the live chain is already one
    complete state (any staged leftovers come from a compaction that
    crashed before its marker and are simply dropped).  With a marker,
    the staged chain is complete on disk, so finishing the copy -- every
    write is atomic and idempotent -- and advancing the head restores
    exactly the post-compaction state.  The marker is removed last, so a
    crash anywhere in the roll-forward is itself recovered by the next
    open.
    """
    leftovers = [
        name
        for name in os.listdir(directory)
        if name.startswith(_STAGED_PREFIX)
    ]
    marker_path = os.path.join(directory, _COMPACT_MARKER)
    if not os.path.exists(marker_path):
        for name in leftovers:
            _unlink_quietly(os.path.join(directory, name))
        if leftovers:
            _fsync_directory(directory)
        return
    with open(marker_path, "rb") as fh:
        raw = fh.read()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise CheckpointError("compaction marker is corrupt") from None
    if not re.fullmatch(r"\d+", text):
        raise CheckpointError("compaction marker is corrupt")
    new_head = int(text)
    if _read_head_optional(store) != new_head:
        # The head was never advanced: finish the roll-forward.  The
        # marker is only written after every staged segment, so a missing
        # staged file here means the staging area itself is damaged.
        for index in range(new_head + 1):
            staged_path = os.path.join(directory, _staged_name(index))
            try:
                with open(staged_path, "rb") as fh:
                    raw_segment = fh.read()
            except FileNotFoundError:
                raise CheckpointError(
                    "compaction staging area is incomplete; the chain "
                    "cannot be rolled forward"
                ) from None
            _atomic_write(directory, _segment_name(index), raw_segment)
        _commit_head(store, new_head)
    # Committed (or just finished): drop the staging area, the segments
    # the compaction made unreachable, and finally the marker.
    for name in os.listdir(directory):
        index = _segment_index(name)
        if name.startswith(_STAGED_PREFIX) or (
            index is not None and index > new_head
        ):
            _unlink_quietly(os.path.join(directory, name))
    _unlink_quietly(marker_path)
    _fsync_directory(directory)


def _unlink_quietly(path):
    # Garbage collection only: a file that cannot be removed (a read-only
    # directory, say) is unreachable anyway and must not fail the open.
    try:
        os.unlink(path)
    except OSError:
        pass


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
        return save_chain_memory(document, target)
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
