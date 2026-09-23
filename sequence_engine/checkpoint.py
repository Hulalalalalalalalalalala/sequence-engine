"""Checkpoint codec and atomic on-disk commit (standard library only).

Envelope layout (all multi-byte integers big-endian)::

    magic        8 bytes  ``SEQENCKP``
    version      u16      envelope format version (1)
    payload_len  u64      number of payload bytes
    digest       32 bytes SHA-256 of the payload
    payload      payload_len bytes   tagged self-delimiting document
    end_magic    8 bytes  ``ENDSEQCK``
    total_len    u64      total file length including header and footer

A reader verifies the magic, the version, the exact file length, both
length fields, the trailing marker and the digest before decoding a single
payload byte.  A process killed while writing leaves its bytes in a sibling
``.tmp.*`` file (the real target is only ever installed atomically with
``os.replace``); such a leftover, or any truncated/corrupted checkpoint,
fails these checks and is rejected.

Payload tags:

    u8 0x4C list:  u32 count, then that many values
    u8 0x49 int:   signed i64
    u8 0x46 float: raw f64 bits (round-trips -0.0 exactly; non-finite
                   values are rejected on both encode and decode)
    u8 0x4E none
    u8 0x53 string: u32 byte count, then UTF-8
    u8 0x44 dict:  u32 count, then alternating string keys and values
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct

_MAGIC = b"SEQENCKP"
_END_MAGIC = b"ENDSEQCK"
_ENVELOPE_VERSION = 1
_DOCUMENT_VERSION = 1

_TAG_LIST = 0x4C
_TAG_INT = 0x49
_TAG_FLOAT = 0x46
_TAG_NONE = 0x4E
_TAG_STRING = 0x53
_TAG_DICT = 0x44

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_MAX_DEPTH = 64
_UINT32_MAX = (1 << 32) - 1

_HEADER_LEN = 8 + 2 + 8 + 32
_FOOTER_LEN = 8 + 8

_DOCUMENT_KEYS = frozenset({"ver", "hshapes", "layers"})
_LAYER_KEYS = frozenset({"params", "hidden"})
_PARAM_KEYS = frozenset({"shape", "data", "grad"})
_SLOT_KEYS = frozenset({"shape", "data"})


class CheckpointFormatError(ValueError):
    """Raised for any checkpoint that cannot be parsed or validated."""


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _append_int(buf, value):
    buf.append(_TAG_INT)
    buf.extend(struct.pack(">q", value))


def _append_float(buf, value):
    buf.append(_TAG_FLOAT)
    buf.extend(struct.pack(">d", value))


def _append_string(buf, value):
    raw = value.encode("utf-8")
    if len(raw) > _UINT32_MAX:
        raise CheckpointFormatError("string too long to encode")
    buf.append(_TAG_STRING)
    buf.extend(struct.pack(">I", len(raw)))
    buf.extend(raw)


def _encode_value(buf, value, depth):
    if depth > _MAX_DEPTH:
        raise CheckpointFormatError("checkpoint nesting is too deep")
    if isinstance(value, bool):
        raise ValueError("checkpoint state must not contain booleans")
    if isinstance(value, int):
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError("checkpoint state contains an out-of-range integer")
        _append_int(buf, value)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("checkpoint state contains a non-finite float")
        _append_float(buf, value)
    elif value is None:
        buf.append(_TAG_NONE)
    elif isinstance(value, str):
        _append_string(buf, value)
    elif isinstance(value, list):
        if len(value) > _UINT32_MAX:
            raise CheckpointFormatError("list too long to encode")
        buf.append(_TAG_LIST)
        buf.extend(struct.pack(">I", len(value)))
        for item in value:
            _encode_value(buf, item, depth + 1)
    elif isinstance(value, dict):
        if len(value) > _UINT32_MAX:
            raise CheckpointFormatError("dict too large to encode")
        buf.append(_TAG_DICT)
        buf.extend(struct.pack(">I", len(value)))
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointFormatError("checkpoint dict keys must be strings")
            _append_string(buf, key)
            _encode_value(buf, item, depth + 1)
    else:
        raise ValueError(
            f"checkpoint state must contain numbers only, got {type(value).__name__}"
        )


def encode(document):
    """Encode a validated checkpoint document and return envelope bytes."""
    _validate_document(document)
    payload = bytearray()
    _encode_value(payload, document, 0)
    digest = hashlib.sha256(payload).digest()
    out = bytearray()
    out += _MAGIC
    out += struct.pack(">H", _ENVELOPE_VERSION)
    out += struct.pack(">Q", len(payload))
    out += digest
    out += payload
    out += _END_MAGIC
    out += struct.pack(">Q", _HEADER_LEN + len(payload) + _FOOTER_LEN)
    return bytes(out)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


class _Reader:
    def __init__(self, data):
        self._data = data
        self._pos = 0
        self._end = len(data)

    def remaining(self):
        return self._end - self._pos

    def take(self, count):
        if count < 0 or self._pos + count > self._end:
            raise CheckpointFormatError("checkpoint payload is truncated")
        chunk = self._data[self._pos : self._pos + count]
        self._pos += count
        return chunk

    def u8(self):
        return self.take(1)[0]

    def packed(self, fmt, size):
        return struct.unpack(fmt, self.take(size))[0]


def _decode_value(reader, depth):
    if depth > _MAX_DEPTH:
        raise CheckpointFormatError("checkpoint nesting is too deep")
    tag = reader.u8()
    if tag == _TAG_INT:
        return reader.packed(">q", 8)
    if tag == _TAG_FLOAT:
        value = reader.packed(">d", 8)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("checkpoint contains a non-finite float")
        return value
    if tag == _TAG_NONE:
        return None
    if tag == _TAG_STRING:
        length = reader.packed(">I", 4)
        raw = reader.take(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise CheckpointFormatError("checkpoint contains invalid UTF-8") from None
    if tag == _TAG_LIST:
        count = reader.packed(">I", 4)
        # Every encoded value needs at least one byte; this bounds memory
        # against a corrupted count before any list is allocated.
        if count > reader.remaining():
            raise CheckpointFormatError("checkpoint list length is implausible")
        return [_decode_value(reader, depth + 1) for _ in range(count)]
    if tag == _TAG_DICT:
        count = reader.packed(">I", 4)
        if 5 * count > reader.remaining():
            raise CheckpointFormatError("checkpoint dict length is implausible")
        result = {}
        for _ in range(count):
            key_tag = reader.u8()
            if key_tag != _TAG_STRING:
                raise CheckpointFormatError("checkpoint dict keys must be strings")
            length = reader.packed(">I", 4)
            key = reader.take(length)
            try:
                key = key.decode("utf-8")
            except UnicodeDecodeError:
                raise CheckpointFormatError(
                    "checkpoint contains invalid UTF-8 in a dict key"
                ) from None
            result[key] = _decode_value(reader, depth + 1)
        return result
    raise CheckpointFormatError(f"unknown checkpoint payload tag 0x{tag:02x}")


def _decode_envelope(blob):
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise CheckpointFormatError("checkpoint source must be bytes-like")
    data = bytes(blob)
    if len(data) < _HEADER_LEN + _FOOTER_LEN:
        raise CheckpointFormatError("checkpoint is truncated")
    if data[:8] != _MAGIC:
        raise CheckpointFormatError("not a sequence-engine checkpoint (bad magic)")
    version = struct.unpack(">H", data[8:10])[0]
    if version != _ENVELOPE_VERSION:
        raise CheckpointFormatError(
            f"unsupported checkpoint envelope version {version}"
        )
    payload_len = struct.unpack(">Q", data[10:18])[0]
    digest = data[18:50]
    total = _HEADER_LEN + payload_len + _FOOTER_LEN
    if total != len(data):
        raise CheckpointFormatError("checkpoint length does not match its contents")
    payload = data[_HEADER_LEN : _HEADER_LEN + payload_len]
    if not hmac.compare_digest(hashlib.sha256(payload).digest(), digest):
        raise CheckpointFormatError("checkpoint is corrupted (digest mismatch)")
    footer = data[_HEADER_LEN + payload_len :]
    if footer[:8] != _END_MAGIC:
        raise CheckpointFormatError("checkpoint end marker is missing")
    footer_total = struct.unpack(">Q", footer[8:16])[0]
    if footer_total != len(data):
        raise CheckpointFormatError("checkpoint total length field is wrong")
    return payload


def decode(blob):
    """Validate the envelope and return the checked checkpoint document."""
    payload = _decode_envelope(blob)
    reader = _Reader(payload)
    document = _decode_value(reader, 0)
    if reader.remaining() != 0:
        raise CheckpointFormatError("checkpoint has trailing payload bytes")
    _validate_document(document)
    return document


# ---------------------------------------------------------------------------
# Document schema
# ---------------------------------------------------------------------------


def _fail(message):
    raise CheckpointFormatError(message)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_shape(shape, allow_empty=False):
    if not isinstance(shape, list):
        _fail("tensor shape must be a list of dimensions")
    if not shape and not allow_empty:
        _fail("tensor shape must be a non-empty list of dimensions")
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            _fail("tensor shape must contain positive integers")
    return shape


def _validate_tree(value, shape, depth=0):
    if depth > _MAX_DEPTH:
        _fail("tensor data nesting is too deep")
    if not shape:
        if not _is_number(value):
            _fail("tensor data must contain numbers only")
        return
    dim = shape[0]
    if not isinstance(value, list) or len(value) != dim:
        _fail("tensor data does not match its declared shape")
    for item in value:
        _validate_tree(item, shape[1:], depth + 1)


def _validate_param(record):
    if not isinstance(record, dict) or set(record) != _PARAM_KEYS:
        _fail("parameter record has missing or unexpected fields")
    shape = _validate_shape(record["shape"], allow_empty=True)
    _validate_tree(record["data"], shape)
    _validate_tree(record["grad"], shape)


def _validate_slot(record):
    if record is None:
        return None
    if not isinstance(record, dict) or set(record) != _SLOT_KEYS:
        _fail("hidden slot record has missing or unexpected fields")
    shape = _validate_shape(record["shape"])
    _validate_tree(record["data"], shape)
    return shape


def _validate_document(document):
    if not isinstance(document, dict) or set(document) != _DOCUMENT_KEYS:
        _fail("checkpoint document has missing or unexpected fields")
    ver = document["ver"]
    if isinstance(ver, bool) or not isinstance(ver, int) or ver != _DOCUMENT_VERSION:
        raise CheckpointFormatError(
            f"unsupported checkpoint document version {ver!r}"
        )
    layers = document["layers"]
    hshapes = document["hshapes"]
    if not isinstance(layers, list):
        _fail("layers must be a list")
    if not isinstance(hshapes, list) or len(hshapes) != len(layers):
        _fail("hidden shape table must parallel the layer table")
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != _LAYER_KEYS:
            _fail(f"layer {index} record has missing or unexpected fields")
        params = layer["params"]
        if not isinstance(params, list):
            _fail(f"layer {index} parameters must be a list")
        for record in params:
            _validate_param(record)
        saved_shape = _validate_slot(layer["hidden"])
        cached = hshapes[index]
        if cached is not None:
            _validate_shape(cached)
        if saved_shape is not None and cached is None:
            _fail(f"layer {index} has hidden values but no cached slot shape")
        if saved_shape is not None and cached is not None and saved_shape != cached:
            _fail(f"layer {index} hidden slot disagrees with cached hidden shape")
    return document


# ---------------------------------------------------------------------------
# Atomic file commit
# ---------------------------------------------------------------------------


_tmp_counter = 0


def write_atomic(path, data):
    """Write *data* to *path* via a temp file and ``os.replace``.

    Any failure to create, write, flush or rename propagates as OSError;
    an existing target is replaced in place and no backup is kept.
    """
    global _tmp_counter
    directory = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(path)
    # Sweep leftovers from processes that died mid-commit for this target.
    try:
        for entry in os.listdir(directory):
            if entry.startswith(f".{base}.tmp."):
                try:
                    os.unlink(os.path.join(directory, entry))
                except OSError:
                    pass
    except OSError:
        pass
    _tmp_counter += 1
    tmp_path = os.path.join(
        directory, f".{base}.tmp.{os.getpid()}.{_tmp_counter}"
    )
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # The file is already committed; directory durability hints are
            # best effort on platforms that do not support fsync on dirs.
            pass
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
