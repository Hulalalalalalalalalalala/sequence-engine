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
  **compacted** in place by a streaming online fold: the new basis is
  checked and written segment by segment, the tail deltas are converted a
  segment at a time in slots the chain already owned (so peak disk usage
  stays within the original chain plus the one new basis), and saves and
  loads interleave the whole time.  A marker plus a cross-process lease
  name a fold in progress, with the tail promoted and the head advanced
  in one final step, so a crash mid-compaction is rolled forward on the
  next open and the directory always holds exactly one complete chain --
  the old head's or the new head's.  A chain directory can also be
  **verified** strictly read-only: each segment is walked like a load and
  the first bad one is located and reported without a single write.
* **Chain families** -- a chain can be **forked** at any committed
  segment: the new chain directory shares every segment up to and
  including the fork point with the source chain (the prefix files are
  hard-linked, so they are stored once) and from then on maintains only
  its own head and the deltas it appends.  The chains of a family save,
  load, stream-compact and verify independently; shared segments are
  immutable (every write anywhere is a temp file plus a rename), and a
  shared segment's bytes are reclaimed by reachability -- exactly when no
  chain's head can reach it any more, whether the reference went away
  through a compaction or through the **deletion** of a whole branch.
  Reclamation is a pure reachability function, so sweeping again changes
  nothing; segments no head can reach (orphans left by a hard kill) and
  staging directories from a killed fork, delete or compaction are swept
  deterministically on the next fork, compaction, deletion or merge.  A
  family of chains can also be compacted in one call --
  :func:`compact_family` folds the prefix segments every member reaches
  through the same underlying files into one new basis segment shared by
  every member, with each member's own tail following it -- and verified
  read-only, with the first bad segment reported together with every
  chain that reaches it.
* **Branch merges** -- one chain's current state can be merged onto
  another: the target keeps every segment it had and receives one new,
  target-owned delta segment after its head carrying only the tensors in
  which the source's head state genuinely differs (an empty delta when
  the states already agree), so loading the target is bit for bit the
  source's merge-time state while the source's head and files never move
  and shared segments are never written twice.  The merge uses the same
  segment-then-head append commit a save does -- a kill leaves only the
  old or the new head, both one complete chain -- and its orphan residue
  is reclaimed by the next fork, compaction, deletion or merge.

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

import contextlib
import errno
import json
import math
import os
import re
import struct
import tempfile
import threading
import time
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

# Fork staging: a branch chain is fully populated (segment links first,
# the head pointer last -- the chain commit order) under a private
# sibling directory and then renamed over the target in one step, so the
# target either appears as one complete chain or not at all.  A killed
# fork leaves only a staging directory, which the next fork to the same
# target reclaims.
_FORK_TMP_PREFIX = ".seqfork.tmp-"

# Branch deletion: the chain directory is first renamed aside under this
# sibling prefix (the unlink that makes the deleted chain unreachable is
# one directory-entry change, so the family observes either the complete
# branch or none of it) and then emptied and removed.  A kill between the
# rename and the teardown leaves only a directory carrying this prefix,
# which the next fork, compaction or deletion sweeps deterministically.
_DELETE_TMP_PREFIX = ".seqdel.tmp-"

# Streaming-compaction leases.  A compaction no longer folds the chain in
# one critical section: it stages a new basis and then converts one tail
# segment per lock acquisition, so saves and loads keep flowing through
# the directory.  Two pieces identify a stream that is still running:
#
# * an exclusive ``flock`` on a small, never-recreated lease file held for
#   the whole run -- the cross-process lease (the kernel releases it when
#   the owner dies);
# * this in-process registry -- lets another thread in the owner process
#   read through the marker instead of rolling the fold forward.
#
# The marker names the fold point while the stream runs (``{"u": u}``)
# and the final new head once publication is in force (``{"u": u,
# "h": new_head}``); it is rewritten atomically at that one transition.
# Progress otherwise lives in the converted tail slots.  A marker whose
# lease is unheld belongs to a dead owner and is rolled forward by the
# next open.
_LEASE_NAME = ".seqcompact.lock"
_compactions_guard = threading.Lock()
_active_compactions: set[str] = set()

# Family-level compaction.  One fold covers the prefix segments every
# member of a chain family reaches through the same underlying (hard
# linked) segment files; the folded prefix becomes one new basis segment
# shared by every member, and each member's member-owned tail follows it.
#
# The family operation adds no per-chain protocol of its own: every
# member runs the ordinary streaming fold (its own staged basis, marker
# and lease), so saves, loads, read-only verification and crash recovery
# keep working unchanged.  The only family-level pieces are:
#
# * one cross-process lease file in the family (parent) directory, held
#   for the whole run so two family operations serialise and a dead
#   owner's staging residue is recognisable as debris;
# * the folded basis, written once to a ``.seqfam.tmp-*`` file in the
#   parent directory and hard-linked into each member's staged-basis
#   slot, so the new basis physically exists once;
# * link-preserving publication: each member's staged basis link is
#   ``rename``d atomically over its basis slot instead of being copied,
#   so every member's new basis is the one shared inode.
#
# Killed mid-fold, each member's own dead marker is rolled forward by
# the member's next open (the ordinary protocol); the family lease is
# released by the kernel and the parent staging file -- plus any other
# family staging residue -- is reclaimed deterministically by the next
# family operation.
_FAMILY_LOCK_NAME = ".seqfam.lock"
_FAMILY_TMP_PREFIX = ".seqfam.tmp-"
_family_locks_guard = threading.Lock()
_family_locks: dict[str, threading.Lock] = {}
_active_families_guard = threading.Lock()
_active_families: set[str] = set()


def _lease_register(directory):
    with _compactions_guard:
        _active_compactions.add(os.path.abspath(directory))


def _lease_active_in_process(directory):
    with _compactions_guard:
        return os.path.abspath(directory) in _active_compactions


def _lease_release(directory):
    with _compactions_guard:
        _active_compactions.discard(os.path.abspath(directory))


def _lease_path(directory):
    return os.path.join(directory, _LEASE_NAME)


def _marker_live(directory):
    """Whether a streaming fold over *directory* is currently running.

    Honors both the in-process registry and a cross-process ``flock`` on
    the lease file.  Best effort: platforms without ``fcntl`` see only the
    in-process lease, which is the only kind such a platform creates.
    """
    if _lease_active_in_process(directory):
        return True
    if fcntl is None:
        return False
    path = _lease_path(directory)
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                return True
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        return False
    finally:
        os.close(fd)


def _wait_for_live_marker(directory):
    """Block until any live fold over *directory* has ended."""
    if not _marker_live(directory):
        return
    if fcntl is None or _lease_active_in_process(directory):
        # The owner runs in this process -- or this platform has no
        # cross-process locks.  Blocking on flock would self-deadlock
        # against an owner in the same process, so wait at the owner's
        # lock-handoff boundaries until it deregisters.
        while _lease_active_in_process(directory):
            with _chain_lock(directory):
                pass
            if _lease_active_in_process(directory):
                time.sleep(0.0)
        return
    # Another process owns the fold; its flock is released when the fold
    # completes or the owner dies.
    try:
        fd = os.open(_lease_path(directory), os.O_RDONLY)
    except FileNotFoundError:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
    finally:
        os.close(fd)


def _family_dir_lock(parent):
    """The in-process lock serialising family operations in one parent dir."""
    key = os.path.abspath(parent)
    with _family_locks_guard:
        lock = _family_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _family_locks[key] = lock
    return lock


def _family_register(parent):
    with _active_families_guard:
        _active_families.add(os.path.abspath(parent))


def _family_active_in_process(parent):
    with _active_families_guard:
        return os.path.abspath(parent) in _active_families


def _family_release(parent):
    with _active_families_guard:
        _active_families.discard(os.path.abspath(parent))


def _family_lease_path(parent):
    return os.path.join(parent, _FAMILY_LOCK_NAME)


def _family_stage_basis(parent, folded_bytes):
    """Write the one shared folded basis to a private name in *parent*.

    Returns the staging path.  The file is the physical file every
    member's staged basis is linked to, so it is written with the same
    temp-file-plus-fsync discipline as a segment.
    """
    fd, path = tempfile.mkstemp(prefix=_FAMILY_TMP_PREFIX, dir=parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(folded_bytes)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        _unlink_quietly(path)
        raise
    return path


def _family_lease_live(parent):
    """Whether a family operation over *parent* is currently running.

    Honors the in-process registry and, on POSIX, a cross-process
    advisory ``flock`` on the family lease file.  Best effort: platforms
    without ``fcntl`` see only the in-process lease.
    """
    if _family_active_in_process(parent):
        return True
    if fcntl is None:
        return False
    path = _family_lease_path(parent)
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                return True
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        return False
    finally:
        os.close(fd)


def _try_family_lease(parent):
    """Take the cross-process family lease non-blockingly.

    Returns the locked fd, ``None`` when another live process holds it,
    or ``False`` on platforms without ``fcntl`` (the directory and
    in-process locks then provide best-effort exclusion).
    """
    if fcntl is None:
        return False
    path = _family_lease_path(parent)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            return None
        raise
    return fd


def _release_family_lease(fd):
    if not fd:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    os.close(fd)


def _wait_for_family_lease(parent):
    """Block until any live family operation over *parent* has ended."""
    if not _family_lease_live(parent):
        return
    if fcntl is None or _family_active_in_process(parent):
        while _family_active_in_process(parent):
            with _family_dir_lock(parent):
                pass
            if _family_active_in_process(parent):
                time.sleep(0.0)
        return
    try:
        fd = os.open(_family_lease_path(parent), os.O_RDONLY)
    except FileNotFoundError:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
    finally:
        os.close(fd)


class _FamilyFoldGuard:
    """Serialise a single-chain fold with family folds in the same parent.

    Saves, loads, verifies, forks, merges and deletions do not take this
    guard -- only a fold (single-chain or family) does -- so ordinary
    chain traffic keeps flowing while maintenance folds serialise: a
    family fold blocks here until any single-chain fold of a sibling
    member finishes, and a single-chain fold blocks while a family fold
    is active.  On platforms without ``fcntl`` the in-process family lock
    provides the same guarantee within the process.
    """

    def __init__(self, chain_directory):
        self._parent = os.path.dirname(os.path.abspath(chain_directory))
        self._fd = None
        self._thread_lock = None

    def __enter__(self):
        if fcntl is None:
            self._thread_lock = _family_dir_lock(self._parent)
            self._thread_lock.acquire()
            return self
        fd = os.open(_family_lease_path(self._parent), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            # Blocking: wait out any family fold (or another single-chain
            # fold of a sibling) instead of spinning.  flock serialises
            # separate opens even within one process.
            fcntl.flock(fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(self, *exc_info):
        if self._fd is not None:
            fd = self._fd
            self._fd = None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)
            # Reclaim the lease file when no fold still uses it (a live
            # family fold always keeps its marker): the flock above is
            # released by close, so an unheld file here is only debris.
            if not os.path.exists(_family_marker_path(self._parent)):
                path = _family_lease_path(self._parent)
                try:
                    probe = os.open(path, os.O_RDONLY)
                    try:
                        try:
                            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except OSError:
                            pass
                        else:
                            os.unlink(path)
                        try:
                            fcntl.flock(probe, fcntl.LOCK_UN)
                        except OSError:
                            pass
                    finally:
                        os.close(probe)
                except OSError:
                    pass
        if self._thread_lock is not None:
            self._thread_lock.release()
            self._thread_lock = None
        return False


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
            except FileNotFoundError:
                # The directory was deleted (renamed aside by a branch
                # delete) between the caller's existence check and the
                # lock: report the documented missing-directory error
                # rather than degrading to a headless chain.
                self._thread_lock.release()
                raise
            except OSError:
                fd = None
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except OSError:
                    os.close(fd)
                else:
                    # The flock may have been waited out while another
                    # process holding it renamed this directory aside (a
                    # branch delete) and emptied it; the fd then names a
                    # detached staging directory, not the chain.  Compare
                    # identities after acquiring: the inode must still
                    # live at the locked path.
                    try:
                        fd_stat = os.fstat(fd)
                        path_stat = os.stat(self._directory)
                    except FileNotFoundError:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                        os.close(fd)
                        self._thread_lock.release()
                        raise
                    if (fd_stat.st_dev, fd_stat.st_ino) != (
                        path_stat.st_dev,
                        path_stat.st_ino,
                    ):
                        fcntl.flock(fd, fcntl.LOCK_UN)
                        os.close(fd)
                        self._thread_lock.release()
                        raise FileNotFoundError(
                            "incremental checkpoint directory not found: "
                            f"{self._directory!r}"
                        )
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

    def remove_head(self):
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

    def remove_head(self):
        with self._lock:
            # A memory delete drops the pointer and everything it made
            # unreachable in one critical section.
            self._objects.pop(_HEAD_NAME, None)

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
    try:
        with _DirectoryChainLock(directory):
            store = _DirectoryChainStore(directory)
            _recover_directory_chain(directory, store)
            _save_chain_directory(document, directory, store)
    except FileNotFoundError:
        # The directory was deleted (a branch delete) between the check
        # and the lock: saving into a missing directory is an OSError.
        raise OSError(
            errno.ENOENT,
            f"incremental checkpoint directory does not exist: {directory!r}",
        ) from None
    return None


def _save_chain_directory(document, directory, store):
    """Append under a directory lock, streaming a live fold if present.

    A save never waits for a fold to finish: it appends the next
    old-numbered delta exactly as in the ordinary path, reading the
    previous head state through the marker when a fold is mid-flight.
    The folder converts the new segment on its next lock acquisition.
    """
    if isinstance(document, dict) and document.get("optim") is None:
        document = _migrate_add_optim(dict(document))
    head_index = _read_head_optional(store)
    if head_index is None:
        store.write_segment(_segment_name(_BASIS_INDEX), build_bytes(document))
        _commit_head(store, _BASIS_INDEX)
        return
    fold = _read_directory_marker(directory)
    if fold is None:
        previous = _load_chain_store(store, head_index)
    else:
        previous = _load_through_marker(store, head_index, fold, None)
    next_index = head_index + 1
    delta_bytes = _build_delta_between(previous, document, next_index)
    store.write_segment(_segment_name(next_index), delta_bytes)
    _commit_head(store, next_index)


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
        return _load_chain_directory_root(directory, store, up_to)


def load_chain_memory(store, up_to=None):
    """Reassemble a full state document from an in-memory chain."""
    if not isinstance(store, MemoryChain):
        raise TypeError("store must be a MemoryChain")
    with store._lock:
        return _load_chain_root(store, up_to)


def _load_chain_directory_root(directory, store, up_to):
    """Load a directory chain, reading through a live streaming fold."""
    head = _read_head_optional(store)
    if head is None:
        raise CheckpointError(
            "chain has no head pointer (no basis segment committed)"
        )
    target = head
    if up_to is not None:
        if isinstance(up_to, bool) or not isinstance(up_to, int) or up_to < 0:
            raise CheckpointError("up_to must be a non-negative segment index")
        if up_to > head:
            raise CheckpointError(
                f"up_to segment {up_to} is beyond the chain head {head}"
            )
        target = up_to
    fold = _read_directory_marker(directory)
    if fold is None:
        return _load_chain_store(store, target)
    return _load_through_marker(store, head, fold, target)


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


class _ChainWalker:
    """Streaming chain assembly: fold one segment at a time.

    The basis is parsed on construction; :meth:`apply_delta` folds one
    delta segment and validates it against exactly the state accumulated
    so far, so a load, a streaming compaction and a verification all walk
    a chain identically.  ``tensors`` holds the post-segment flat tensor
    state (params, grads, optimizer moments, step count, then hidden
    slots); ``hidden_count``/``pending`` track the chain-level flags.

    Every failure is annotated with the offending segment's position;
    callers never see a structural error that does not name a segment.
    """

    def __init__(self, basis_raw):
        try:
            self.basis = parse_bytes(basis_raw)
        except CheckpointError as exc:
            raise CheckpointError(
                f"chain validation failed at basis segment 0: {exc}"
            ) from exc
        self.param_count = len(self.basis["params"])
        self.tensors = _tensors_from_document(self.basis)
        self.hidden_count = (
            len(self.basis["hidden"]) if self.basis["hidden"] is not None else None
        )
        self.pending = self.basis["pending"]
        self.hidden_base = _flat_hidden_base(self.param_count)

    def apply_delta(self, index, raw, wire_number=None):
        """Fold one delta segment into the accumulated state.

        *index* is the segment's position in the chain being walked;
        *wire_number* is the segment number recorded inside the frame when
        it differs (a still-unconverted old-numbering tail delta read while
        a streaming compaction is mid-flight).
        """
        expected = index if wire_number is None else wire_number
        try:
            self._apply_delta_checked(index, raw, expected)
        except CheckpointError as exc:
            raise CheckpointError(
                f"chain validation failed at delta segment {expected}: {exc}"
            ) from exc

    def _apply_delta_checked(self, index, raw, expected_number):
        _number, delta_hc, delta_pending, items = _parse_delta(raw, expected_number)
        segment_version = struct.unpack("<I", raw[8:12])[0]
        param_count = self.param_count
        hidden_base = self.hidden_base
        current_hc = self.hidden_count

        # Hidden-state transitions: absent -> present exactly once with a
        # full set of slots; afterwards the count is fixed.
        introducing = delta_hc is not None and current_hc is None
        if current_hc is not None and delta_hc != current_hc:
            raise CheckpointError("delta changes the hidden slot count; chain rejected")
        # v2 segments index hidden slots right after the gradients; v3
        # segments place the optimizer state in between.
        seg_hidden_base = hidden_base if segment_version >= 3 else 2 * param_count
        if not introducing and delta_hc is None and current_hc is None:
            expected_max = seg_hidden_base - 1
        elif delta_hc is None:
            raise CheckpointError("delta drops hidden state the chain already fixed")
        else:
            expected_max = seg_hidden_base + delta_hc - 1

        replaced_hidden = set()
        for flat_index, shape, tree in items:
            if flat_index > expected_max:
                raise CheckpointError(
                    f"delta names tensor {flat_index} beyond its state"
                )
            if segment_version < 3 and flat_index >= 2 * param_count:
                # Remap a v2 hidden index into the current flat layout.
                slot = flat_index - 2 * param_count
                if slot >= delta_hc:
                    raise CheckpointError(
                        f"delta names hidden slot {slot} beyond its count"
                    )
                flat_index = hidden_base + slot
                prior = self.tensors.get(flat_index)
                if prior is not None and list(prior[0]) != list(shape):
                    raise CheckpointError(
                        f"delta hidden slot {slot} shape disagrees with the "
                        "slot introduced earlier in the chain"
                    )
                replaced_hidden.add(slot)
                self.tensors[flat_index] = (list(shape), tree)
                continue
            if flat_index < 2 * param_count:
                expected_shape = (
                    self.basis["params"][flat_index]["s"]
                    if flat_index < param_count
                    else self.basis["grads"][flat_index - param_count]["s"]
                )
                if list(shape) != list(expected_shape):
                    raise CheckpointError(
                        f"delta tensor {flat_index} shape disagrees with basis"
                    )
            elif flat_index < hidden_base:
                # Optimizer state: moments mirror the parameter shapes and
                # the step count is a scalar.
                if flat_index < 3 * param_count:
                    expected_shape = self.basis["params"][
                        flat_index - 2 * param_count
                    ]["s"]
                elif flat_index < 4 * param_count:
                    expected_shape = self.basis["params"][
                        flat_index - 3 * param_count
                    ]["s"]
                else:
                    expected_shape = []
                    if isinstance(tree, bool) or not isinstance(tree, int) or tree < 0:
                        raise CheckpointError(
                            "delta carries an invalid optimizer step count"
                        )
                if list(shape) != list(expected_shape):
                    raise CheckpointError(
                        f"delta tensor {flat_index} shape disagrees with basis"
                    )
            else:
                slot = flat_index - hidden_base
                if slot >= delta_hc:
                    raise CheckpointError(
                        f"delta names hidden slot {slot} beyond its count"
                    )
                prior = self.tensors.get(flat_index)
                if prior is not None and list(prior[0]) != list(shape):
                    raise CheckpointError(
                        f"delta hidden slot {slot} shape disagrees with the "
                        "slot introduced earlier in the chain"
                    )
                replaced_hidden.add(slot)
            self.tensors[flat_index] = (list(shape), tree)

        if introducing:
            layer_count = len(self.basis["layers"])
            if delta_hc != layer_count:
                raise CheckpointError(
                    f"delta introduces {delta_hc} hidden slots for "
                    f"{layer_count} layers"
                )
            if replaced_hidden != set(range(delta_hc)):
                raise CheckpointError("delta introduces hidden state incompletely")
        self.hidden_count = delta_hc
        self.pending = delta_pending

    def document(self):
        applied = {
            "tensors": self.tensors,
            "pending": self.pending,
            "_hc": self.hidden_count,
        }
        return _assemble(self.basis, applied)


def _load_chain_store(store, head):
    walker = _ChainWalker(store.read_segment(_segment_name(_BASIS_INDEX)))
    for index in range(1, head + 1):
        raw = store.read_segment(_segment_name(index))
        walker.apply_delta(index, raw)
    return walker.document()


# ---------------------------------------------------------------------------
# Chain verification (read-only)
#
# A verifier walks a chain exactly the way a load does -- basis then every
# delta through the head -- checking each segment's framing/CRC, its order
# (the segment number recorded in the frame matches its position), its
# basis reference, every tensor's shape against the basis and the hidden
# slot/layer invariants.  It only ever reads: it creates no temp files,
# never advances the head and never performs a recovery.  The first bad
# segment is reported with its position and the reason; an intact chain
# returns a :class:`ChainVerification` success report.
# ---------------------------------------------------------------------------


class ChainVerification:
    """Result of a successful read-only chain verification."""

    __slots__ = ("ok", "head", "segments")

    def __init__(self, head, segments):
        self.ok = True
        self.head = head
        self.segments = segments

    def __bool__(self):
        return True

    def __repr__(self):
        return (
            f"ChainVerification(ok=True, head={self.head}, "
            f"segments={self.segments})"
        )


class ChainFamilyVerification:
    """Result of a successful read-only verification of a chain family.

    ``members`` holds the verified chain directories (as paths, in the
    order they were given) and ``chains`` the per-chain
    :class:`ChainVerification` reports in the same order.
    """

    __slots__ = ("ok", "members", "chains")

    def __init__(self, members, chains):
        self.ok = True
        self.members = members
        self.chains = chains

    def __bool__(self):
        return True

    def __repr__(self):
        heads = tuple(report.head for report in self.chains)
        return (
            f"ChainFamilyVerification(ok=True, members={len(self.members)}, "
            f"heads={heads})"
        )


def _verify_failure(position, what, exc):
    return CheckpointError(
        f"chain verification failed at {position} ({what}): {exc}"
    )


def _verify_normal_chain(store, head):
    """Validate the ordinary on-disk chain ``0..head``; read-only.

    Any defect -- including a referenced segment file that is absent -- is
    a verification failure naming the first bad segment.  The final
    assembly (completeness of the tensor space) is checked too.
    """
    try:
        basis_raw = store.read_segment(_segment_name(_BASIS_INDEX))
    except FileNotFoundError:
        raise CheckpointError(
            "chain verification failed at segment 0 (basis): "
            "the basis segment file is missing"
        ) from None
    try:
        walker = _ChainWalker(basis_raw)
    except CheckpointError as exc:
        raise _verify_failure("segment 0", "basis", exc) from exc
    for index in range(1, head + 1):
        try:
            raw = store.read_segment(_segment_name(index))
        except FileNotFoundError:
            raise CheckpointError(
                f"chain verification failed at segment {index} (delta): "
                "the segment file is missing"
            ) from None
        try:
            walker.apply_delta(index, raw)
        except CheckpointError as exc:
            raise _verify_failure(f"segment {index}", "delta", exc) from exc
    try:
        walker.document()
    except CheckpointError as exc:
        raise _verify_failure(f"segment {head}", "chain assembly", exc) from exc


def verify_chain_memory(store):
    """Verify an in-memory chain read-only; return a success report."""
    if not isinstance(store, MemoryChain):
        raise TypeError("store must be a MemoryChain")
    with store._lock:
        try:
            head = _read_head_optional(store)
        except CheckpointError as exc:
            raise _verify_failure("the head pointer", "head", exc) from exc
        if head is None:
            raise CheckpointError(
                "chain verification failed at the head pointer: the chain "
                "has no head pointer (no basis segment committed)"
            )
        _verify_normal_chain(store, head)
        return ChainVerification(head, head + 1)


def verify_chain(directory):
    """Verify an incremental chain directory without changing one byte.

    Every segment from the basis through the ``head`` pointer is checked
    for completeness (framing and CRC), segment order, the basis
    reference and tensor/layer shapes; the first bad segment's position
    and reason name the failure.  A sound chain returns a
    :class:`ChainVerification` report.  The directory is only read: no
    segment, pointer or marker is created, replaced or removed, and a
    fold interrupted on disk is inspected in place rather than rolled
    forward.

    *directory* may also be a list (or tuple) of chain directories -- a
    chain family whose members share prefix segments.  Every member is
    then verified in turn with the same read-only walk and a sound
    family returns a :class:`ChainFamilyVerification`; the first bad
    segment across the family is reported with its position, the reason
    and every chain whose head reaches it (see :func:`verify_family`).

    A missing directory raises FileNotFoundError; an operating-system
    level read failure (permissions, I/O) propagates as OSError; any
    corruption, truncation, missing field, ordering, reference or shape
    defect rejects the whole chain with ValueError whose message names
    the first bad segment and the reason.
    """
    if isinstance(directory, (list, tuple)):
        return verify_family(directory)
    if not isinstance(directory, (str, os.PathLike)):
        raise TypeError("chain directory must be a path")
    directory = os.fspath(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"incremental checkpoint directory not found: {directory!r}"
        )
    # Serialise against writers like any open, but never run recovery:
    # verification is strictly read-only.  The directory is opened
    # read-only, so a read-only directory verifies fine.
    with _DirectoryChainLock(directory):
        store = _DirectoryChainStore(directory)
        try:
            head = _read_head_optional(store)
        except CheckpointError as exc:
            raise _verify_failure("the head pointer", "head", exc) from exc
        if head is None:
            raise CheckpointError(
                "chain verification failed at the head pointer: the chain "
                "has no head pointer (no basis segment committed)"
            )
        if not _marker_exists(directory):
            _verify_normal_chain(store, head)
            return ChainVerification(head, head + 1)
        # A fold is (or was) in flight.  Verify the one coherent logical
        # chain reachable through the marker, still without writing.
        return _verify_through_marker_read_only(directory, store, head)


def verify_family(members):
    """Verify a chain family read-only; report the first bad segment.

    *members* is a non-empty sequence of chain directories whose chains
    may share prefix segments (the result of one or more forks).  Every
    member is walked exactly as :func:`verify_chain` walks a single
    chain -- completeness (framing and CRC), segment order, the basis
    reference and tensor/layer shapes -- and nothing is written: no
    chain directory changes by a single byte.  A shared segment that is
    truncated, out of order or shape-inconsistent is located at the
    first chain (in the given order) and the first segment position
    that exhibits it, and the whole family is rejected.

    A sound family returns a :class:`ChainFamilyVerification` holding
    one :class:`ChainVerification` per member.  A failure raises
    ValueError naming the owning chain (its position and path), the
    first bad segment and the reason; when the bad segment is a shared
    one, the report covers every chain whose head reaches it, never
    just one of them.  A missing member directory raises
    FileNotFoundError and an operating-system level read failure
    propagates as OSError.
    """
    if isinstance(members, (str, bytes, os.PathLike)):
        raise TypeError("a chain family must be a sequence of chain directories")
    try:
        members = list(members)
    except TypeError:
        raise TypeError(
            "a chain family must be a sequence of chain directories"
        ) from None
    if not members:
        raise CheckpointError("a chain family needs at least one chain")
    paths = []
    for member in members:
        if not isinstance(member, (str, os.PathLike)):
            raise TypeError("chain family members must be chain directory paths")
        paths.append(os.fspath(member))
    reports = []
    for position, path in enumerate(paths):
        try:
            report = verify_chain(path)
        except CheckpointError as exc:
            raise _attributed_family_error(paths, position, exc) from exc
        reports.append(report)
    return ChainFamilyVerification(tuple(paths), tuple(reports))


# Matches a genuine segment defect ("segment 3 (delta)" / "segment 0
# (basis)") in a verification failure, so the family report can attribute
# a shared segment to every chain that reaches it.
_FAMILY_SEGMENT_RE = re.compile(r"segment (\d+) \((?:basis|delta)\)")


def _attributed_family_error(paths, position, exc):
    """The family failure for member *position*, with shared-segment
    attribution covering every chain whose head reaches the bad segment.
    """
    message = (
        f"chain family verification failed at chain {position} "
        f"({paths[position]!r}): {exc}"
    )
    match = _FAMILY_SEGMENT_RE.search(str(exc))
    if match is not None:
        refs = _chains_sharing_segment(paths, position, int(match.group(1)))
        if len(refs) > 1:
            reached = ", ".join(f"chain {j} ({paths[j]!r})" for j in refs)
            message += (
                "; the segment is shared and the defect is reachable "
                f"from {reached}"
            )
    return CheckpointError(message)


def _chains_sharing_segment(paths, failed, segment_index):
    """Member indices whose head reaches *segment_index* through the same
    underlying (hard-linked) segment file as the failed chain.

    Read-only: stats the candidate files and reads each member's head
    pointer.  A member whose file is missing, different, unreadable or
    beyond its head does not reference the bad segment.
    """
    name = _segment_name(segment_index)
    failed_file = os.path.join(paths[failed], name)
    if not os.path.exists(failed_file):
        return [failed]
    refs = []
    for j, path in enumerate(paths):
        candidate = os.path.join(path, name)
        try:
            if not os.path.samefile(failed_file, candidate):
                continue
        except OSError:
            continue
        try:
            head = _read_head_optional(_DirectoryChainStore(path))
        except (CheckpointError, OSError):
            continue
        if head is not None and head >= segment_index:
            refs.append(j)
    return refs or [failed]


def _verify_through_marker_read_only(directory, store, head):
    """Read-only verification of a directory holding a compaction marker.

    No recovery is performed: the verifier assembles the one coherent
    chain the marker describes from whichever slots currently hold each
    segment, using the same slot rules the roll-forward does.  Any defect
    is reported at the segment being assembled.
    """
    with open(_marker_path(directory), "rb") as fh:
        marker = _decode_marker(fh.read())

    if "legacy_head" in marker:
        # Pre-streaming build: every new segment was staged in full.
        new_head = marker["legacy_head"]
        try:
            walker = _ChainWalker(
                _read_existing(os.path.join(directory, _staged_name(0)))
            )
            for new_index in range(1, new_head + 1):
                walker.apply_delta(
                    new_index,
                    _read_existing(
                        os.path.join(directory, _staged_name(new_index))
                    ),
                )
            walker.document()
        except FileNotFoundError:
            raise CheckpointError(
                "chain verification failed at a staged segment of the "
                "compacted chain: a required staged file is missing"
            ) from None
        return ChainVerification(new_head, new_head + 1)

    fold = marker["u"]
    if head < fold:
        raise CheckpointError(
            "chain verification failed at the compaction marker: the fold "
            f"point {fold} is beyond the chain head {head}"
        )

    if "h" not in marker:
        # Streaming phase: the staged basis plus the tail slots, with a
        # converted slot distinguished from an old one by its frame number.
        try:
            _load_through_marker(store, head, fold, None)
        except FileNotFoundError:
            raise CheckpointError(
                "chain verification failed at a segment of the folding "
                "chain: a required segment file is missing"
            ) from None
        new_head = head - fold
        return ChainVerification(new_head, new_head + 1)

    # Publication phase: basis promoted and the tail moved down ascending.
    # A source slot still present means that move has not happened yet;
    # an absent source means the segment already reached its new slot.
    new_head = marker["h"]
    walker = _ChainWalker(
        _read_existing(os.path.join(directory, _segment_name(_BASIS_INDEX)))
    )
    for new_index in range(1, new_head + 1):
        source = os.path.join(directory, _segment_name(fold + new_index))
        if os.path.exists(source):
            raw = _read_existing(source)
        else:
            raw = _read_existing(
                os.path.join(directory, _segment_name(new_index))
            )
        walker.apply_delta(new_index, raw)
    walker.document()
    return ChainVerification(new_head, new_head + 1)


def _read_existing(path):
    with open(path, "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Streaming chain compaction
#
# Instead of folding the whole chain in one long critical section, the
# fold streams:
#
#   1. the chain is walked and validated segment by segment (the same walk
#      a load does); nothing beyond one folded document is held;
#   2. the folded basis is written once to a private staged name, an
#      exclusive flock on a small lease file marks the owner alive across
#      processes, and an immutable streaming marker records the fold
#      point *u*;
#   3. each remaining tail delta is converted (renumbered against the new
#      basis) in place: old segment ``seg-(u+d)`` is read and atomically
#      replaced by new delta *d*, under the directory lock, then the lock
#      is released so appends and loads interleave;
#   4. once the stream reaches the live head it publishes, all in one
#      lock acquisition: the marker is advanced to its publish form, the
#      folded prefix is released, the staged basis takes the basis name,
#      the converted tail slots move down to their new numbers, and the
#      head advances; marker and lease are removed last.
#
# Disk footprint never exceeds the original chain plus the one staged
# basis: every converted delta occupies a slot the original chain already
# owned.  Between steps the lock is free: appends land on the still-old
# head numbering and are converted when the stream catches them, while
# loads read through the marker (staged basis + converted slots + the
# untouched old tail).  If the owner dies the lease flock is released by
# the kernel; the next open rolls the same deterministic steps forward,
# so at every instant the head names one complete chain -- the old or
# the new.
# ---------------------------------------------------------------------------


def _check_up_to(up_to, head):
    if up_to is None:
        return head
    if isinstance(up_to, bool) or not isinstance(up_to, int) or up_to < 0:
        raise CheckpointError("up_to must be a non-negative segment index")
    if up_to > head:
        raise CheckpointError(
            f"up_to segment {up_to} is beyond the chain head {head}"
        )
    return up_to


def _walk_to(store, head, stop_at):
    """Validate ``0..head`` segment by segment; return the folded doc at
    *stop_at*.  The walk holds nothing but one folded document at a time.
    """
    walker = _ChainWalker(store.read_segment(_segment_name(_BASIS_INDEX)))
    if stop_at == 0:
        return walker.document()
    folded = None
    for index in range(1, head + 1):
        walker.apply_delta(index, store.read_segment(_segment_name(index)))
        if index == stop_at:
            folded = walker.document()
    if folded is None:
        raise CheckpointError(
            f"fold point {stop_at} is not within the chain of head {head}"
        )
    return folded


def _walker_from_document(document):
    """A walker positioned on an already-folded full state document."""
    walker = _ChainWalker.__new__(_ChainWalker)
    walker.basis = document
    walker.param_count = len(document["params"])
    walker.tensors = _tensors_from_document(document)
    walker.hidden_count = (
        len(document["hidden"]) if document["hidden"] is not None else None
    )
    walker.pending = document["pending"]
    walker.hidden_base = _flat_hidden_base(walker.param_count)
    return walker


def _delta_wire_number(raw):
    """The segment number recorded inside a delta frame."""
    _version, header, _payload, _leaves = _read_frame(
        bytes(raw), DELTA_MAGIC, DELTA_END_MAGIC, "delta segment"
    )
    number = header["n"]
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise CheckpointError("delta segment number must be a positive int")
    return number


def compact_chain(directory, up_to=None):
    """Fold the basis and the deltas through *up_to* into one new basis.

    The fold is a streaming online compaction: segments are checked one
    by one as the new basis is written, each tail delta is converted in
    the slot the original chain already owned, and saves and loads keep
    working through the whole run.  Peak on-disk footprint stays within
    the original chain plus the one staged basis, and at every instant
    the head names one complete chain -- the old or the new; a process
    killed at any point is rolled forward on the next open.

    The reassembled state (parameters, gradients, optimizer moments and
    step count, hidden state) is bit for bit identical, compaction
    advances no optimizer step, and the segment count drops by exactly the
    merged range.  Nothing to merge (no chain, basis only, or
    ``up_to=0``) leaves the directory untouched.  Old-version segments
    participate exactly as on load and the compacted chain is stored in
    the current format version.

    A missing directory raises FileNotFoundError; an unwritable directory
    or a full disk raises OSError; any corrupt, truncated or inconsistent
    segment rejects the compaction with ValueError before the chain is
    touched.
    """
    if not isinstance(directory, (str, os.PathLike)):
        raise TypeError("chain directory must be a path")
    directory = os.fspath(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"incremental checkpoint directory not found: {directory!r}"
        )
    # Deterministic GC: staging directories killed forks/deletes left in
    # the family directory are reclaimed on every compaction.
    _sweep_parent_staging(os.path.dirname(os.path.abspath(directory)))

    with _FamilyFoldGuard(directory):
        return _compact_chain_locked(directory, up_to)


def _compact_chain_locked(directory, up_to):
    """The body of :func:`compact_chain`, already holding the family guard."""
    # Claim the fold -- waiting out a fold another process/thread may
    # already run -- then validate and stage.  The lease is taken inside
    # the directory lock so two compactors can never both stage a basis.
    lease_fd = None
    try:
        while True:
            with _DirectoryChainLock(directory):
                store = _DirectoryChainStore(directory)
                _recover_directory_chain(directory, store)
                if _member_fold_busy(directory):
                    contended = True
                else:
                    lease_fd = _try_compaction_lease(directory)
                    contended = lease_fd is None and fcntl is not None
                if contended:
                    lease_fd = None
            if not contended:
                break
            _wait_for_live_marker(directory)

        with _DirectoryChainLock(directory):
            store = _DirectoryChainStore(directory)
            head = _read_head_optional(store)
            if head is None:
                raise CheckpointError(
                    "chain has no head pointer (no basis segment committed)"
                )
            # Deterministic GC: segments beyond the head are unreachable
            # crash residue and are swept on every compaction.
            _sweep_chain_debris(directory, head)
            fold = _check_up_to(up_to, head)
            if fold == 0 or head == 0:
                _release_compaction_lease(lease_fd)
                _unlink_quietly(_lease_path(directory))
                return None
            # Validate the entire chain before anything is written; the
            # folded document becomes the staged basis.
            folded_doc = _walk_to(store, head, fold)
            folded_bytes = build_bytes(folded_doc)

        # Publish the staged basis and the immutable streaming marker.
        # Recovery only ever follows a marker whose lease is dead.
        with _DirectoryChainLock(directory):
            store = _DirectoryChainStore(directory)
            _atomic_write(directory, _staged_name(0), folded_bytes)
            _atomic_write(directory, _COMPACT_MARKER, _encode_marker({"u": fold}))
            _lease_register(directory)
            _fsync_directory(directory)

        _stream_tail(directory, fold, folded_doc)
    except BaseException:
        # A live caller giving up (a full disk, say) releases the lease;
        # the next open treats the marker like any interrupted fold and
        # rolls it forward.
        _release_compaction_lease(lease_fd)
        _lease_release(directory)
        raise
    _release_compaction_lease(lease_fd)
    _lease_release(directory)
    return None


def _try_compaction_lease(directory):
    """Take the cross-process lease non-blockingly.

    Returns the locked fd, or ``None`` when another live process holds
    it.  On platforms without ``fcntl`` returns a sentry fd-less "lease"
    (``False`` is distinct from ``None`` so callers still enter the fold;
    cross-process exclusion there is best effort by the directory lock).
    """
    if fcntl is None:
        return False
    path = _lease_path(directory)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            return None
        raise
    return fd


def _release_compaction_lease(fd):
    if not fd:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    os.close(fd)


def _stream_tail(directory, fold, folded_doc):
    """Convert tail deltas one per lock acquisition until published.

    The live head may grow while the stream runs (a concurrent append);
    every newly appended old-numbered segment is converted in turn before
    publication.  The owner keeps the folded walker in memory, so each
    step folds exactly one new segment (a process resuming a dead owner
    rebuilds that state from disk via :func:`_fold_walker`).
    """
    walker = _walker_from_document(folded_doc)
    previous = folded_doc
    done = 0
    while True:
        with _DirectoryChainLock(directory):
            store = _DirectoryChainStore(directory)
            if not _marker_exists(directory):
                return
            head = _read_head_optional(store)
            next_new = done + 1
            next_old = fold + next_new
            if next_old <= head:
                raw = store.read_segment(_segment_name(next_old))
                walker.apply_delta(next_new, raw, wire_number=next_old)
                current = walker.document()
                delta = _build_delta_between(previous, current, next_new)
                _atomic_write(directory, _segment_name(next_old), delta)
                previous = current
                done = next_new
                continue
            # Caught the live head while holding the lock, so no append
            # can land between the checks: publish now.
            _publish_compaction(directory, store, fold, head - fold)
            return


def _publish_compaction(directory, store, fold, new_head):
    """Switch point: promote the folded chain, advance head, clear marker.

    Runs entirely under the directory lock, so no append can interleave;
    other processes either see the complete old chain (head still old) or
    the complete new chain (head advanced).  The publish marker is committed
    first, making the destructive phase recognisable and resumable after a
    kill: folded prefix release, basis promotion, then the tail rotation
    (each target move is complete exactly when its source is gone).
    """
    _atomic_write(
        directory, _COMPACT_MARKER, _encode_marker({"u": fold, "h": new_head})
    )
    for index in range(1, fold + 1):
        _unlink_quietly(os.path.join(directory, _segment_name(index)))
    staged_path = os.path.join(directory, _staged_name(0))
    if os.path.exists(staged_path):
        with open(staged_path, "rb") as fh:
            staged_basis = fh.read()
        _atomic_write(directory, _segment_name(_BASIS_INDEX), staged_basis)
        _unlink_quietly(staged_path)
    _promote_tail_slots(directory, fold, new_head)
    for name in list(os.listdir(directory)):
        index = _segment_index(name)
        if index is not None and index > new_head:
            _unlink_quietly(os.path.join(directory, name))
    _commit_head(store, new_head)
    _unlink_quietly(os.path.join(directory, _staged_name(0)))
    _unlink_quietly(_marker_path(directory))
    _unlink_quietly(_lease_path(directory))
    _fsync_directory(directory)


def _promote_tail_slots(directory, fold, new_head):
    """Move converted tail deltas down to their new slots, ascending.

    Source ``seg-(fold+d)`` is read, published atomically (temp file plus
    rename) at ``seg-d`` and only then removed.  Ascending order makes this
    an in-place rotation: slot *d* is a released prefix slot when
    ``d <= fold`` and the source freed by an earlier move afterwards.
    Resumption is driven by source presence -- a missing source means that
    move committed -- never by frame contents, which are ambiguous while
    prefix slots linger.
    """
    for new_index in range(1, new_head + 1):
        source = os.path.join(directory, _segment_name(fold + new_index))
        target = _segment_name(new_index)
        if not os.path.exists(source):
            if not os.path.exists(os.path.join(directory, target)):
                raise CheckpointError(
                    "compaction publication lost both the source and the "
                    f"target of new segment {new_index}"
                )
            continue
        with open(source, "rb") as fh:
            raw = fh.read()
        _atomic_write(directory, target, raw)
        _unlink_quietly(source)


def compact_chain_memory(store, up_to=None):
    """Compact an in-memory chain; same semantics as :func:`compact_chain`.

    The same streaming fold drives it (one segment converted per step),
    but a memory chain needs neither durable markers nor lock handoffs:
    the store lock is re-entrant and held for the whole call, so the
    conversion happens in one critical section with identical results.
    """
    if not isinstance(store, MemoryChain):
        raise TypeError("store must be a MemoryChain")
    with store._lock:
        head = _read_head_optional(store)
        if head is None:
            raise CheckpointError(
                "chain has no head pointer (no basis segment committed)"
            )
        fold = _check_up_to(up_to, head)
        if fold == 0 or head == 0:
            return None
        folded_doc = _walk_to(store, head, fold)

        staged = {0: build_bytes(folded_doc)}
        walker = _walker_from_document(folded_doc)
        previous = folded_doc
        for old_index in range(fold + 1, head + 1):
            new_index = old_index - fold
            raw = store.read_segment(_segment_name(old_index))
            walker.apply_delta(new_index, raw, wire_number=old_index)
            current = walker.document()
            staged[new_index] = _build_delta_between(
                previous, current, new_index
            )
            previous = current

        new_head = head - fold
        for index, raw in staged.items():
            store.write_segment(_segment_name(index), raw)
        for name in list(store._objects):
            index = _segment_index(name)
            if index is not None and index > new_head:
                del store._objects[name]
        _commit_head(store, new_head)
    return None


# ---------------------------------------------------------------------------
# Family-level compaction
#
# A family compaction folds, in one call, the prefix segments every member
# chain of a family reaches through the *same underlying segment files*
# (the hard-linked prefix a fork creates) into one new basis segment
# shared by every member.  Each member's head then points at this new
# history; the tail segments the member alone owned follow the new basis
# in their original order and byte for byte unchanged.
#
# The family operation invents no per-chain protocol: every member runs
# the ordinary streaming fold (its own staged basis, marker and lease),
# so saves, loads, read-only verification and crash recovery all keep
# working unchanged.  The family-level pieces are:
#
# * one advisory flock (``.seqfam.lock``) plus an in-process registry,
#   held for the whole run so family operations in the family directory
#   serialise and a killed owner is recognisable;
# * one marker file (``.seqfamily``) recording the fold point, the
#   shared staged-basis name and the member paths, so a dead fold can be
#   resumed deterministically;
# * the folded basis, written once to a ``.seqfam.tmp-*`` file in the
#   family directory and hard-linked into each member's staged-basis
#   slot, so the new basis physically exists once;
# * link-preserving publication: each member's staged basis link is
#   atomically ``rename``d over its basis slot (rather than copied), so
#   every member's new basis is the one shared inode.
#
# A process killed at any point leaves every member one complete
# loadable chain: each member's own dead marker is rolled forward by the
# member's next open exactly as for a single-chain fold, and the next
# family operation rolls the family marker forward (finishing members
# the kill reached, reclaiming the parent staging file) so no reachable
# segment is lost and no unreachable residue leaks.
# ---------------------------------------------------------------------------


_FAMILY_MARKER_NAME = ".seqfamily"
# Per-member family-fold barrier.  While a family fold is finishing the
# remaining members, an already-promoted member keeps this marker (plus
# its compaction lease flock): its new chain is fully on disk (head
# advanced, slots renumbered, basis shared), so saves and loads use it
# normally, but a new single-chain compaction, fork or deletion waits for
# the family fold to finish rather than renumbering slots the family is
# still working from.
_FAMILY_BARRIER_NAME = ".seqfc"
_FAMILY_MARKER_KEYS = frozenset(("v", "u", "b", "m", "d"))


def _family_marker_path(parent):
    return os.path.join(parent, _FAMILY_MARKER_NAME)


def _common_family_prefix(members, heads):
    """Largest *u* <= every head such that seg-0..seg-u are one physical
    file shared (hard-linked) by every member.

    Read-only: a missing or differing file simply ends the common run;
    the members are fully validated separately.
    """
    u = 0
    limit = min(heads)
    while u <= limit:
        name = _segment_name(u)
        anchor = os.path.join(members[0], name)
        if not os.path.exists(anchor):
            return u - 1
        common = True
        for directory in members[1:]:
            candidate = os.path.join(directory, name)
            try:
                if not os.path.samefile(anchor, candidate):
                    common = False
                    break
            except OSError:
                common = False
                break
        if not common:
            return u - 1
        u += 1
    return limit


def _family_shape_signature(document):
    return (
        [list(entry["s"]) for entry in document["params"]],
        [
            (layer["kind"], [list(shape) for shape in layer["shapes"]])
            for layer in document["layers"]
        ],
    )


def _format_family_shape_mismatch(signatures, members):
    """A rejection message naming the shapes and layer order that disagree."""
    reference_shapes, reference_layers = signatures[0]
    lines = [
        "family compaction rejected: the members' parameter shapes or "
        "layer order do not agree"
    ]
    lines.append(
        f"chain 0 ({members[0]!r}) fixes parameter shapes "
        f"{reference_shapes!r} and layer order {reference_layers!r}"
    )
    for index, (shapes, layers) in enumerate(signatures[1:], start=1):
        if shapes != reference_shapes:
            lines.append(
                f"chain {index} ({members[index]!r}) fixes parameter shapes "
                f"{shapes!r}"
            )
        if layers != reference_layers:
            lines.append(
                f"chain {index} ({members[index]!r}) fixes layer order "
                f"{layers!r}"
            )
    return "; ".join(lines)


def _wait_for_member_quiescent(directory):
    """Wait out a live fold or family-fold barrier over one member.

    A dead streaming fold is rolled forward first; a dead barrier is
    reaped (its chain is already complete).  A live family fold -- a
    member mid-conversion, or an already-promoted member still fenced by
    its family barrier while the other members finish -- is waited out.
    """
    while True:
        with _DirectoryChainLock(directory):
            store = _DirectoryChainStore(directory)
            _recover_directory_chain(directory, store)
            if not _member_fold_busy(directory):
                return
        _wait_for_live_marker(directory)


def compact_family(members):
    """Fold the prefix shared by every member of a chain family into one
    new basis segment.

    *members* is a non-empty sequence of chain directories belonging to
    one family (the result of one or more forks) and living in one parent
    directory.  The prefix segments all members reach through the same
    underlying (hard-linked) files are merged into a single new basis
    segment stored once and shared by every member; each member's head is
    advanced to the new history and the tail segments it alone owned
    follow the new basis in their original order.  The state each member
    reassembles to -- parameters, gradients, optimizer moments and step
    count, hidden state -- is bit for bit unchanged, the fold advances no
    optimizer step, and every member's segment count drops deterministically
    by exactly the folded range.  Repeating the same fold changes nothing;
    when the family has no foldable prefix the member chains are left as
    they are (their storage is still unified when their basis files are
    already byte-identical).

    Saves, loads and incremental appends proceed on every member while
    the fold runs; at every instant each member's head names one complete
    chain, and the fold serialises with forks, merges, deletions and
    single-chain compactions through the directory locks and leases.  A
    process killed mid-fold leaves every member loadable; the residue is
    reclaimed deterministically by the next family operation, and shared
    segments are neither deleted while reachable nor leaked once
    unreachable.

    A missing member directory or a missing referenced segment raises
    FileNotFoundError and leaves every other member untouched.  A
    truncated segment, a missing field, an out-of-order segment, or
    members whose parameter shapes or layer order disagree reject the
    whole family fold with ValueError before anything is written; the
    message names the disagreeing shapes and layer order.  An unwritable
    directory or a full disk raises OSError.
    """
    if isinstance(members, (str, bytes, os.PathLike)):
        raise TypeError("a chain family must be a sequence of chain directories")
    try:
        members = list(members)
    except TypeError:
        raise TypeError(
            "a chain family must be a sequence of chain directories"
        ) from None
    if not members:
        raise CheckpointError("a chain family needs at least one chain")
    paths = []
    for member in members:
        if not isinstance(member, (str, os.PathLike)):
            raise TypeError("chain family members must be chain directory paths")
        paths.append(os.fspath(member))
    if len(paths) == 1:
        return compact_chain(paths[0])

    for path in paths:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"incremental checkpoint directory not found: {path!r}"
            )
    abs_paths = [os.path.abspath(path) for path in paths]
    if len(set(abs_paths)) != len(abs_paths):
        raise CheckpointError("a chain family must not list a member twice")
    parents = {os.path.dirname(path) for path in abs_paths}
    if len(parents) != 1:
        raise CheckpointError(
            "a family compaction requires every member directory to live "
            "in the same parent directory"
        )
    parent = parents.pop()
    _sweep_parent_staging(parent)

    lease_fd = None
    try:
        while True:
            with _family_dir_lock(parent):
                if _family_lease_live(parent):
                    contended = True
                else:
                    lease_fd = _try_family_lease(parent)
                    contended = lease_fd is None and fcntl is not None
            if not contended:
                break
            _wait_for_family_lease(parent)
        _family_register(parent)

        # A previous owner killed mid-fold is rolled forward first; its
        # completed fold then simply makes this call a no-op.
        if os.path.exists(_family_marker_path(parent)):
            _resume_family_compaction(parent)
        _sweep_family_residue(parent)

        # Measure the family and install every member's fold marker and
        # staged-basis link before any tail is converted.  All markers go
        # in first so a single-chain compaction of a member not yet
        # converted waits out the family fold rather than renumbering the
        # family's fold point underneath it.
        plan = _measure_and_stage_family(parent, abs_paths)
        if plan is None:
            # No foldable prefix: only storage may need reunifying; the
            # family lease for this (no-op) call is reclaimed too.
            _converge_identical_bases(abs_paths)
            _unlink_quietly(_family_lease_path(parent))
            _sweep_family_residue(parent)
            return None
        fold, folded_doc, family_staged, lease_fds = plan

        published = set()
        try:
            done = []
            for index, directory in enumerate(abs_paths):
                if not os.path.isdir(directory):
                    # Concurrently deleted: its own delete waits out the
                    # member marker first, so a vanished directory is
                    # simply no longer part of the family.
                    continue
                _stream_family_tail(directory, fold, folded_doc, family_staged)
                # The member is published but keeps its family barrier and
                # lease until the whole family finishes; keep its in
                # process registry entry and lease fd as well.
                published.add(directory)
                done.append(index)
                _atomic_write(
                    parent,
                    _FAMILY_MARKER_NAME,
                    _encode_family_marker(
                        {
                            "v": 1,
                            "u": fold,
                            "b": os.path.basename(family_staged),
                            "m": abs_paths,
                            "d": done,
                        }
                    ),
                )

            _finish_family_compaction(parent, family_staged, abs_paths)
        finally:
            # Release every member lease: on success the barriers are
            # already gone; on a failure/kill a still-unpublished member
            # keeps its streaming marker and a published one its barrier,
            # both now with a dead lease, so each member's next open
            # either rolls its fold forward or reaps the barrier, and the
            # next family operation converges the family.
            for directory in lease_fds:
                _lease_release(directory)
            for fd in lease_fds.values():
                _release_compaction_lease(fd)
    finally:
        # The family flock is released on every path (a killed owner's is
        # released by the kernel); dropping the in-process registry entry
        # here is what lets later family calls in this process proceed.
        _release_family_lease(lease_fd)
        _family_release(parent)
    return None


def _encode_family_marker(marker):
    return json.dumps(marker, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )


def _decode_family_marker(raw):
    try:
        marker = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("family compaction marker is corrupt") from exc
    if not isinstance(marker, dict) or set(marker) != _FAMILY_MARKER_KEYS:
        raise CheckpointError("family compaction marker is corrupt")
    if marker["v"] != 1:
        raise CheckpointError("family compaction marker is corrupt")
    u = marker["u"]
    if isinstance(u, bool) or not isinstance(u, int) or u <= 0:
        raise CheckpointError("family compaction marker is corrupt")
    basis_name = marker["b"]
    if (
        not isinstance(basis_name, str)
        or not basis_name
        or os.path.basename(basis_name) != basis_name
        or not basis_name.startswith(_FAMILY_TMP_PREFIX)
    ):
        raise CheckpointError("family compaction marker is corrupt")
    members = marker["m"]
    if not isinstance(members, list) or not members or any(
        not isinstance(item, str) or not item for item in members
    ):
        raise CheckpointError("family compaction marker is corrupt")
    done = marker["d"]
    if not isinstance(done, list) or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= len(members)
        for value in done
    ) or len(set(done)) != len(done):
        raise CheckpointError("family compaction marker is corrupt")
    return marker


def _measure_and_stage_family(parent, members):
    """Validate and measure the family, then stage every member's fold.

    Returns ``(fold, folded_doc, family_staged, lease_fds)`` where
    *lease_fds* maps each member directory to its held per-chain lease
    fd, or ``None`` when the family has no foldable prefix.

    Each round takes every member directory lock together (in sorted
    path order, a deadlock-free total order).  If any member carries a
    live fold marker, or a member's compaction lease cannot be taken
    non-blockingly, *all* locks are dropped first and the caller waits
    out that fold outside any member lock -- this thread therefore
    never blocks on a member lock or lease while holding another
    member's lock, so it cannot deadlock against a streaming
    single-chain compaction (which converts one member a lock-step at a
    time).  Once a quiet round claims every lease, the family is
    measured, validated and staged -- markers and staged-basis links for
    every member -- all under that one set of locks, after which no
    save, fork, merge, deletion or single-chain compaction can move a
    member before its marker names the family fold point.

    On the ``None`` (no-foldable-prefix) path every lease is released
    and no registry entry remains.  A missing referenced segment raises
    FileNotFoundError; a framing, ordering or shape defect raises
    CheckpointError, both before any persistent fold state exists (only
    the one shared staged basis and the markers are written, after
    validation succeeds).
    """
    ordered = sorted(members)
    lease_fds = {}
    staged_fold = None
    while True:
        contended = []
        with contextlib.ExitStack() as stack:
            for directory in ordered:
                stack.enter_context(_DirectoryChainLock(directory))
            stores = {d: _DirectoryChainStore(d) for d in ordered}
            for directory in ordered:
                _recover_directory_chain(directory, stores[directory])
            live = [d for d in ordered if _member_fold_busy(d)]
            if live:
                # A fold or a live family-fold barrier is in force on a
                # member.  Drop every lock first and wait outside any
                # member lock, then retry the whole round -- never wait
                # while holding another member's lock.
                contended = live
            else:
                for directory in ordered:
                    fd = _try_compaction_lease(directory)
                    if fd is None and fcntl is not None:
                        contended.append(directory)
                    else:
                        lease_fds[directory] = fd
                if not contended:
                    # Every lease is held and every lock is held in one
                    # critical section: measure, validate and install the
                    # markers/links before any lock is released, so no
                    # save, fork, merge, deletion or single-chain
                    # compaction can move a member in between.
                    try:
                        staged_fold = _measure_validate_stage_locked(
                            parent, members, ordered, stores, lease_fds
                        )
                    except BaseException:
                        # Detach the claimed leases so the partially
                        # installed fold is seen as dead (and rolled
                        # forward) rather than live, and remove the lease
                        # files this rejected attempt created, so a
                        # rejected family fold leaves no member touched.
                        _drop_member_leases(lease_fds)
                        raise
                    if staged_fold is None:
                        _drop_member_leases(lease_fds)
                        lease_fds = {}
                else:
                    for fd in lease_fds.values():
                        _release_compaction_lease(fd)
                    lease_fds = {}
        if contended:
            for directory in contended:
                _wait_for_live_marker(directory)
            continue
        break

    if staged_fold is None:
        # No foldable prefix.
        return None
    fold, folded_doc, family_staged = staged_fold
    return fold, folded_doc, family_staged, lease_fds


def _drop_member_leases(lease_fds):
    """Release every member lease and remove its lease file.

    Used only for a family fold that never published (a validation
    rejection or a no-foldable-prefix result): the lease file is unlinked
    while the flock is still held (a later opener then creates a fresh
    inode and lock), and the in-process registry entry is dropped, so the
    rejected attempt leaves no file in any member directory.
    """
    for directory, fd in lease_fds.items():
        _unlink_quietly(_lease_path(directory))
        _lease_release(directory)
        _release_compaction_lease(fd)


def _measure_validate_stage_locked(
    parent, members, ordered, stores, lease_fds
):
    """Measure, validate and stage the family while all member locks are held.

    Returns ``(fold, folded_doc, family_staged)`` or ``None`` when there
    is no foldable prefix.  Runs after every member lease was claimed and
    while every member directory lock is held, so the on-disk family
    cannot move underneath it.  The caller releases the claimed leases if
    this raises.
    """
    heads = {}
    documents = {}
    for directory in ordered:
        store = stores[directory]
        head = _read_head_optional(store)
        if head is None:
            raise CheckpointError(
                "chain has no head pointer (no basis segment "
                f"committed): {directory!r}"
            )
        _sweep_chain_debris(directory, head)
        # Missing referenced segment -> FileNotFoundError; any
        # framing/order/shape defect -> CheckpointError.
        documents[directory] = _load_chain_store(store, head)
        heads[directory] = head

    member_docs = [documents[d] for d in members]
    signatures = [_family_shape_signature(doc) for doc in member_docs]
    reference = signatures[0]
    if any(signature != reference for signature in signatures[1:]):
        raise CheckpointError(
            _format_family_shape_mismatch(signatures, members)
        )

    member_heads = [heads[d] for d in members]
    fold = _common_family_prefix(members, member_heads)
    if fold < 1:
        return None

    folded_doc = _walk_to(stores[members[0]], heads[members[0]], fold)
    family_staged = _family_stage_basis(parent, build_bytes(folded_doc))
    _atomic_write(
        parent,
        _FAMILY_MARKER_NAME,
        _encode_family_marker(
            {
                "v": 1,
                "u": fold,
                "b": os.path.basename(family_staged),
                "m": members,
                "d": [],
            }
        ),
    )
    for directory in ordered:
        staged_link = os.path.join(directory, _staged_name(0))
        if not os.path.exists(staged_link):
            os.link(family_staged, staged_link)
        _atomic_write(
            directory, _COMPACT_MARKER, _encode_marker({"u": fold})
        )
        _lease_register(directory)
    _fsync_directory(parent)
    return fold, folded_doc, family_staged


def _stream_family_tail(directory, fold, folded_doc, family_staged):
    """Convert one member's tail a segment at a time, then publish links.

    Identical in shape to :func:`_stream_tail`: each conversion runs
    under the member directory lock, so an append on the still-old
    numbering is converted in turn before publication.
    """
    walker = _walker_from_document(folded_doc)
    previous = folded_doc
    done = 0
    while True:
        with _DirectoryChainLock(directory):
            store = _DirectoryChainStore(directory)
            if not _marker_exists(directory):
                return
            head = _read_head_optional(store)
            next_new = done + 1
            next_old = fold + next_new
            if next_old <= head:
                raw = store.read_segment(_segment_name(next_old))
                walker.apply_delta(next_new, raw, wire_number=next_old)
                current = walker.document()
                delta = _build_delta_between(previous, current, next_new)
                _atomic_write(directory, _segment_name(next_old), delta)
                previous = current
                done = next_new
                continue
            # Caught the live head under the lock: publish now.
            _publish_compaction_shared(
                directory, store, fold, head - fold, family_staged
            )
            return


def _shared_basis_link(directory, source_path):
    """A fresh name in *directory* hard-linked to the shared basis file."""
    fd, tmp_path = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=directory)
    os.close(fd)
    os.unlink(tmp_path)
    os.link(source_path, tmp_path)
    return tmp_path


def _publish_compaction_shared(directory, store, fold, new_head, family_staged):
    """Publish one folded member keeping its basis on the shared inode.

    Mirrors :func:`_publish_compaction`, except the basis is installed by
    atomically renaming the staged link (itself a hard link to the one
    family staged file) over the basis slot, so every member's new basis
    is one physical file.  The member's head is advanced and its chain is
    complete, but its streaming marker is replaced by a family-fold
    barrier (and the compaction lease is deliberately kept) until the
    whole family fold finishes, so a fresh single-chain compaction, fork
    or deletion cannot start on this member while the family operation is
    still converting the others.  Saves, loads and verifies see only the
    completed new chain.
    """
    _atomic_write(
        directory, _COMPACT_MARKER, _encode_marker({"u": fold, "h": new_head})
    )
    for index in range(1, fold + 1):
        _unlink_quietly(os.path.join(directory, _segment_name(index)))
    staged_link = os.path.join(directory, _staged_name(0))
    os.replace(staged_link, os.path.join(directory, _segment_name(_BASIS_INDEX)))
    _promote_tail_slots(directory, fold, new_head)
    for name in list(os.listdir(directory)):
        index = _segment_index(name)
        if index is not None and index > new_head:
            _unlink_quietly(os.path.join(directory, name))
    _commit_head(store, new_head)
    # Swap the streaming marker for the family barrier; keep the lease
    # (file, flock and registry) until the family fold completes.
    _atomic_write(
        directory,
        _FAMILY_BARRIER_NAME,
        _encode_marker({"u": fold, "h": new_head}),
    )
    _unlink_quietly(_marker_path(directory))
    _fsync_directory(directory)


def _family_barrier_path(directory):
    return os.path.join(directory, _FAMILY_BARRIER_NAME)


def _family_barrier_exists(directory):
    return os.path.exists(_family_barrier_path(directory))


def _member_fold_busy(directory):
    """Whether a writer (compact/fork/delete) must wait on this member.

    Either a streaming fold marker is in force, or a family-fold barrier
    held by a live owner is.  A barrier whose lease is dead (the family
    owner was killed) does not block; the caller reaps it.
    """
    if _marker_exists(directory):
        return True
    return _family_barrier_exists(directory) and _marker_live(directory)


def _reap_dead_family_barrier(directory):
    """Remove a family barrier whose owner is dead.  Caller holds the lock."""
    barrier = _family_barrier_path(directory)
    if os.path.exists(barrier) and not _marker_live(directory):
        _unlink_quietly(barrier)
        _unlink_quietly(_lease_path(directory))


def _finish_family_compaction(parent, family_staged, members):
    """Settle every member, then remove barriers and the family marker.

    Runs while the family owner still holds every member lease.  Each
    member's barrier is removed under its own directory lock; the chain
    underneath is the completed folded one.  Members whose own dead-marker
    recovery copied the basis (rather than linking it) are then reunified
    onto one inode, byte for byte; the family marker, the one shared
    staging file and the family lease are removed last.
    """
    surviving = []
    for directory in members:
        if not os.path.isdir(directory):
            continue
        surviving.append(directory)
        with _DirectoryChainLock(directory):
            store = _DirectoryChainStore(directory)
            # No streaming marker remains here (each member reached its
            # publish barrier); ordinary debris is swept defensively.
            _recover_directory_chain(directory, store)
            _unlink_quietly(_family_barrier_path(directory))
            _unlink_quietly(_lease_path(directory))
            _fsync_directory(directory)
    _converge_identical_bases(surviving)
    _unlink_quietly(_family_marker_path(parent))
    _unlink_quietly(family_staged)
    _unlink_quietly(_family_lease_path(parent))
    _sweep_family_residue(parent)
    _fsync_directory(parent)

def _sweep_family_residue(parent):
    """Remove family staging files a dead fold left in *parent*.

    Garbage collection only: the one staging file a live family marker
    names is never swept by the fold's owner, and operating-system
    failures never reach the caller.  A staging file whose owner is dead
    (no marker names it) is unreachable residue.
    """
    changed = False
    try:
        names = os.listdir(parent)
    except OSError:
        return
    live_basis = None
    marker_path = _family_marker_path(parent)
    if os.path.exists(marker_path):
        try:
            with open(marker_path, "rb") as fh:
                live_basis = _decode_family_marker(fh.read())["b"]
        except CheckpointError:
            live_basis = None
    for name in names:
        if not name.startswith(_FAMILY_TMP_PREFIX):
            continue
        if name == live_basis:
            continue
        _unlink_quietly(os.path.join(parent, name))
        changed = True
    if changed:
        _fsync_directory(parent)


def _converge_identical_bases(members):
    """Unify members' basis files onto one inode when bytes agree.

    Pure storage housekeeping: no head, slot or file byte changes -- only
    a hard link replaces a byte-identical basis, restoring the one-copy
    storage the fork and the family fold maintain.
    """
    if not members:
        return
    anchor = os.path.join(members[0], _segment_name(_BASIS_INDEX))
    try:
        with open(anchor, "rb") as fh:
            anchor_bytes = fh.read()
    except OSError:
        return
    for directory in members[1:]:
        basis = os.path.join(directory, _segment_name(_BASIS_INDEX))
        try:
            if os.path.samefile(anchor, basis):
                continue
            with open(basis, "rb") as fh:
                if fh.read() != anchor_bytes:
                    continue
            with _DirectoryChainLock(directory):
                link_path = _shared_basis_link(directory, anchor)
                os.replace(link_path, basis)
                _fsync_directory(directory)
        except OSError:
            continue


def _resume_family_compaction(parent):
    """Roll a dead family fold forward to completion.

    Called with the family lease held after the owner died.  Every
    surviving member first runs its ordinary single-chain recovery,
    which rolls a dead per-member fold forward from the member's staged
    basis; members the kill never reached (no marker, old prefix still
    present) are folded now from the shared staged basis.  The shared
    basis file is rebuilt from an unfolded member if its parent name was
    lost, and at the end members' basis files -- which single-chain
    recovery may have copied rather than linked -- are reunified onto
    one inode, so the folded basis stays stored once.  Pure
    reachability: running the resume again changes nothing.
    """
    marker_path = _family_marker_path(parent)
    with open(marker_path, "rb") as fh:
        marker = _decode_family_marker(fh.read())
    fold = marker["u"]
    members = marker["m"]
    family_staged = os.path.join(parent, marker["b"])
    surviving = [d for d in members if os.path.isdir(d)]

    # Roll every member's own dead (or, on another thread, live) fold.
    for directory in surviving:
        _wait_for_member_quiescent(directory)

    # Whether a member was folded is settled by its basis bytes: the
    # folded basis is exactly the state at the family fold point, so a
    # member standing on those bytes is folded (its seg-0..seg-fold-1
    # prefix slots are gone), regardless of how long its new tail is.
    if not os.path.exists(family_staged):
        rebuilt = None
        for directory in surviving:
            try:
                with open(
                    os.path.join(directory, _segment_name(_BASIS_INDEX)), "rb"
                ) as fh:
                    basis_bytes = fh.read()
                with _DirectoryChainLock(directory):
                    store = _DirectoryChainStore(directory)
                    head = _read_head_optional(store)
                # An unfolded member still carries the old numbered
                # prefix: walking to its recorded fold point rebuilds the
                # exact shared folded basis.
                if head is not None and head >= fold:
                    candidate = build_bytes(_walk_to(store, head, fold))
                    if candidate != basis_bytes:
                        rebuilt = candidate
                        break
            except (CheckpointError, FileNotFoundError, OSError):
                continue
        if rebuilt is None:
            # Every member was already folded by its own opens; only
            # storage reunification and cleanup remain.
            _converge_identical_bases(surviving)
            _unlink_quietly(marker_path)
            _unlink_quietly(_family_lease_path(parent))
            _sweep_family_residue(parent)
            _fsync_directory(parent)
            return
        _atomic_write(parent, marker["b"], rebuilt)

    with open(family_staged, "rb") as fh:
        folded_raw = fh.read()
    folded_doc = parse_bytes(folded_raw)

    for directory in surviving:
        try:
            with open(
                os.path.join(directory, _segment_name(_BASIS_INDEX)), "rb"
            ) as fh:
                basis_bytes = fh.read()
        except OSError:
            continue
        if basis_bytes == folded_raw:
            continue  # already folded (by the live owner or its own open)
        # Re-check under the member lock, then fold without holding it.
        with _DirectoryChainLock(directory):
            store = _DirectoryChainStore(directory)
            head = _read_head_optional(store)
            still_unfolded = (
                head is not None
                and head >= fold
                and not _marker_exists(directory)
            )
        if still_unfolded:
            _fold_one_resumed_member(
                directory, fold, folded_doc, family_staged
            )

    _finish_family_compaction(parent, family_staged, members)


def _fold_one_resumed_member(directory, fold, folded_doc, family_staged):
    """Stream-fold one member during a family resume.

    Runs after the member was observed unfolded (head reaches seg-fold)
    and quiescent; installs its staged basis link and marker under the
    member lock, then converts its tail and publishes on the shared
    inode via the ordinary streaming loop.
    """
    lease_fd = None
    try:
        with _DirectoryChainLock(directory):
            lease_fd = _try_compaction_lease(directory)
            if lease_fd is None and fcntl is not None:
                raise CheckpointError(
                    "family compaction resume could not claim the member "
                    f"compaction lease: {directory!r}"
                )
            staged_link = os.path.join(directory, _staged_name(0))
            if not os.path.exists(staged_link):
                os.link(family_staged, staged_link)
            _atomic_write(directory, _COMPACT_MARKER, _encode_marker({"u": fold}))
            _lease_register(directory)
            _fsync_directory(directory)
        _stream_family_tail(directory, fold, folded_doc, family_staged)
    except BaseException:
        _release_compaction_lease(lease_fd)
        _lease_release(directory)
        raise
    _release_compaction_lease(lease_fd)


def compact_family_memory(members):
    """Fold the common shared prefix of a family of in-memory chains.

    Same semantics as :func:`compact_family` for :class:`MemoryChain`
    members.  Memory chains share a prefix by holding the very same
    segment ``bytes`` objects (the sharing :func:`fork_chain_memory`
    creates); the folded basis is likewise one object referenced by
    every member, and each member's own tail is renumbered after it.
    The conversion runs under one ordered set of store locks, so
    concurrent saves and loads observe one complete chain throughout.
    Returns ``None``.
    """
    if isinstance(members, MemoryChain):
        raise TypeError("a chain family must be a sequence of MemoryChains")
    try:
        members = list(members)
    except TypeError:
        raise TypeError(
            "a chain family must be a sequence of MemoryChains"
        ) from None
    if not members:
        raise CheckpointError("a chain family needs at least one chain")
    for member in members:
        if not isinstance(member, MemoryChain):
            raise TypeError("chain family members must be MemoryChains")
    if len(set(id(member) for member in members)) != len(members):
        raise CheckpointError("a chain family must not list a member twice")
    if len(members) == 1:
        return compact_chain_memory(members[0])

    # Deadlock-free total order on the member stores.
    ordered = sorted(members, key=id)
    for member in ordered:
        member._lock.acquire()
    try:
        heads = []
        documents = []
        for member in members:
            head = _read_head_optional(member)
            if head is None:
                raise CheckpointError(
                    "chain has no head pointer (no basis segment committed)"
                )
            _validate_reachable_chain(member, head)
            heads.append(head)
            documents.append(_load_chain_store(member, head))
        signatures = [_family_shape_signature(doc) for doc in documents]
        reference = signatures[0]
        for index, signature in enumerate(signatures[1:], start=1):
            if signature != reference:
                raise CheckpointError(
                    "family compaction rejected: chain "
                    f"{index} parameter shapes or layer order do not agree "
                    f"with chain 0's {reference!r}"
                )
        fold = _common_family_prefix_memory(members, heads)
        if fold < 1:
            _converge_identical_bases_memory(members)
            return None
        folded_doc = _walk_to(members[0], heads[0], fold)
        folded_bytes = build_bytes(folded_doc)
        for member, head in zip(members, heads):
            _compact_memory_member_for_family(
                member, fold, head, folded_doc, folded_bytes
            )
    finally:
        for member in reversed(ordered):
            member._lock.release()
    return None


def _common_family_prefix_memory(members, heads):
    """Largest *u* with seg-0..seg-u one shared bytes object per member."""
    u = 0
    limit = min(heads)
    first = members[0]
    while u <= limit:
        name = _segment_name(u)
        anchor = first._objects.get(name)
        if anchor is None:
            return u - 1
        if any(
            member._objects.get(name) is not anchor for member in members[1:]
        ):
            return u - 1
        u += 1
    return limit


def _converge_identical_bases_memory(members):
    """Reference-share members' basis bytes when they are equal objects."""
    anchor = members[0]._objects.get(_segment_name(_BASIS_INDEX))
    if anchor is None:
        return
    for member in members[1:]:
        name = _segment_name(_BASIS_INDEX)
        if member._objects.get(name) is not anchor and member._objects.get(name) == anchor:
            member._objects[name] = anchor


def _compact_memory_member_for_family(
    store, fold, head, folded_doc, folded_bytes
):
    """Renumber one memory member onto the shared folded basis."""
    staged = {0: folded_bytes}
    walker = _walker_from_document(folded_doc)
    previous = folded_doc
    for old_index in range(fold + 1, head + 1):
        new_index = old_index - fold
        raw = store.read_segment(_segment_name(old_index))
        walker.apply_delta(new_index, raw, wire_number=old_index)
        current = walker.document()
        staged[new_index] = _build_delta_between(previous, current, new_index)
        previous = current
    new_head = head - fold
    for index, raw in staged.items():
        store.write_segment(_segment_name(index), raw)
    for name in list(store._objects):
        index = _segment_index(name)
        if index is not None and index > new_head:
            del store._objects[name]
    _commit_head(store, new_head)




# -- marker encoding, read-through and recovery ----------------------------

def _encode_marker(marker):
    return json.dumps(marker, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )


def _decode_marker(raw):
    """Parse a compaction marker.

    * streaming: ``{"u": fold}`` -- fold in progress, marker immutable;
    * publish: ``{"u": fold, "h": new_head}`` -- promotion in force;
    * legacy: a bare decimal string from the pre-streaming build meaning
      every new segment was already staged and only publication remained.
    """
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise CheckpointError("compaction marker is corrupt") from None
    if re.fullmatch(r"\d+", text):
        return {"legacy_head": int(text)}
    try:
        marker = json.loads(text)
    except json.JSONDecodeError:
        raise CheckpointError("compaction marker is corrupt") from None
    keys = set(marker) if isinstance(marker, dict) else set()
    u = marker.get("u") if isinstance(marker, dict) else None
    if (
        keys not in ({"u"}, {"u", "h"})
        or isinstance(u, bool)
        or not isinstance(u, int)
        or u <= 0
    ):
        raise CheckpointError("compaction marker is corrupt")
    if "h" in marker and (
        isinstance(marker["h"], bool)
        or not isinstance(marker["h"], int)
        or marker["h"] < 0
    ):
        raise CheckpointError("compaction marker is corrupt")
    return marker


def _marker_path(directory):
    return os.path.join(directory, _COMPACT_MARKER)


def _marker_exists(directory):
    return os.path.exists(_marker_path(directory))


def _read_directory_marker(directory):
    """The fold point of a live streaming marker in *directory*, else None."""
    path = _marker_path(directory)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        marker = _decode_marker(fh.read())
    if "legacy_head" in marker or "h" in marker:
        # Recovery always completes such a marker before ordinary opens
        # proceed, so reaching it here means an inconsistent directory.
        raise CheckpointError("compaction marker is in an unexpected state")
    return marker["u"]


def _load_through_marker(store, head, fold, up_to):
    """Assemble chain state while a streaming fold is on disk.

    Logical chain (old numbering): the staged basis folds ``0..fold``;
    new delta *d* occupies the old slot ``seg-(fold+d)`` once converted,
    otherwise that slot still holds old delta ``fold+d``.  The two are
    told apart by the segment number recorded in the frame.  Reads never
    block on the fold and always observe one complete state.  *up_to* is
    a position in the old numbering (``None`` means the live head).
    """
    target = head if up_to is None else up_to
    if target <= fold:
        # The folded prefix stays on disk until publication, so a
        # historical prefix read walks the ordinary on-disk segments.
        walker = _ChainWalker(store.read_segment(_segment_name(_BASIS_INDEX)))
        for index in range(1, target + 1):
            walker.apply_delta(
                index, store.read_segment(_segment_name(index))
            )
        return walker.document()

    walker = _walker_from_document(
        parse_bytes(store.read_segment(_staged_name(0)))
    )
    new_target = target - fold
    for new_index in range(1, new_target + 1):
        raw = store.read_segment(_segment_name(fold + new_index))
        number = _delta_wire_number(raw)
        walker.apply_delta(new_index, raw, wire_number=number)
    return walker.document()


def _recover_directory_chain(directory, store):
    """Resolve an interrupted compaction at the start of every chain open.

    Runs under the directory lock.  With no marker the live chain is one
    complete state (staged debris from a pre-marker crash is dropped).
    With a marker owned by a live fold nothing is done -- the owner keeps
    streaming and other opens read through the marker.  A marker whose
    owner is dead is rolled deterministically forward and the directory
    left holding one complete chain.
    """
    marker_path = _marker_path(directory)
    if not os.path.exists(marker_path):
        # A live family-fold barrier means this member already stands on
        # its completed folded chain while the family operation finishes
        # the other members: the chain is coherent and the barrier's
        # lease must stay held until the family releases it.  Reads and
        # appends proceed against the ordinary (new) slots.
        if _family_barrier_exists(directory) and _marker_live(directory):
            return
        leftovers = [
            name
            for name in os.listdir(directory)
            if name.startswith(_STAGED_PREFIX)
        ]
        for name in leftovers:
            _unlink_quietly(os.path.join(directory, name))
        if leftovers:
            _fsync_directory(directory)
        # A dead family barrier (its owner was killed) is finished
        # state: drop the barrier and its lease -- the member already
        # holds one complete folded chain.
        _reap_dead_family_barrier(directory)
        if not _family_barrier_exists(directory):
            # A lease without a marker can only be debris from a crash
            # before the marker was published.
            _unlink_quietly(_lease_path(directory))
        # A staging sentinel in a committed chain is debris from a fork
        # killed between the staging rename and the sentinel unlink.
        if not _staging_dir_is_live(directory):
            _unlink_quietly(os.path.join(directory, _STAGING_LOCK_NAME))
        return

    if _marker_live(directory):
        return

    with open(marker_path, "rb") as fh:
        marker = _decode_marker(fh.read())
    head = _read_head_optional(store)

    if "legacy_head" in marker:
        _roll_forward_legacy(directory, store, marker["legacy_head"], head)
        return

    fold = marker["u"]
    if head < fold:
        raise CheckpointError("compaction marker is inconsistent with its head")
    final_head = marker.get("h")
    if final_head is None:
        # Streaming phase interrupted: finish converting the tail.
        final_head = head - fold
        staged_path = os.path.join(directory, _staged_name(0))
        try:
            with open(staged_path, "rb") as fh:
                staged_basis = fh.read()
        except FileNotFoundError:
            raise CheckpointError(
                "compaction staging area is incomplete; the chain cannot "
                "be rolled forward"
            ) from None
        walker = _walker_from_document(parse_bytes(staged_basis))
        previous = walker.document()
        for new_index in range(1, final_head + 1):
            slot_name = _segment_name(fold + new_index)
            raw = store.read_segment(slot_name)
            number = _delta_wire_number(raw)
            if number == new_index:
                walker.apply_delta(new_index, raw)
                previous = walker.document()
            else:
                walker.apply_delta(new_index, raw, wire_number=number)
                current = walker.document()
                delta = _build_delta_between(previous, current, new_index)
                _atomic_write(directory, slot_name, delta)
                previous = current
        _publish_compaction(directory, store, fold, final_head)
        return

    # Publication phase interrupted: finish the resumable promotion.
    if head == final_head and not os.path.exists(
        os.path.join(directory, _staged_name(0))
    ):
        # Head already switched; only debris may remain.
        _promote_tail_slots(directory, fold, final_head)
        _gc_published_chain(directory, final_head)
        return
    _finish_publication(directory, store, fold, final_head)


def _finish_publication(directory, store, fold, final_head):
    """Complete a promotion whose publish marker is on disk."""
    staged_path = os.path.join(directory, _staged_name(0))
    if os.path.exists(staged_path):
        with open(staged_path, "rb") as fh:
            staged_basis = fh.read()
        _atomic_write(directory, _segment_name(_BASIS_INDEX), staged_basis)
        _unlink_quietly(staged_path)
    for index in range(1, fold + 1):
        _unlink_quietly(os.path.join(directory, _segment_name(index)))
    _promote_tail_slots(directory, fold, final_head)
    for name in list(os.listdir(directory)):
        index = _segment_index(name)
        if index is not None and index > final_head:
            _unlink_quietly(os.path.join(directory, name))
    _commit_head(store, final_head)
    _gc_published_chain(directory, final_head)


def _gc_published_chain(directory, new_head):
    for name in list(os.listdir(directory)):
        index = _segment_index(name)
        if name.startswith(_STAGED_PREFIX) or (
            index is not None and index > new_head
        ):
            _unlink_quietly(os.path.join(directory, name))
    _unlink_quietly(_marker_path(directory))
    _unlink_quietly(_lease_path(directory))
    _fsync_directory(directory)


def _roll_forward_legacy(directory, store, new_head, head):
    """Finish a pre-streaming build's already-staged compaction."""
    if head == new_head:
        _gc_published_chain(directory, new_head)
        return
    if head <= new_head:
        raise CheckpointError("compaction marker is inconsistent with its head")
    for index in range(new_head + 1):
        staged_path = os.path.join(directory, _staged_name(index))
        try:
            with open(staged_path, "rb") as fh:
                raw = fh.read()
        except FileNotFoundError:
            raise CheckpointError(
                "compaction staging area is incomplete; the chain cannot "
                "be rolled forward"
            ) from None
        _atomic_write(directory, _segment_name(index), raw)
    _commit_head(store, new_head)
    _gc_published_chain(directory, new_head)


def _unlink_quietly(path):
    # Garbage collection only: a file that cannot be removed (a read-only
    # directory, say) is unreachable anyway and must not fail the open.
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Chain forks: deriving branch chains that share a prefix segment family
#
# Forking a chain at segment *k* creates a second chain directory whose
# segments ``0..k`` are hard links to the source chain's files -- the
# shared prefix is stored once -- plus its own ``head`` pointer naming
# segment *k*.  From then on each chain maintains only its own head and
# the delta segments it appends: appends landing on the same segment
# position in two chains live in different directories and never touch
# each other's bytes, and a state reassembled from either chain is bit
# for bit the state an unforked chain would hold.
#
# Shared segments are immutable by construction (every write anywhere in
# the engine is a temp file plus an atomic rename), so a reader of a
# shared segment never observes half-written content.  Reclamation is by
# reachability and deterministic: a compaction folding the prefix away,
# or the deletion of a whole branch directory, only drops that chain's
# own directory entries, and the filesystem reclaims a shared segment's
# bytes exactly when the last chain whose head can reach it lets go.  A
# process killed at any point therefore neither loses a reachable
# segment nor leaks an unreachable one, and every chain still standing
# reopens as one complete state.
# ---------------------------------------------------------------------------


def _check_fork_point(up_to, head):
    """The segment index a fork branches from; the head when omitted.

    The fork point must name an existing segment boundary: a point
    between positions (a non-integer), a negative index or a segment the
    chain has not committed yet is refused with ValueError.
    """
    if up_to is None:
        return head
    if isinstance(up_to, bool) or not isinstance(up_to, int):
        raise CheckpointError(
            "fork point must name a segment boundary (a non-negative "
            "integer segment index)"
        )
    if up_to < 0 or up_to > head:
        raise CheckpointError(
            f"fork point {up_to} names no segment of the chain "
            f"(the head is segment {head})"
        )
    return up_to


def fork_chain(source, target, up_to=None):
    """Derive a new chain from *source* at segment *up_to* into *target*.

    The branch chain directory shares every segment up to and including
    the fork point with the source chain (the prefix files are
    hard-linked, so they are stored once) and receives its own ``head``
    pointer; afterwards each chain maintains only its own head and the
    delta segments it appends.  Saves, loads, streaming compactions and
    verifications on either chain proceed independently and in parallel:
    a state reassembled from either chain is bit for bit identical to
    the state an unforked chain would hold, appends landing on the same
    segment position in both chains leave each other's bytes untouched,
    and a shared segment -- immutable by construction -- is never
    observed half-written.

    Reclamation is by reachability and deterministic: unlinking a shared
    segment from one chain (a compaction folding the prefix away, or the
    deletion of a whole branch directory) releases only that chain's
    reference, and the segment's bytes are reclaimed exactly when no
    chain's head can reach it any more.  A process killed at any point
    neither loses a reachable segment nor leaks an unreachable one, and
    every chain still standing reopens as one complete state.  The fork
    itself is atomic: the branch is staged under a private sibling
    directory and renamed into place, so *target* either appears as one
    complete chain or not at all (a killed attempt leaves only staging
    debris, which the next fork to the same target reclaims).

    *up_to* defaults to the source chain's current head.  A fork point
    that falls between segment positions (a non-integer) or names a
    segment the chain has not committed, and a *target* that already
    exists, raise ``ValueError``; a missing source directory or a
    missing referenced segment raises ``FileNotFoundError``; an
    unwritable destination or a full disk raises ``OSError``.  The
    shared prefix is fully validated before anything is linked, so a
    corrupt chain is rejected before the target appears.
    """
    if not isinstance(source, (str, os.PathLike)):
        raise TypeError("fork source must be a chain directory path")
    if not isinstance(target, (str, os.PathLike)):
        raise TypeError("fork target must be a chain directory path")
    source = os.fspath(source)
    target = os.fspath(target)
    if not os.path.isdir(source):
        raise FileNotFoundError(
            f"incremental checkpoint directory not found: {source!r}"
        )
    source_abs = os.path.abspath(source)
    target_abs = os.path.abspath(target)
    if os.path.exists(target_abs) or target_abs == source_abs:
        raise CheckpointError(f"fork target already exists: {target!r}")
    if target_abs.startswith(source_abs + os.sep):
        raise CheckpointError(
            "fork target must not live inside the source chain directory"
        )
    parent = os.path.dirname(target_abs)
    staging_prefix = _FORK_TMP_PREFIX + os.path.basename(target_abs) + "-"
    # Deterministically reclaim debris of killed earlier forks/deletes in
    # the family directory (staging dirs whose owner is gone).
    _sweep_parent_staging(parent)

    # Serialise against saves and compactions on the source: the shared
    # prefix must not be replaced or released while it is being linked.
    # A live streaming fold is waited out first (a dead one is rolled
    # forward by the recovery step), so the linked prefix always comes
    # from a quiescent chain.
    while True:
        with _DirectoryChainLock(source):
            store = _DirectoryChainStore(source)
            _recover_directory_chain(source, store)
            if _member_fold_busy(source):
                contended = True
            else:
                contended = False
                head = _read_head_optional(store)
                if head is None:
                    raise CheckpointError(
                        "chain has no head pointer (no basis segment committed)"
                    )
                fork_point = _check_fork_point(up_to, head)
                # Validate the whole shared prefix before anything is
                # linked; a missing referenced segment surfaces as
                # FileNotFoundError, any corruption as ValueError.
                _walk_to(store, fork_point, fork_point)
                # Deterministic GC: segments beyond the head are
                # unreachable crash residue and are swept on every fork.
                _sweep_chain_debris(source, head)
                _materialize_fork(
                    source, target_abs, parent, staging_prefix, fork_point
                )
        if not contended:
            return None
        _wait_for_live_marker(source)


def _materialize_fork(source, target, parent, staging_prefix, fork_point):
    """Link the shared prefix into a staging dir and rename it into place.

    Runs under the source directory lock, so no save or compaction can
    replace or release a prefix segment while it is being linked.  The
    branch appears atomically: the staging directory is fully populated
    -- segment links first, the head pointer last, mirroring the chain
    commit order -- and then renamed over the (nonexistent) target.  A
    flock-held sentinel marks the staging as owned by this live fork for
    the whole window, so the deterministic staging sweep only ever
    reclaims stagings whose creator died.
    """
    staging = None
    lease_fd = None
    committed = False
    try:
        # Create the staging and lock its sentinel under the parent
        # directory's flock, which is exactly the lock a sweep holds
        # while scanning: a sweep therefore never sees a staging that
        # exists but is not yet recognisably alive.
        with _DirectoryChainLock(parent):
            staging = tempfile.mkdtemp(prefix=staging_prefix, dir=parent)
            _fork_staging_register(staging)
            lease_fd = _acquire_staging_lease(staging)
        for index in range(fork_point + 1):
            os.link(
                os.path.join(source, _segment_name(index)),
                os.path.join(staging, _segment_name(index)),
            )
        _atomic_write(staging, _HEAD_NAME, str(fork_point).encode("ascii"))
        _fsync_directory(staging)
        if os.path.exists(target):
            raise CheckpointError(f"fork target already exists: {target!r}")
        os.rename(staging, target)
        committed = True
        _fsync_directory(parent)
        # The sentinel's work is done; a kill before this unlink leaves
        # it as debris the next open of the chain reclaims.
        _unlink_quietly(os.path.join(target, _STAGING_LOCK_NAME))
    finally:
        if staging is not None:
            _release_staging_lease(lease_fd, staging)
        if not committed:
            if staging is not None:
                _remove_tree_quietly(staging)


def _remove_tree_quietly(directory):
    """Best-effort removal of a fork staging directory and its contents."""
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        _unlink_quietly(os.path.join(directory, name))
    try:
        os.rmdir(directory)
    except OSError:
        pass


def fork_chain_memory(store, up_to=None):
    """Derive an in-memory branch chain; same semantics as :func:`fork_chain`.

    The branch shares the prefix segments with the source chain (segment
    bytes are immutable, so sharing them by reference is safe) and
    receives its own head pointer; later appends and compactions on
    either chain are fully independent.  Returns the new
    :class:`MemoryChain`.
    """
    if not isinstance(store, MemoryChain):
        raise TypeError("store must be a MemoryChain")
    with store._lock:
        head = _read_head_optional(store)
        if head is None:
            raise CheckpointError(
                "chain has no head pointer (no basis segment committed)"
            )
        fork_point = _check_fork_point(up_to, head)
        _walk_to(store, fork_point, fork_point)
        branch = MemoryChain()
        for index in range(fork_point + 1):
            name = _segment_name(index)
            branch._objects[name] = store._objects[name]
        branch._objects[_HEAD_NAME] = str(fork_point).encode("ascii")
    return branch


# ---------------------------------------------------------------------------
# Branch deletion and deterministic reachability cleanup
#
# Deleting a branch chain removes exactly that chain's references: its
# directory, its head and the delta segments it alone owned.  The
# directory is renamed aside in one directory-entry change and only then
# emptied, so the family observes either the complete branch or none of
# it -- never a half one -- and a shared segment's bytes are reclaimed by
# the filesystem exactly when the last chain whose head can reach them
# lets go.
#
# Reclamation is a pure function of reachability, so it is idempotent:
# sweeping again changes nothing and advances no chain's state.  Two
# kinds of crash residue are swept deterministically on the next fork,
# compaction or deletion (never on a plain load or verify):
#
# * staging directories a killed fork or delete left behind in the family
#   directory (``.seqfork.tmp-*`` / ``.seqdel.tmp-*``) -- a fork staging
#   dir is removed only when the flock sentinel its creator holds is
#   free, so a fork still populating its staging area is never disturbed;
# * segment files no head can reach -- an index greater than the head,
#   left when a process was killed between the segment write and the head
#   advance -- plus crash debris carrying the temp-file prefix.
# ---------------------------------------------------------------------------


_STAGING_LOCK_NAME = ".seqstag.lock"

_fork_stagings_guard = threading.Lock()
_active_fork_stagings: set[str] = set()


def _fork_staging_register(directory):
    with _fork_stagings_guard:
        _active_fork_stagings.add(os.path.abspath(directory))


def _fork_staging_release(directory):
    with _fork_stagings_guard:
        _active_fork_stagings.discard(os.path.abspath(directory))


def _acquire_staging_lease(staging):
    """Create the staging sentinel and hold an exclusive flock on it.

    The sentinel exists for the whole population window, so a sweeper can
    tell a fork killed mid-linking (lock free) from a live one (lock
    held).  Returns the locked fd (on platforms without ``fcntl`` only
    the in-process registry guards the staging).
    """
    fd = os.open(
        os.path.join(staging, _STAGING_LOCK_NAME),
        os.O_RDWR | os.O_CREAT,
        0o644,
    )
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_staging_lease(fd, staging):
    _fork_staging_release(staging)
    if not fd:
        return
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    os.close(fd)


def _staging_dir_is_live(staging):
    """Whether a fork staging directory belongs to a still-running fork."""
    with _fork_stagings_guard:
        if os.path.abspath(staging) in _active_fork_stagings:
            return True
    sentinel = os.path.join(staging, _STAGING_LOCK_NAME)
    if fcntl is None or not os.path.exists(sentinel):
        return False
    try:
        fd = os.open(sentinel, os.O_RDONLY)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                return True
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        return False
    finally:
        os.close(fd)


def _sweep_parent_staging(parent):
    """Remove killed fork/delete staging directories in *parent*.

    Garbage collection only: a directory still owned by a live fork is
    skipped and operating-system failures never reach the caller.  A
    delete staging directory is always detached (the rename is its only
    creation step), so it needs no liveness check.  The scan runs under
    the parent directory's advisory flock, which is the same lock a fork
    holds while creating its staging and locking its sentinel, so a
    staging is either not there yet or already recognisably alive --
    there is no window in which a half-created live staging can be
    swept.
    """
    with _DirectoryChainLock(parent):
        try:
            names = os.listdir(parent)
        except OSError:
            return
        removed = False
        for name in names:
            if name.startswith(_FORK_TMP_PREFIX):
                path = os.path.join(parent, name)
                if not os.path.isdir(path):
                    _unlink_quietly(path)
                    removed = True
                elif not _staging_dir_is_live(path):
                    _remove_tree_quietly(path)
                    removed = True
            elif name.startswith(_DELETE_TMP_PREFIX):
                path = os.path.join(parent, name)
                if os.path.isdir(path):
                    _remove_tree_quietly(path)
                else:
                    _unlink_quietly(path)
                removed = True
            elif name == _FAMILY_LOCK_NAME:
                # A family-fold lease file a killed process left behind is
                # reclaimed once no family fold (in this process or
                # another) still holds it; a live family fold always
                # keeps its marker, so an unheld lock here is debris.
                if not _family_lease_live(parent):
                    _unlink_quietly(os.path.join(parent, name))
                    removed = True
        if removed:
            _fsync_directory(parent)


def _sweep_chain_debris(directory, head):
    """Drop temp-file residue and segments no head of *directory* reaches.

    Runs under the directory lock and only when no compaction marker is in
    force, so every ``seg-*`` slot beyond *head* is unreachable crash
    residue (a converted compaction tail always lands at or below the
    head).  Pure reachability: idempotent, writes nothing into the chain
    and advances no chain's state.
    """
    changed = False
    for name in list(os.listdir(directory)):
        path = os.path.join(directory, name)
        index = _segment_index(name)
        unreachable = index is not None and head is not None and index > head
        if unreachable or name.startswith(_TMP_PREFIX):
            _unlink_quietly(path)
            changed = True
    if changed:
        _fsync_directory(directory)


def _validate_reachable_chain(store, head):
    """Walk the whole reachable chain, like a load, for delete validation.

    A missing referenced segment -- with the directory itself present --
    is an unparseable chain structure and therefore a ``ValueError``, not
    the ``FileNotFoundError`` a missing chain directory gets.
    """
    try:
        _load_chain_store(store, head)
    except FileNotFoundError as exc:
        raise CheckpointError(
            "chain is missing a segment its head reaches; the chain "
            "structure is unparseable"
        ) from exc


def delete_chain(directory):
    """Delete one branch chain directory from a chain family.

    The directory, its ``head`` and the incremental segments it alone
    owned are removed.  Shared segments are dropped by reachability only:
    the filesystem keeps their bytes while any chain's head can still
    reach them (hard links from a forked branch hold them), and reclaims
    them exactly when the last reference goes.  The directory is renamed
    aside in one directory-entry change and then emptied, so concurrent
    forks and compactions on the family always observe either the
    complete branch or no branch at all, and every surviving chain stays
    one complete state.

    Staging directories of killed forks/deletes and segments no head can
    reach are swept deterministically as part of the deletion; the sweep
    is a pure reachability function -- repeating it changes nothing and
    advances no chain's optimizer state.

    A *directory* path that does not exist (including a chain already
    deleted) raises ``FileNotFoundError`` and leaves every other chain
    untouched.  A directory that exists but has no ``head``, or holds a
    chain that cannot be parsed (a corrupt, truncated, missing,
    out-of-order or shape-inconsistent reachable segment), raises
    ``ValueError`` and removes not a single shared segment.  An
    unwritable directory/parent or a full disk raises ``OSError``; a
    teardown interrupted at that point leaves only whole segment files in
    a detached staging directory, which the next fork, compaction or
    deletion reclaims -- never a half-written segment.
    """
    if not isinstance(directory, (str, os.PathLike)):
        raise TypeError("chain directory must be a path")
    directory = os.path.abspath(os.fspath(directory))
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"incremental checkpoint directory not found: {directory!r}"
        )
    parent = os.path.dirname(directory)
    # Deterministic GC: staging directories killed forks/deletes left in
    # the family directory are reclaimed on every deletion.
    _sweep_parent_staging(parent)

    # Wait out a live fold exactly like fork: deletion must not race the
    # marker protocol, and a dead fold is rolled forward first.
    while True:
        with _DirectoryChainLock(directory):
            store = _DirectoryChainStore(directory)
            _recover_directory_chain(directory, store)
            if _member_fold_busy(directory):
                contended = True
            else:
                contended = False
                head = _read_head_optional(store)
                if head is None:
                    raise CheckpointError(
                        "chain has no head pointer (no basis segment committed)"
                    )
                # Validate the whole reachable chain before anything moves;
                # only then may crash residue be swept (unreachable files
                # only -- no reachable shared segment is ever touched).
                _validate_reachable_chain(store, head)
                _sweep_chain_debris(directory, head)

                # Reserve the detached name before touching the chain: an
                # unwritable parent fails here, with the chain intact.
                staging = tempfile.mkdtemp(
                    prefix=_DELETE_TMP_PREFIX
                    + os.path.basename(directory)
                    + "-",
                    dir=parent,
                )
                os.rmdir(staging)
                try:
                    os.rename(directory, staging)
                except FileNotFoundError:
                    # Another deleter won between the check and the rename.
                    raise FileNotFoundError(
                        f"incremental checkpoint directory not found: {directory!r}"
                    ) from None
                _fsync_directory(parent)
                _empty_and_remove(staging)
                _fsync_directory(parent)
        if not contended:
            return None
        _wait_for_live_marker(directory)


def _empty_and_remove(directory):
    """Unlink every entry of a detached chain dir and remove the dir.

    Only whole files are removed (segment files are immutable), so an
    interrupted teardown never leaves a half segment; whatever remains is
    detached residue the next family operation sweeps.  Permission or I/O
    failures propagate as ``OSError``.  A concurrent sweep may reclaim
    the same detached directory; that cooperation is not an error.
    """
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return
    for name in names:
        path = os.path.join(directory, name)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                _empty_and_remove(path)
            else:
                os.unlink(path)
        except FileNotFoundError:
            pass
    try:
        os.rmdir(directory)
    except FileNotFoundError:
        pass


def delete_chain_memory(store):
    """Delete an in-memory chain; same semantics as :func:`delete_chain`.

    Drops the head pointer and every segment the store held in one
    critical section, so a concurrent observer never sees a partial
    chain.  The emptied store can be reused: its next save writes a
    fresh basis.
    """
    if not isinstance(store, MemoryChain):
        raise TypeError("store must be a MemoryChain")
    with store._lock:
        head = _read_head_optional(store)
        if head is None:
            raise CheckpointError(
                "chain has no head pointer (no basis segment committed)"
            )
        _validate_reachable_chain(store, head)
        store.remove_head()
        for name in list(store._objects):
            if _segment_index(name) is not None:
                del store._objects[name]
    return None


# ---------------------------------------------------------------------------
# Branch merge: landing one chain's current state onto another
#
# A merge appends exactly one segment to the target chain: the delta
# between the target's head state and the source chain's head state,
# encoded by the same leaf-identity diff a plain save uses.  The target's
# existing segments are all kept and the new delta is the only segment
# added, so loading the target reassembles bit for bit to the state the
# source held at merge time while the source's head and segment files are
# never touched.  Segments the family already shares (the hard-linked
# fork prefix) are never rewritten; the new delta carries only tensors
# the source alone owned that genuinely differ from the target's current
# state, and is empty (but still appended) when the two states already
# agree -- so merging the same state twice keeps the state unchanged.
#
# The merge runs as two independently serialised steps rather than one
# critical section over both directories: the source chain's current
# state is assembled under the source lock (a live fold is waited out
# first, exactly as for a fork), and only then is the one merge delta
# appended to the target under the target lock (segment-then-head, as a
# save does).  Serialising the steps instead of holding the two locks
# together is deadlock-free for any two merges, including a pair
# running in opposite directions at once; the captured source state is
# a fully materialised document, so a later move of the source chain
# cannot disturb the append.  A process killed before the head moves
# leaves only an orphan segment beyond the old head (reclaimed like any
# save residue by the next fork, compaction, deletion or merge), and
# the head never names anything but a complete chain, old or new.
# ---------------------------------------------------------------------------


def merge_chains(source, target):
    """Merge the current state of chain *source* onto chain *target*.

    Every segment *target* already had is kept; one new delta segment is
    appended after its head carrying the tensors in which the source
    chain's head state differs from the target's head state (an empty
    delta when the two already agree).  Loading *target* afterwards
    reassembles bit for bit to the state *source* held at merge time,
    while *source* -- its head and its segment files -- is not modified
    at all, and the two chains keep evolving independently.  Segments
    the family already shares are never stored twice: the appended
    segment belongs to the target alone and holds only the part the
    source alone owned that genuinely differs.  The merge advances no
    optimizer step -- the target simply inherits the source's complete
    state, its step count included; a subsequent full save lands exactly
    the merged state.

    The read of the source and the append to the target are two steps,
    each serialised by its chain directory lock (a live fold is waited
    out first), so merges -- even two merging the same two chains in
    opposite directions at once -- and forks, compactions, deletions and
    saves serialise without deadlocking and every chain always loads as
    one complete state.

    Repeating the merge of the same state appends another empty delta
    and changes nothing else.  A missing source or target directory, or
    a segment either head reaches that is absent, raises
    ``FileNotFoundError`` and leaves every other chain untouched.
    Merging a chain into itself, two chains whose parameter shapes or
    layer order disagree, or an unparseable chain structure rejects the
    whole merge with ``ValueError`` before one byte of the target is
    rewritten.  An unwritable directory or a full disk raises
    ``OSError``; a process killed mid-merge leaves only the old head
    (plus orphan segment residue) or the new head, both one complete
    chain, and the residue is reclaimed deterministically by the next
    fork, compaction, deletion or merge.
    """
    if not isinstance(source, (str, os.PathLike)):
        raise TypeError("merge source must be a chain directory path")
    if not isinstance(target, (str, os.PathLike)):
        raise TypeError("merge target must be a chain directory path")
    source = os.fspath(source)
    target = os.fspath(target)
    if not os.path.isdir(source):
        raise FileNotFoundError(
            f"incremental checkpoint directory not found: {source!r}"
        )
    if not os.path.isdir(target):
        raise FileNotFoundError(
            f"incremental checkpoint directory not found: {target!r}"
        )
    source_abs = os.path.abspath(source)
    target_abs = os.path.abspath(target)
    if source_abs == target_abs:
        raise CheckpointError("a chain cannot be merged into itself")
    # Deterministic GC of killed fork/delete staging in the family
    # directory the merge writes into, exactly as the other writers do.
    _sweep_parent_staging(os.path.dirname(target_abs))

    # Step 1: capture the source's current state under its own lock.
    source_doc = _capture_chain_state(source_abs)
    # Step 2: append the one merge delta under the target lock.
    _append_merge_delta(target_abs, source_doc)
    return None


def _capture_chain_state(directory):
    """Assemble a chain's current head state, waiting out a live fold.

    The directory is opened exactly the way a fork opens its source: any
    interrupted fold is rolled forward and a live one is waited out, so
    the captured document comes from a quiescent chain.  A missing
    referenced segment raises ``FileNotFoundError``; any other defect
    raises :class:`CheckpointError`.
    """
    while True:
        with _DirectoryChainLock(directory):
            store = _DirectoryChainStore(directory)
            _recover_directory_chain(directory, store)
            if not _member_fold_busy(directory):
                head = _read_head_optional(store)
                if head is None:
                    raise CheckpointError(
                        "chain has no head pointer (no basis segment committed)"
                    )
                return _load_chain_store(store, head)
        _wait_for_live_marker(directory)


def _append_merge_delta(directory, source_doc):
    """Validate *source_doc* against the target and append the merge delta.

    The target is opened quiescent and the whole append runs in one lock
    acquisition: the target head is read, its current state assembled and
    the delta frozen (the pure diff is also the full compatibility
    check) before a single byte is written, so a rejected merge leaves
    the target byte for byte untouched.
    """
    while True:
        with _DirectoryChainLock(directory):
            store = _DirectoryChainStore(directory)
            _recover_directory_chain(directory, store)
            if not _member_fold_busy(directory):
                target_head = _read_head_optional(store)
                if target_head is None:
                    raise CheckpointError(
                        "chain has no head pointer (no basis segment committed)"
                    )
                target_doc = _load_chain_store(store, target_head)
                next_index = target_head + 1
                # The diff is also the compatibility check: equal
                # parameter counts, equal layer order, consistent
                # hidden-slot transitions and matching tensor shapes are
                # all required before the delta can be frozen.
                delta_bytes = _build_delta_between(
                    target_doc, source_doc, next_index
                )
                # Validation is complete; only now may crash residue be
                # reclaimed (unreachable files only -- no reachable
                # segment is touched).
                _sweep_chain_debris(directory, target_head)
                store.write_segment(_segment_name(next_index), delta_bytes)
                _commit_head(store, next_index)
                return None
        _wait_for_live_marker(directory)


def merge_chains_memory(source, target):
    """Merge an in-memory chain's state onto another; same semantics as
    :func:`merge_chains`.

    The source state is captured under the source store lock and the one
    merge delta is then appended under the target store lock, so the two
    steps are serialised independently and opposing merges cannot
    deadlock.  The target receives exactly one new delta segment (empty
    when the two states already agree) and reassembles afterwards to the
    source's head state; the source store is never modified.  Merging a
    store into itself raises ``ValueError``.
    """
    if not isinstance(source, MemoryChain):
        raise TypeError("merge source must be a MemoryChain")
    if not isinstance(target, MemoryChain):
        raise TypeError("merge target must be a MemoryChain")
    if source is target:
        raise CheckpointError("a chain cannot be merged into itself")
    with source._lock:
        source_head = _read_head_optional(source)
        if source_head is None:
            raise CheckpointError(
                "merge source chain has no head pointer "
                "(no basis segment committed)"
            )
        source_doc = _load_chain_store(source, source_head)
    with target._lock:
        target_head = _read_head_optional(target)
        if target_head is None:
            raise CheckpointError(
                "merge target chain has no head pointer "
                "(no basis segment committed)"
            )
        target_doc = _load_chain_store(target, target_head)
        next_index = target_head + 1
        delta_bytes = _build_delta_between(
            target_doc, source_doc, next_index
        )
        target.write_segment(_segment_name(next_index), delta_bytes)
        _commit_head(target, next_index)
    return None


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
