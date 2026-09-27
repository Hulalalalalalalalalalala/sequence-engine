"""Cross-family incremental synchronization.

Two families of incremental checkpoint chains can be synchronized
without repacking the history they already hold identically:

* :func:`diff_families` compares a source family with a target family
  member by member (aligned in order, member *i* with member *i*) and
  writes one *sync artifact* holding only the segments that genuinely
  differ.  Segment positions the corresponding members already reach
  through identical bytes are recorded as shared and carry no block; a
  member identical to its counterpart carries no blocks at all.  Each
  member's head, its member-owned tail beyond the target's head and the
  segment positions are expressed in the artifact exactly as the source
  holds them.

* :func:`apply_family_diff` lands such an artifact on the target family
  atomically.  The whole artifact is validated, and matched against the
  target family, before anything is written; every member's new chain
  is built in a private sibling staging directory (segment files first,
  no member byte touched), and the family switches over with every
  member lock held at once, per member through an atomic rotation, so
  the application either takes effect for every member or for none.

The sync artifact converts in both directions with the full family
artifact (:func:`full_to_incremental_artifact`,
:func:`incremental_to_full_artifact`); a family restored from either
form loads to exactly the same state.

Commit protocol on disk
-----------------------

Per application the parent directory holds a family marker
(``.seqsync``, naming the artifact digest) and one private copy of the
artifact bytes (``.seqsync.art-<digest>``); each member's new chain is
staged in the sibling directory ``.seqsync.stage-<member>``.  The
rotation itself runs under the member's directory lock (the same lock
a save/load uses) with a small state marker (``<member>/.seqsync``):
the marker records stage 2, the staged segment files are renamed over
the member's slots, the head is committed and the marker and staging
directory are removed.  A kill before the rotation leaves the ordinary
chain (and the family marker for the re-run); a kill mid-rotation is
rolled forward deterministically on the member's next open or the next
family operation, following the marker -- every member always stands on
one complete chain, old or new, and re-running converges to one
completed application.  All sync-specific residue is reclaimed,
idempotently, by the next family operation over the parent.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import struct
import zlib

from .checkpoint import (
    DELTA_MAGIC,
    FAMILY_EXPORT_MAGIC,
    MAGIC,
    ChainVerification,
    CheckpointError,
    MemoryChain,
    _ChainWalker,
    _DirectoryChainLock,
    _DirectoryChainStore,
    _FAMILY_EXPORT_VERSION,
    _FAMILY_SEGMENT_KEYS,
    _FamilyParentGuard,
    _HEAD_NAME,
    _atomic_write,
    _check_family_member_name,
    _check_family_shape_agreement,
    _exact_keys,
    _family_export_snapshot,
    _family_register,
    _family_release,
    _fsync_directory,
    _load_chain_store,
    _member_fold_busy,
    _parse_delta,
    _parse_family_export,
    _read_head_optional,
    _read_member_segments,
    _recover_directory_chain,
    _remove_tree_quietly,
    _segment_index,
    _segment_name,
    _sweep_parent_staging,
    _unlink_quietly,
    _wait_for_member_quiescent,
    parse_bytes,
)
from .checkpoint import (
    register_chain_recovery_hook,
    register_family_guarded_maintenance_hook,
    register_parent_residue_sweep,
    register_readonly_verify_hook,
)

# ---------------------------------------------------------------------------
# File names used by one application in the family parent and each member
# ---------------------------------------------------------------------------

SYNC_MARKER_NAME = ".seqsync"
SYNC_STAGING_PREFIX = ".seqsync.stage-"
SYNC_ARTIFACT_PREFIX = ".seqsync.art-"
_SYNC_MARKER_MAGIC = "seq-sync-member"

SYNC_EXPORT_MAGIC = b"SEQFAMX2"
SYNC_EXPORT_END_MAGIC = b"SEQFAMX2END"
_SYNC_EXPORT_VERSION = 1

_SYNC_FAMILY_MARKER_KEYS = frozenset(("v", "d", "m"))
_SYNC_FAMILY_HEAD_KEYS = frozenset(("n", "h"))
_SYNC_MEMBER_MARKER_KEYS = frozenset(("v", "a", "h", "n", "s"))
_SYNC_MANIFEST_KEYS = frozenset(("v", "target", "members", "segments"))
_SYNC_MEMBER_RECORD_KEYS = frozenset(("name", "head", "segments", "sh"))


def _sync_marker_path(directory):
    return os.path.join(directory, SYNC_MARKER_NAME)


# ---------------------------------------------------------------------------
# Small argument / IO helpers
# ---------------------------------------------------------------------------


def _normalize_dir_family(members, what="a chain family"):
    if isinstance(members, (str, bytes, os.PathLike)):
        raise TypeError(f"{what} must be a sequence of chain directories")
    try:
        members = list(members)
    except TypeError:
        raise TypeError(f"{what} must be a sequence of chain directories") from None
    if not members:
        raise CheckpointError("a chain family needs at least one chain")
    paths = []
    for member in members:
        if not isinstance(member, (str, os.PathLike)):
            raise TypeError("chain family members must be chain directory paths")
        paths.append(os.fspath(member))
    return paths


def _one_parent(paths, operation):
    parents = {os.path.dirname(path) for path in paths}
    if len(parents) != 1:
        raise CheckpointError(
            f"{operation} requires every member directory to live in the "
            "same parent directory"
        )
    return parents.pop()


def _artifact_bytes(artifact):
    if isinstance(artifact, (str, os.PathLike)):
        with open(os.fspath(artifact), "rb") as fh:
            return fh.read()
    if isinstance(artifact, (bytes, bytearray, memoryview)):
        return bytes(artifact)
    raise TypeError("the artifact must be bytes or a filesystem path")


# ---------------------------------------------------------------------------
# Target-family digest
# ---------------------------------------------------------------------------


def _member_digest_body(head, records):
    """Fingerprint of one member's head-reachable bytes (position/len/CRC)."""
    pieces = []
    for index in range(head + 1):
        raw = records[_segment_name(index)][0]
        pieces.append(f"{index}:{len(raw)}:{zlib.crc32(raw) & 0xFFFFFFFF}")
    return ";".join(pieces)


def _family_digest(names, heads, records):
    body = "\n".join(
        f"{name}|{heads[name]}|{_member_digest_body(heads[name], records[name])}"
        for name in names
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Artifact framing / parsing
# ---------------------------------------------------------------------------


def _frame_artifact(magic, manifest, blobs):
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    end_magic = magic + b"END"
    body = bytearray()
    body += magic
    body += struct.pack("<I", manifest["v"])
    body += struct.pack("<Q", len(manifest_bytes))
    body += manifest_bytes
    body += struct.pack("<Q", len(blobs))
    crc = zlib.crc32(manifest_bytes)
    for raw in blobs:
        body += struct.pack("<Q", len(raw))
        body += raw
        crc = zlib.crc32(raw, crc)
    body += end_magic
    body += struct.pack("<I", crc & 0xFFFFFFFF)
    return bytes(body)


def _read_envelope(raw, magic, what):
    """Validate framing/CRC; return ``(version, manifest, blobs)``."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise CheckpointError(f"{what} must be bytes")
    data = bytes(raw)
    end_magic = magic + b"END"
    prefix_len = len(magic) + 4 + 8
    if len(data) < prefix_len:
        raise CheckpointError(f"{what} is truncated while reading its header")
    if data[: len(magic)] != magic:
        raise CheckpointError(f"not a {what} (bad magic)")
    (version,) = struct.unpack("<I", data[len(magic) : len(magic) + 4])
    (manifest_len,) = struct.unpack("<Q", data[len(magic) + 4 : prefix_len])
    if manifest_len <= 0:
        raise CheckpointError(f"{what} manifest length is invalid")
    trailer_len = len(end_magic) + 4
    trailer_pos = data.rfind(end_magic)
    if trailer_pos < 0 or len(data) - trailer_pos != trailer_len:
        raise CheckpointError(f"{what} trailer is missing or the file is torn")
    manifest_start = prefix_len
    if manifest_len > trailer_pos - manifest_start:
        raise CheckpointError(f"{what} manifest overruns the block area")
    manifest_bytes = data[manifest_start : manifest_start + manifest_len]
    (stored_crc,) = struct.unpack(
        "<I", data[trailer_pos + len(end_magic) :]
    )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"{what} manifest is invalid: {exc}") from exc
    block_count_pos = manifest_start + manifest_len
    if block_count_pos + 8 > trailer_pos:
        raise CheckpointError(f"{what} is truncated before its block inventory")
    (block_count,) = struct.unpack(
        "<Q", data[block_count_pos : block_count_pos + 8]
    )
    inventory = manifest.get("segments") if isinstance(manifest, dict) else None
    if not isinstance(inventory, list):
        raise CheckpointError(f"{what} segment inventory is invalid")
    if block_count != len(inventory):
        raise CheckpointError(
            f"{what} block count does not match its segment inventory"
        )
    pos = block_count_pos + 8
    blobs = []
    running_crc = zlib.crc32(manifest_bytes)
    for block_id, entry in enumerate(inventory):
        _exact_keys(entry, _FAMILY_SEGMENT_KEYS, f"{what} segment entry {block_id}")
        name = entry["name"]
        if not isinstance(name, str) or _segment_index(name) is None:
            raise CheckpointError(
                f"{what} segment entry {block_id} has a bad segment file name"
            )
        size = entry["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CheckpointError(
                f"{what} segment entry {block_id} ({name}) has an invalid size"
            )
        crc_value = entry["crc"]
        if (
            isinstance(crc_value, bool)
            or not isinstance(crc_value, int)
            or not 0 <= crc_value <= 0xFFFFFFFF
        ):
            raise CheckpointError(
                f"{what} segment entry {block_id} ({name}) has an invalid crc"
            )
        if pos + 8 > trailer_pos:
            raise CheckpointError(
                f"{what} is truncated before segment block {block_id} ({name})"
            )
        (raw_len,) = struct.unpack("<Q", data[pos : pos + 8])
        pos += 8
        if raw_len != size or size > trailer_pos - pos:
            raise CheckpointError(
                f"{what} segment block {block_id} ({name}) is truncated or its "
                "length disagrees with the manifest"
            )
        chunk = data[pos : pos + size]
        pos += size
        if (zlib.crc32(chunk) & 0xFFFFFFFF) != crc_value:
            raise CheckpointError(
                f"{what} segment block {block_id} ({name}) fails its CRC"
            )
        running_crc = zlib.crc32(chunk, running_crc)
        blobs.append(chunk)
    if pos != trailer_pos:
        raise CheckpointError(f"{what} has trailing or misaligned block bytes")
    if (running_crc & 0xFFFFFFFF) != stored_crc:
        raise CheckpointError(f"{what} CRC mismatch: the file is corrupt or torn")
    return version, manifest, blobs


def parse_sync_artifact(raw):
    """Validate and decode a sync artifact.

    Returns ``(target_digest, members, blobs)``; each member plan is
    ``{"name", "head", "shared", "shared_set", "carried"}`` with
    *shared* the sorted positions shared byte for byte with the target
    and *carried* a slot -> block-id map.  Framing, CRCs, the reach
    layout (shared positions and carried blocks partition ``0..head``
    exactly once) and every carried segment's own frame (a full basis
    at slot 0, a delta whose number matches its position elsewhere) are
    checked here.  Chain continuity across shared slots is checked
    against the target family at application/conversion time.
    """
    version, manifest, blobs = _read_envelope(
        raw, SYNC_EXPORT_MAGIC, "family sync artifact"
    )
    if version != _SYNC_EXPORT_VERSION:
        raise CheckpointError(
            f"unsupported family sync artifact version {version}; this build "
            f"reads version {_SYNC_EXPORT_VERSION}"
        )
    if not isinstance(manifest, dict) or set(manifest) != _SYNC_MANIFEST_KEYS:
        raise CheckpointError(
            "family sync artifact manifest must be an object with exactly "
            f"the fields {sorted(_SYNC_MANIFEST_KEYS)}"
        )
    if (
        not isinstance(manifest["v"], int)
        or isinstance(manifest["v"], bool)
        or manifest["v"] != version
    ):
        raise CheckpointError("family sync artifact manifest version mismatch")
    target_digest = manifest["target"]
    if not isinstance(target_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", target_digest
    ):
        raise CheckpointError("family sync artifact target digest is invalid")
    members_raw = manifest["members"]
    if not isinstance(members_raw, list) or not members_raw:
        raise CheckpointError("family sync artifact must name at least one member")
    members = []
    seen_names = set()
    for position, entry in enumerate(members_raw):
        what = f"family sync artifact member {position}"
        _exact_keys(entry, _SYNC_MEMBER_RECORD_KEYS, what)
        name = entry["name"]
        _check_family_member_name(name, what)
        if name in seen_names:
            raise CheckpointError(
                f"family sync artifact lists member {name!r} more than once"
            )
        seen_names.add(name)
        head = entry["head"]
        if isinstance(head, bool) or not isinstance(head, int) or head < 0:
            raise CheckpointError(f"{what} ({name!r}) has an invalid head")
        block_refs = entry["segments"]
        shared = entry["sh"]
        if not isinstance(block_refs, list) or not isinstance(shared, list):
            raise CheckpointError(f"{what} ({name!r}) has invalid segment lists")
        shared_positions = []
        previous_pos = -1
        for slot_index in shared:
            if (
                isinstance(slot_index, bool)
                or not isinstance(slot_index, int)
                or not 0 <= slot_index <= head
            ):
                raise CheckpointError(
                    f"{what} ({name!r}) names an invalid shared position"
                )
            if slot_index <= previous_pos:
                raise CheckpointError(
                    f"{what} ({name!r}) lists shared positions out of order"
                )
            previous_pos = slot_index
            shared_positions.append(slot_index)
        shared_set = set(shared_positions)
        carried = {}
        next_slot = 0
        for carried_index, block_id in enumerate(block_refs):
            if (
                isinstance(block_id, bool)
                or not isinstance(block_id, int)
                or not 0 <= block_id < len(blobs)
            ):
                raise CheckpointError(
                    f"{what} ({name!r}) names no segment block at entry "
                    f"{carried_index}"
                )
            while next_slot in shared_set:
                next_slot += 1
            if next_slot > head:
                raise CheckpointError(
                    f"{what} ({name!r}) carries more blocks than non-shared "
                    "positions"
                )
            carried[next_slot] = block_id
            next_slot += 1
        if len(carried) + len(shared_positions) != head + 1:
            raise CheckpointError(
                f"{what} ({name!r}) must name one block or shared slot for "
                "every segment position 0..head"
            )
        members.append(
            {
                "name": name,
                "head": head,
                "shared": shared_positions,
                "shared_set": shared_set,
                "carried": carried,
            }
        )
    used = {
        block_id for member in members for block_id in member["carried"].values()
    }
    if used != set(range(len(blobs))):
        raise CheckpointError(
            "family sync artifact carries a segment block no member reaches"
        )
    _validate_carried_frames(members, blobs)
    return target_digest, members, blobs


def _validate_carried_frames(members, blobs):
    """Frame-level validation of every carried block at its slot."""
    seen_slots = {}
    for position, member in enumerate(members):
        for slot, block_id in member["carried"].items():
            previous = seen_slots.get(block_id)
            if previous is not None and previous != slot:
                raise CheckpointError(
                    f"family sync artifact block {block_id} fills two "
                    f"different positions ({previous} and {slot})"
                )
            seen_slots[block_id] = slot
            raw = blobs[block_id]
            try:
                if slot == 0:
                    if raw[: len(MAGIC)] != MAGIC:
                        raise CheckpointError(
                            "the block at position 0 is not a full basis "
                            "snapshot"
                        )
                    parse_bytes(raw)
                else:
                    if raw[: len(DELTA_MAGIC)] != DELTA_MAGIC:
                        raise CheckpointError(
                            f"the block at position {slot} is not a delta "
                            "segment"
                        )
                    _parse_delta(raw, slot)
            except CheckpointError as exc:
                raise CheckpointError(
                    f"family sync artifact member {position} "
                    f"({member['name']!r}) is invalid at segment "
                    f"{_segment_name(slot)}: {exc}"
                ) from exc


# ---------------------------------------------------------------------------
# Diff construction
# ---------------------------------------------------------------------------


def _build_sync_artifact(source, target, target_digest):
    """Freeze two captured families; shared-with-target blocks omitted.

    *source* / *target* are lists of ``(name, head, records)``; records
    map a segment name to ``(raw_bytes, share_key)``.  Source segments
    reached through the same share key become one artifact block.
    """
    blocks = []
    block_of_share = {}
    manifest_members = []
    for position, (name, src_head, src_records) in enumerate(source):
        _tgt_name, tgt_head, tgt_records = target[position]
        reach = []
        shared = []
        for slot in range(src_head + 1):
            seg_name = _segment_name(slot)
            raw, share_key = src_records[seg_name]
            if slot <= tgt_head and tgt_records[seg_name][0] == raw:
                shared.append(slot)
                continue
            block_id = block_of_share.get(share_key)
            if block_id is None:
                block_id = len(blocks)
                block_of_share[share_key] = block_id
                blocks.append((seg_name, raw))
            reach.append(block_id)
        manifest_members.append(
            {"name": name, "head": src_head, "segments": reach, "sh": shared}
        )
    manifest = {
        "v": _SYNC_EXPORT_VERSION,
        "target": target_digest,
        "members": manifest_members,
        "segments": [
            {"name": name, "size": len(raw), "crc": zlib.crc32(raw) & 0xFFFFFFFF}
            for name, raw in blocks
        ],
    }
    return _frame_artifact(
        SYNC_EXPORT_MAGIC, manifest, [raw for _name, raw in blocks]
    )


def _check_aligned_names(source_abs, target_abs):
    source_names = [os.path.basename(path) for path in source_abs]
    target_names = [os.path.basename(path) for path in target_abs]
    if source_names != target_names:
        raise CheckpointError(
            "a family sync aligns members in order: the member names must "
            f"agree across the families (source {source_names!r}, target "
            f"{target_names!r})"
        )
    for name in source_names:
        _check_family_member_name(name, "family sync")
    return source_names


def diff_families(source_members, target_members, artifact):
    """Pack the incremental difference between two chain families.

    *source_members* and *target_members* are equal-length sequences of
    chain directories aligned position by position; *artifact* is the
    sync artifact's destination path.  Only segments that genuinely
    differ are stored: positions the two families already hold byte for
    byte identically carry no block and an unchanged member carries no
    blocks at all.  Each member's head, its member-owned tail and its
    segment positions are recorded exactly as the source holds them;
    segments one source family physically shares are stored once.
    Producing the diff changes neither family by a byte, advances no
    member's optimizer step and is deterministic (the same two families
    diff twice produce identical artifacts).

    Members keep saving, loading and appending while the diff runs;
    each family is captured as one consistent, quiescent snapshot.  A
    missing member directory or referenced segment raises
    ``FileNotFoundError`` without touching the other family.  Differing
    member counts or names, disagreeing parameter shapes or layer
    order, or any truncated, missing-field or out-of-order segment
    reject the whole diff with ``ValueError`` before the artifact is
    written.  An unwritable destination directory or a full disk raises
    ``OSError``; the artifact is written via a temp file and atomically
    renamed.
    """
    source_paths = _normalize_dir_family(source_members, "a sync source family")
    target_paths = _normalize_dir_family(target_members, "a sync target family")
    if not isinstance(artifact, (str, os.PathLike)):
        raise TypeError("sync artifact target must be a filesystem path")
    artifact = os.fspath(artifact)
    for path in source_paths + target_paths:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"incremental checkpoint directory not found: {path!r}"
            )
    source_abs = [os.path.abspath(path) for path in source_paths]
    target_abs = [os.path.abspath(path) for path in target_paths]
    if len(set(source_abs)) != len(source_abs) or len(set(target_abs)) != len(
        target_abs
    ):
        raise CheckpointError("a chain family must not list a member twice")
    if len(source_abs) != len(target_abs):
        raise CheckpointError(
            "a family sync needs one target member per source member "
            f"({len(source_abs)} source members, {len(target_abs)} target "
            "members)"
        )
    if set(source_abs) & set(target_abs):
        raise CheckpointError(
            "a family sync diffs two distinct families: a member directory "
            "may not appear in both families"
        )
    names = _check_aligned_names(source_abs, target_abs)
    source_parent = _one_parent(source_abs, "a family sync")
    target_parent = _one_parent(target_abs, "a family sync")
    artifact_abs = os.path.abspath(artifact)
    if os.path.isdir(artifact_abs):
        raise CheckpointError(f"sync artifact target is a directory: {artifact!r}")
    artifact_parent = os.path.dirname(artifact_abs)
    if not os.path.isdir(artifact_parent):
        raise FileNotFoundError(
            f"family sync destination directory not found: {artifact_parent!r}"
        )
    _sweep_parent_staging(source_parent)
    _sweep_parent_staging(target_parent)
    if artifact_parent not in (source_parent, target_parent):
        _sweep_parent_staging(artifact_parent)

    def snapshot(abs_paths):
        with _FamilyParentGuard(os.path.dirname(abs_paths[0])):
            return _family_export_snapshot(abs_paths)

    src_heads, src_records, src_documents = snapshot(source_abs)
    tgt_heads, tgt_records, tgt_documents = snapshot(target_abs)
    _check_family_shape_agreement(
        [src_documents[path] for path in source_abs]
        + [tgt_documents[path] for path in target_abs],
        names + [f"target:{name}" for name in names],
        "family sync",
    )
    digest = _family_digest(
        names,
        {names[i]: tgt_heads[path] for i, path in enumerate(target_abs)},
        {names[i]: tgt_records[path] for i, path in enumerate(target_abs)},
    )
    source = [
        (names[i], src_heads[path], src_records[path])
        for i, path in enumerate(source_abs)
    ]
    target = [
        (names[i], tgt_heads[path], tgt_records[path])
        for i, path in enumerate(target_abs)
    ]
    raw = _build_sync_artifact(source, target, digest)
    _atomic_write(artifact_parent, os.path.basename(artifact_abs), raw)
    return None


# ---------------------------------------------------------------------------
# Family snapshot and target-side validation (directories)
# ---------------------------------------------------------------------------


def _capture_family(abs_paths):
    """One consistent, quiescent snapshot of a directory family."""
    ordered = sorted(abs_paths)
    while True:
        with contextlib.ExitStack() as stack:
            for directory in ordered:
                stack.enter_context(_DirectoryChainLock(directory))
            stores = {d: _DirectoryChainStore(d) for d in ordered}
            for directory, store in stores.items():
                _recover_directory_chain(directory, store)
            busy = [d for d in ordered if _member_fold_busy(d)]
            if not busy:
                heads = {}
                records = {}
                documents = {}
                for directory, store in stores.items():
                    head = _read_head_optional(store)
                    if head is None:
                        raise CheckpointError(
                            "chain has no head pointer (no basis segment "
                            f"committed): {directory!r}"
                        )
                    heads[directory] = head
                    records[directory], documents[directory] = (
                        _read_member_segments(directory, store, head)
                    )
                return heads, records, documents
        for directory in busy:
            _wait_for_member_quiescent(directory)


def _raw_at(records, member, slot, blobs):
    name = _segment_name(slot)
    if slot in member["shared_set"]:
        return records[name][0]
    return blobs[member["carried"][slot]]


def _walk_member(member, blobs, records):
    walker = _ChainWalker(_raw_at(records, member, 0, blobs))
    for slot in range(1, member["head"] + 1):
        walker.apply_delta(slot, _raw_at(records, member, slot, blobs))
    return walker.document()


def _validate_against_target(plan, blobs, target_paths, heads, records, documents):
    """Pre-write validation: shared slots resolve and chains walk/agree."""
    assembled = []
    for member, path in zip(plan, target_paths):
        target_head = heads[path]
        member_records = records[path]
        for slot in member["shared"]:
            if slot > target_head:
                raise CheckpointError(
                    f"family sync artifact member {member['name']!r} marks "
                    f"segment {slot} shared but the target member's head is "
                    f"{target_head}"
                )
            if _segment_name(slot) not in member_records:
                raise CheckpointError(
                    f"family sync target member {member['name']!r} is "
                    f"missing the shared segment {_segment_name(slot)}"
                )
        assembled.append(_walk_member(member, blobs, member_records))
    _check_family_shape_agreement(
        assembled + [documents[path] for path in target_paths],
        [member["name"] for member in plan]
        + [f"target:{member['name']}" for member in plan],
        "family sync",
    )


# ---------------------------------------------------------------------------
# Staging and rotation
# ---------------------------------------------------------------------------


def _family_marker_path(parent):
    return os.path.join(parent, SYNC_MARKER_NAME)


def _family_artifact_path(parent, digest):
    return os.path.join(parent, SYNC_ARTIFACT_PREFIX + digest)


def _staging_dir(parent, name):
    return os.path.join(parent, SYNC_STAGING_PREFIX + name)


def _read_family_marker(parent):
    path = _family_marker_path(parent)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            marker = json.loads(fh.read().decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict):
        return None
    if set(marker) not in (_SYNC_FAMILY_MARKER_KEYS, frozenset(("v", "d"))):
        return None
    if marker["v"] != 1 or not isinstance(marker["d"], str):
        return None
    members = marker.get("m")
    if members is not None:
        if not isinstance(members, list):
            return None
        for entry in members:
            if not isinstance(entry, dict) or set(entry) != _SYNC_FAMILY_HEAD_KEYS:
                return None
            if not isinstance(entry["n"], str):
                return None
            for key in ("h",):
                if isinstance(entry[key], bool) or not isinstance(entry[key], int):
                    return None
    return marker


def _encode_member_marker(artifact_digest, old_head, new_head, stage):
    return json.dumps(
        {
            "magic": _SYNC_MARKER_MAGIC,
            "v": 1,
            "a": artifact_digest,
            "h": old_head,
            "n": new_head,
            "s": stage,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _read_member_marker(directory):
    path = _sync_marker_path(directory)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            marker = json.loads(fh.read().decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict):
        return None
    if marker.get("magic") != _SYNC_MARKER_MAGIC:
        return None
    required = {"magic"} | _SYNC_MEMBER_MARKER_KEYS
    if set(marker) != required:
        return None
    if marker["v"] != 1 or not isinstance(marker["a"], str):
        return None
    for key in ("h", "n", "s"):
        if isinstance(marker[key], bool) or not isinstance(marker[key], int):
            return None
    return marker


def _stage_one_member(
    parent, member, live_path, blobs, canonical_blocks, canonical_inodes
):
    """Build this member's complete new chain in its private staging dir.

    Shared slots are hard-linked from the live member directory; equal
    shared segments already staged for another member are linked from
    that staging file; a carried block is written once (by the first
    member that reaches it) and hard-linked for every later member.
    Either way the post-apply family keeps one physical copy per
    distinct segment.  The staging directory is private, so no member
    byte changes here.
    """
    name = member["name"]
    staging = _staging_dir(parent, name)
    if os.path.exists(staging):
        _remove_tree_quietly(staging)
    os.mkdir(staging)
    for slot in range(member["head"] + 1):
        slot_name = _segment_name(slot)
        destination = os.path.join(staging, slot_name)
        if slot in member["shared_set"]:
            live_slot = os.path.join(live_path, slot_name)
            inode_key = None
            try:
                stat = os.stat(live_slot)
                inode_key = (stat.st_dev, stat.st_ino)
            except OSError:
                inode_key = None
            source_path = (
                canonical_inodes.get(inode_key)
                if inode_key is not None
                else None
            )
            if source_path is not None and os.path.exists(source_path):
                os.link(source_path, destination)
            else:
                os.link(live_slot, destination)
                if inode_key is not None:
                    canonical_inodes.setdefault(inode_key, destination)
        else:
            block_id = member["carried"][slot]
            source_path = canonical_blocks.get(block_id)
            if source_path is not None and os.path.exists(source_path):
                os.link(source_path, destination)
            else:
                _atomic_write(staging, slot_name, blobs[block_id])
                canonical_blocks[block_id] = destination
    _fsync_directory(staging)


def _validate_staging(parent, member):
    staging = _staging_dir(parent, member["name"])
    store = _DirectoryChainStore(staging)
    walker = _ChainWalker(store.read_segment(_segment_name(0)))
    for slot in range(1, member["head"] + 1):
        walker.apply_delta(slot, store.read_segment(_segment_name(slot)))
    walker.document()


def _rotate_one_member(parent, member, old_head, artifact_digest):
    """Switch one member from its old chain to its staged new chain.

    Runs under the member's directory lock (the same lock the rotation
    itself holds), so the staged files and the member marker describe
    one consistent resumable switch.
    The stage-2 marker is committed first (making the destructive phase
    resumable), staged files are renamed over the slots in ascending
    order, the head is advanced, and slots beyond the new head are
    dropped only afterwards; every step is idempotent, so a killed
    rotation converges here on the member's next open.
    """
    directory = os.path.join(parent, member["name"])
    staging = _staging_dir(parent, member["name"])
    new_head = member["head"]
    _atomic_write(
        directory,
        SYNC_MARKER_NAME,
        _encode_member_marker(artifact_digest, old_head, new_head, 2),
    )
    for slot in range(new_head + 1):
        slot_name = _segment_name(slot)
        staged_path = os.path.join(staging, slot_name)
        live_path = os.path.join(directory, slot_name)
        if os.path.exists(staged_path):
            os.replace(staged_path, live_path)
        # else: already promoted on an earlier attempt; the live slot is
        # the intended file (validated by the caller before rotation).
    with open(os.path.join(directory, _HEAD_NAME), "wb") as fh:
        fh.write(str(new_head).encode("ascii"))
        fh.flush()
        os.fsync(fh.fileno())
    for index in range(new_head + 1, old_head + 1):
        _unlink_quietly(os.path.join(directory, _segment_name(index)))
    _fsync_directory(directory)
    _finish_one_member(parent, member)


def _finish_one_member(parent, member):
    """Remove this member's rotation marker and staging directory."""
    directory = os.path.join(parent, member["name"])
    _unlink_quietly(_sync_marker_path(directory))
    _remove_tree_quietly(_staging_dir(parent, member["name"]))
    _fsync_directory(directory)


def _cleanup_application(parent):
    """Drop the family marker and the private artifact copy."""
    marker = _read_family_marker(parent)
    if marker is not None:
        _unlink_quietly(_family_artifact_path(parent, marker["d"]))
    _unlink_quietly(_family_marker_path(parent))
    _fsync_directory(parent)


def apply_family_diff(artifact, target):
    """Apply a family sync artifact to the target family atomically.

    *artifact* is a sync artifact file (or its bytes) produced by
    :func:`diff_families`; *target* is the existing parent directory of
    the target family.  The artifact is validated in full and the
    target family matched against the digest recorded in it before
    anything is written.  Each member's new chain is then built in a
    private staging directory (shared slots hard-linked, each carried
    block written once and hard-linked across members) and the family
    switches over with every member lock held at once: a member that
    moved during staging rejects the whole application before the first
    rename, leaving every target byte in place.  Afterwards every
    target member's parameters, gradients, optimizer state and step
    count and hidden state are bit for bit the source family's; no
    optimizer step advances, and the shared history stays one physical
    copy.

    Applying the same artifact to the family it already produced is a
    no-op.  A process killed mid-apply leaves either the old or the new
    complete chain per member plus a resume marker: a re-run converges
    to one completed application, and the residue is reclaimed
    deterministically by the next member open or family operation over
    the target directory.  Members keep saving and loading throughout
    (a save sees one complete chain, never half a chain); an append
    that lands while the family is staged is kept intact and the
    untouched application rejects with ``ValueError`` for a re-diff.

    A missing artifact, target directory or target member directory
    raises ``FileNotFoundError`` without touching the other side.  A
    torn, truncated, missing-field or reordered artifact, a
    shape/layer-order mismatch, or an artifact that does not match the
    target family rejects the whole apply with ``ValueError`` before
    one target byte is rewritten.  An unwritable directory or a full
    disk raises ``OSError``; an interrupted application leaves only
    whole files, never a half-written segment.
    """
    raw = _artifact_bytes(artifact)
    if not isinstance(target, (str, os.PathLike)):
        raise TypeError("sync target must be a directory path")
    parent = os.path.abspath(os.fspath(target))
    target_digest, plan, blobs = parse_sync_artifact(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if not os.path.isdir(parent):
        raise FileNotFoundError(
            f"family sync target directory not found: {target!r}"
        )
    with _FamilyParentGuard(parent):
        _family_register(parent)
        try:
            _resume_or_apply(parent, raw, digest, target_digest, plan, blobs)
        finally:
            _family_release(parent)
    return None


def _resume_or_apply(parent, raw, digest, target_digest, plan, blobs):
    names = [member["name"] for member in plan]
    target_paths = [os.path.join(parent, name) for name in names]
    for path in target_paths:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"family sync target member directory not found: {path!r}"
            )
    # Finish any member a killed rotation left mid-switch; then finish a
    # killed application of a *different* artifact before this one
    # starts, so its family marker never shadows the new application.
    _finish_mid_rotation_members(parent)
    prior_marker = _read_family_marker(parent)
    if prior_marker is not None and prior_marker["d"] != digest:
        _resume_family_sync(parent)
    _apply_core(parent, raw, digest, target_digest, plan, blobs, names, target_paths)


def _capture_locked_family(target_paths, stores):
    """Read heads/records/documents with every member lock already held."""
    heads = {}
    records = {}
    documents = {}
    for directory, store in zip(target_paths, stores):
        head = _read_head_optional(store)
        if head is None:
            raise CheckpointError(
                "chain has no head pointer (no basis segment "
                f"committed): {directory!r}"
            )
        heads[directory] = head
        records[directory], documents[directory] = (
            _read_member_segments(directory, store, head)
        )
    return heads, records, documents


def _apply_core(parent, raw, digest, target_digest, plan, blobs, names, target_paths):
    """The application, run with the family guard held.

    Every member lock is taken (sorted, the same total order an export
    snapshot uses) before the family is read and held through the
    rotation, so a concurrent save/load/append either completes before
    the snapshot or waits until the member stands on its new complete
    chain -- no member is ever observed as half a chain, and an apply
    that validated never finds a moved member at the switch point.
    """
    ordered_paths = sorted(target_paths)
    ordered = [
        (plan[target_paths.index(path)], path) for path in ordered_paths
    ]
    with contextlib.ExitStack() as stack:
        for _member, path in ordered:
            stack.enter_context(_DirectoryChainLock(path))
        stores = {path: _DirectoryChainStore(path) for _m, path in ordered}
        for path in ordered_paths:
            _recover_directory_chain(path, stores[path])
        heads, records, documents = _capture_locked_family(
            target_paths, [stores[path] for path in target_paths]
        )
        done, pending = _partition_members(
            plan, blobs, target_paths, heads, records
        )
        if not pending:
            _cleanup_application(parent)
            return None
        marker = _read_family_marker(parent)
        if marker is not None and marker["d"] == digest:
            # Resume of this same application: the still-pending members
            # never rotated, so they must sit exactly at the baseline
            # heads the marker recorded (an appended segment after a kill
            # would be silently lost by a rotation and is refused).
            baseline_heads = _marker_baseline_heads(marker, names)
            for member, path in pending:
                if heads[path] != baseline_heads[member["name"]]:
                    raise CheckpointError(
                        f"family sync target member {member['name']!r} "
                        "changed while a killed application was being "
                        "resumed; the member chain is left untouched -- "
                        "re-run the diff against the current target family"
                    )
        else:
            digest_heads = {
                names[i]: heads[path] for i, path in enumerate(target_paths)
            }
            digest_records = {
                names[i]: records[path] for i, path in enumerate(target_paths)
            }
            if _family_digest(names, digest_heads, digest_records) != target_digest:
                raise CheckpointError(
                    "family sync artifact does not match the target family: "
                    "the target family has changed since the diff was "
                    "produced (its member-reachable state differs from the "
                    "digest recorded in the artifact)"
                )
            _validate_against_target(
                plan, blobs, target_paths, heads, records, documents
            )
            _write_application_files(parent, digest, raw, heads, names)
        # Pending members still hold their baseline chains; validate the
        # incoming chains (shared slots live, carried slots from the
        # artifact) against exactly those members.
        _validate_pending(plan, pending, blobs, heads, records)
        try:
            _stage_pending_members(parent, done, pending, blobs)
            for member in plan:
                staging = _staging_dir(parent, member["name"])
                if os.path.isdir(staging):
                    _validate_staging(parent, member)
            # The switch itself: every member lock above is still held.
            baseline_for_rotation = (
                _marker_baseline_heads(marker, names)
                if marker is not None and marker["d"] == digest
                else {member["name"]: heads[path] for member, path in zip(plan, target_paths)}
            )
            for member, path in ordered:
                if any(path == pending_path for _m, pending_path in pending):
                    _rotate_one_member(
                        parent, member, baseline_for_rotation[member["name"]], digest
                    )
        except BaseException:
            # A failure before any member marker went down changed no
            # member byte: drop the private staging/application files so
            # no resumable marker remains.  Once a member marker exists
            # the rotation is rolled forward rather than undone.
            if not any(
                _read_member_marker(path) is not None for _m, path in pending
            ):
                _abort_application(parent, plan)
            raise
        _cleanup_application(parent)
    return None


def _marker_baseline_heads(marker, names):
    recorded = {entry["n"]: entry["h"] for entry in marker.get("m", [])}
    return recorded


def _validate_pending(plan, pending, blobs, heads, records):
    assembled = []
    labels = []
    for member, path in pending:
        member_records = records[path]
        target_head = heads[path]
        for slot in member["shared"]:
            if slot > target_head or _segment_name(slot) not in member_records:
                raise CheckpointError(
                    f"family sync cannot resolve shared segment "
                    f"{_segment_name(slot)} of member {member['name']!r}"
                )
        assembled.append(_walk_member(member, blobs, member_records))
        labels.append(member["name"])
    _check_family_shape_agreement(
        assembled, labels, "family sync"
    )


def _partition_members(plan, blobs, target_paths, heads, records):
    """Split members into already-finished and still-pending for an apply."""
    done = []
    pending = []
    for member, path in zip(plan, target_paths):
        member_records = records[path]
        if heads[path] == member["head"] and all(
            member_records[_segment_name(slot)][0]
            == _raw_at(member_records, member, slot, blobs)
            for slot in range(member["head"] + 1)
        ):
            done.append((member, path))
        else:
            pending.append((member, path))
    return done, pending


def _stage_pending_members(parent, done, pending, blobs):
    """Populate staging chains for the pending members (member locks held).

    Returns nothing; canonical maps are local.  Carried blocks reached
    by several members and equal shared segments are each written once
    and hard-linked, so the post-apply family keeps one physical copy
    per distinct segment.
    """
    canonical_blocks = {}
    canonical_inodes = {}
    for member, path in done:
        for slot, block_id in member["carried"].items():
            canonical_blocks.setdefault(
                block_id, os.path.join(path, _segment_name(slot))
            )
        for slot in member["shared"]:
            slot_path = os.path.join(path, _segment_name(slot))
            try:
                stat = os.stat(slot_path)
            except OSError:
                continue
            canonical_inodes.setdefault((stat.st_dev, stat.st_ino), slot_path)
    for member, path in pending:
        _stage_one_member(
            parent, member, path, blobs, canonical_blocks, canonical_inodes
        )


def _abort_application(parent, plan):
    for member in plan:
        _remove_tree_quietly(_staging_dir(parent, member["name"]))
    marker = _read_family_marker(parent)
    if marker is not None:
        _unlink_quietly(_family_artifact_path(parent, marker["d"]))
    _unlink_quietly(_family_marker_path(parent))
    _fsync_directory(parent)


def _write_application_files(parent, digest, raw, heads, names):
    artifact_path = _family_artifact_path(parent, digest)
    if not os.path.exists(artifact_path):
        _atomic_write(parent, os.path.basename(artifact_path), raw)
    marker = {
        "v": 1,
        "d": digest,
        "m": [{"n": name, "h": heads[os.path.join(parent, name)]} for name in names],
    }
    _atomic_write(
        parent,
        SYNC_MARKER_NAME,
        json.dumps(marker, sort_keys=True, separators=(",", ":")).encode("ascii"),
    )
    _fsync_directory(parent)


# ---------------------------------------------------------------------------
# Crash recovery: killed diff application
# ---------------------------------------------------------------------------


def _resume_family_sync(parent):
    """Finish an application a killed process left in *parent*.

    Runs while the caller holds the family guard.  With no family
    marker there is nothing to do.  Members already in their rotation
    phase are completed first from their self-sufficient member
    markers; the private artifact copy then drives the same core a
    fresh application uses for the remaining members.  Without the
    private artifact copy only the self-sufficient rotations can be
    finished; the family marker is left for an explicit re-run carrying
    the same artifact.
    """
    marker = _read_family_marker(parent)
    if marker is None:
        return
    digest = marker["d"]
    artifact_path = _family_artifact_path(parent, digest)
    try:
        with open(artifact_path, "rb") as fh:
            raw = fh.read()
    except FileNotFoundError:
        _finish_mid_rotation_members(parent)
        return
    target_digest, plan, blobs = parse_sync_artifact(raw)
    names = [member["name"] for member in plan]
    target_paths = [os.path.join(parent, name) for name in names]
    if not all(os.path.isdir(path) for path in target_paths):
        # A member was deleted while the application was dead; leave the
        # marker for an explicit re-run (after the member rotations that
        # can still be completed have been).
        _finish_mid_rotation_members(parent)
        return
    _apply_core(parent, raw, digest, target_digest, plan, blobs, names, target_paths)


def _finish_mid_rotation_members(parent):
    """Roll forward every member left with a stage-2 sync marker.

    Each rotation is completed purely from its member marker and the
    staged files, so the family marker need not be readable here.
    Members are processed in name order, each under its own directory
    lock -- the same lock a live rotation holds, so a member already
    being switched is simply waited out and never touched twice.
    """
    try:
        names = sorted(os.listdir(parent))
    except OSError:
        return
    for name in names:
        directory = os.path.join(parent, name)
        if not os.path.isdir(directory) or name.startswith("."):
            continue
        marker = _read_member_marker(directory)
        if marker is None:
            continue
        with _DirectoryChainLock(directory):
            _complete_rotation_from_marker(parent, directory, marker)


def _complete_rotation_from_marker(parent, directory, marker):
    """Idempotently finish a stage-2 rotation described by *marker*."""
    name = os.path.basename(directory)
    old_head = marker["h"]
    new_head = marker["n"]
    staging = _staging_dir(parent, name)
    for slot in range(new_head + 1):
        slot_name = _segment_name(slot)
        staged_path = os.path.join(staging, slot_name)
        live_path = os.path.join(directory, slot_name)
        if os.path.exists(staged_path):
            os.replace(staged_path, live_path)
    with open(os.path.join(directory, _HEAD_NAME), "wb") as fh:
        fh.write(str(new_head).encode("ascii"))
        fh.flush()
        os.fsync(fh.fileno())
    for index in range(new_head + 1, old_head + 1):
        _unlink_quietly(os.path.join(directory, _segment_name(index)))
    _fsync_directory(directory)
    _unlink_quietly(_sync_marker_path(directory))
    _remove_tree_quietly(staging)
    _fsync_directory(directory)


def _chain_recovery_hook(directory, store):
    """Finish a killed sync rotation on the next ordinary member open.

    Runs under the member directory lock: a live rotation holds the
    same lock and therefore cannot be observed here, so a member marker
    present at this point always belongs to a killed owner and is
    rolled deterministically forward.
    """
    marker = _read_member_marker(directory)
    if marker is None:
        return False
    parent = os.path.dirname(os.path.abspath(directory))
    _complete_rotation_from_marker(parent, directory, marker)
    return True


def _readonly_verify_hook(directory, store, head):
    """Read-only verification of a member holding a sync rotation marker.

    Before rotation (no member marker) verification walks the ordinary
    chain; this hook handles the stage-2 window by assembling the
    incoming chain from staged files where a slot is not promoted yet
    and the live slot where it is.  Nothing is written.
    """
    marker = _read_member_marker(directory)
    if marker is None:
        return None
    parent = os.path.dirname(os.path.abspath(directory))
    name = os.path.basename(directory)
    new_head = marker["n"]
    staging = _staging_dir(parent, name)

    def slot_bytes(slot):
        staged_path = os.path.join(staging, _segment_name(slot))
        if os.path.exists(staged_path):
            with open(staged_path, "rb") as fh:
                return fh.read()
        try:
            return store.read_segment(_segment_name(slot))
        except FileNotFoundError:
            raise CheckpointError(
                f"segment file {_segment_name(slot)!r} is missing from "
                "both the staging directory and the member chain"
            ) from None

    try:
        walker = _ChainWalker(slot_bytes(0))
        for slot in range(1, new_head + 1):
            walker.apply_delta(slot, slot_bytes(slot))
        walker.document()
    except CheckpointError as exc:
        raise CheckpointError(
            f"cross-family sync member chain ({name!r}) fails at segment "
            f"file {_segment_name(0)!r}..{_segment_name(new_head)!r}: {exc}"
        ) from exc
    return ChainVerification(new_head, new_head + 1)


def _family_maintenance_hook(parent):
    """Finish a killed sync application before the next family operation.

    Runs with the family parent guard already held (see
    :func:`checkpoint.family_open` and :func:`checkpoint.import_family`),
    so it neither takes the guard nor registers itself -- the outer
    scope already serialises family-wide operations.  Member rotations
    are self-sufficient (their member marker plus staged files finish
    them) and are always rolled forward; the family marker then drives
    completion of any still-pending members.
    """
    _finish_mid_rotation_members(parent)
    if _read_family_marker(parent) is None:
        return
    _resume_family_sync(parent)


def _parent_residue_sweep(parent):
    """Reclaim sync debris; never touch a resumable application's files.

    Runs under the parent directory lock.  While a family marker exists
    the staging directories and private artifact copy belong to a live
    or resumable application and are left exactly in place; without one
    everything sync-specific is unreachable debris.
    """
    if _read_family_marker(parent) is not None:
        return False
    changed = False
    try:
        names = os.listdir(parent)
    except OSError:
        return False
    for name in names:
        if name.startswith(SYNC_STAGING_PREFIX) or name.startswith(
            SYNC_ARTIFACT_PREFIX
        ):
            path = os.path.join(parent, name)
            if os.path.isdir(path):
                _remove_tree_quietly(path)
            else:
                _unlink_quietly(path)
            changed = True
    if changed:
        _fsync_directory(parent)
    return changed


register_chain_recovery_hook(_chain_recovery_hook)
register_readonly_verify_hook(_readonly_verify_hook)
register_family_guarded_maintenance_hook(_family_maintenance_hook)
register_parent_residue_sweep(_parent_residue_sweep)


# ---------------------------------------------------------------------------
# Full <-> incremental conversion (directories)
# ---------------------------------------------------------------------------


def full_to_incremental_artifact(full_artifact, target_members):
    """Convert a full family artifact into a sync artifact against *target*.

    The full artifact (``SEQFAMX1``) is given as bytes or a file path;
    *target_members* is the target family aligned with the artifact's
    member order and is only read.  Slots the target already holds byte
    for byte become shared (no stored block); every other block is
    carried, and applying the converted artifact lands bit for bit the
    state the full artifact restores.
    """
    plan, blobs = _parse_family_export(_artifact_bytes(full_artifact))
    target_abs = [
        os.path.abspath(path)
        for path in _normalize_dir_family(target_members, "a sync target family")
    ]
    for path in target_abs:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"incremental checkpoint directory not found: {path!r}"
            )
    if len(target_abs) != len(plan):
        raise CheckpointError(
            "the full artifact and the target family must have the same "
            f"member count ({len(plan)} vs {len(target_abs)})"
        )
    names = [member["name"] for member in plan]
    if [os.path.basename(path) for path in target_abs] != names:
        raise CheckpointError(
            "a family sync aligns members in order: the member names must "
            f"agree (artifact {names!r})"
        )
    _sweep_parent_staging(_one_parent(target_abs, "a family sync conversion"))
    with _FamilyParentGuard(_one_parent(target_abs, "a family sync conversion")):
        heads, records, _documents = _capture_family(target_abs)
    digest = _family_digest(
        names,
        {names[i]: heads[path] for i, path in enumerate(target_abs)},
        {names[i]: records[path] for i, path in enumerate(target_abs)},
    )
    return _sync_from_full_plan(
        plan, blobs, target_abs, names, heads, records, digest
    )


def _sync_from_full_plan(plan, blobs, target_keys, names, heads, records, digest):
    out_blocks = []
    manifest_members = []
    for position, member in enumerate(plan):
        key = target_keys[position]
        target_head = heads[key]
        target_records = records[key]
        reach = []
        shared = []
        for slot, block_id in enumerate(member["segments"]):
            raw = blobs[block_id]
            name = _segment_name(slot)
            if slot <= target_head and target_records[name][0] == raw:
                shared.append(slot)
            else:
                reach.append(len(out_blocks))
                out_blocks.append((name, raw))
        manifest_members.append(
            {"name": names[position], "head": member["head"], "segments": reach,
             "sh": shared}
        )
    manifest = {
        "v": _SYNC_EXPORT_VERSION,
        "target": digest,
        "members": manifest_members,
        "segments": [
            {"name": name, "size": len(raw), "crc": zlib.crc32(raw) & 0xFFFFFFFF}
            for name, raw in out_blocks
        ],
    }
    return _frame_artifact(
        SYNC_EXPORT_MAGIC, manifest, [raw for _name, raw in out_blocks]
    )


def incremental_to_full_artifact(sync_artifact, target_members):
    """Convert a sync artifact into a self-contained full family artifact.

    Every shared slot is materialized from the (read-only) target
    family and equal blocks across members are deduplicated as in a
    full export, so importing either artifact lands bit-for-bit
    identical member states.  An artifact that does not match the
    target family raises ``ValueError``.
    """
    target_digest, plan, blobs = parse_sync_artifact(
        _artifact_bytes(sync_artifact)
    )
    target_abs = [
        os.path.abspath(path)
        for path in _normalize_dir_family(target_members, "a sync target family")
    ]
    for path in target_abs:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"incremental checkpoint directory not found: {path!r}"
            )
    if len(target_abs) != len(plan):
        raise CheckpointError(
            "the sync artifact and the target family must have the same "
            f"member count ({len(plan)} vs {len(target_abs)})"
        )
    names = [member["name"] for member in plan]
    if [os.path.basename(path) for path in target_abs] != names:
        raise CheckpointError(
            "a family sync aligns members in order: the member names must "
            f"agree (artifact {names!r})"
        )
    parent = _one_parent(target_abs, "a family sync conversion")
    _sweep_parent_staging(parent)
    with _FamilyParentGuard(parent):
        heads, records, _documents = _capture_family(target_abs)
        actual = _family_digest(
            names,
            {names[i]: heads[path] for i, path in enumerate(target_abs)},
            {names[i]: records[path] for i, path in enumerate(target_abs)},
        )
        if actual != target_digest:
            raise CheckpointError(
                "the sync artifact does not match the given target family "
                "(its member-reachable state differs from the digest "
                "recorded in the artifact)"
            )
        return _full_from_sync_plan(plan, blobs, target_abs, names, records)


def _full_from_sync_plan(plan, blobs, target_keys, names, records):
    out_blocks = []
    block_of_raw = {}
    manifest_members = []
    for position, member in enumerate(plan):
        member_records = records[target_keys[position]]
        reach = []
        for slot in range(member["head"] + 1):
            name = _segment_name(slot)
            raw = (
                member_records[name][0]
                if slot in member["shared_set"]
                else blobs[member["carried"][slot]]
            )
            block_id = block_of_raw.get(raw)
            if block_id is None:
                block_id = len(out_blocks)
                block_of_raw[raw] = block_id
                out_blocks.append((name, raw))
            reach.append(block_id)
        manifest_members.append(
            {"name": names[position], "head": member["head"], "segments": reach}
        )
    manifest = {
        "v": _FAMILY_EXPORT_VERSION,
        "members": manifest_members,
        "segments": [
            {"name": name, "size": len(raw), "crc": zlib.crc32(raw) & 0xFFFFFFFF}
            for name, raw in out_blocks
        ],
    }
    return _frame_artifact(
        FAMILY_EXPORT_MAGIC, manifest, [raw for _name, raw in out_blocks]
    )


# ---------------------------------------------------------------------------
# In-memory family synchronization
# ---------------------------------------------------------------------------


def _memory_names(count):
    return [f"member-{index}" for index in range(count)]


def _check_memory_family(members):
    if isinstance(members, MemoryChain):
        raise TypeError("a synced family must be a sequence of MemoryChains")
    try:
        members = list(members)
    except TypeError:
        raise TypeError(
            "a synced family must be a sequence of MemoryChains"
        ) from None
    if not members:
        raise CheckpointError("a chain family needs at least one chain")
    for member in members:
        if not isinstance(member, MemoryChain):
            raise TypeError("chain family members must be MemoryChains")
    return members


def _capture_memory_family(members):
    ordered = sorted(members, key=id)
    for member in ordered:
        member._lock.acquire()
    try:
        heads = {}
        records = {}
        documents = {}
        for member in members:
            head = _read_head_optional(member)
            if head is None:
                raise CheckpointError(
                    "chain has no head pointer (no basis segment committed)"
                )
            document = _load_chain_store(member, head)
            segments = {}
            for index in range(head + 1):
                name = _segment_name(index)
                raw = member.read_segment(name)
                segments[name] = (raw, id(raw))
            key = id(member)
            heads[key] = head
            records[key] = segments
            documents[key] = document
        return heads, records, documents
    finally:
        for member in reversed(ordered):
            member._lock.release()


def _build_sync_artifact_memory(source_members, target_members, names):
    src_heads, src_records, src_documents = _capture_memory_family(source_members)
    tgt_heads, tgt_records, tgt_documents = _capture_memory_family(target_members)
    _check_family_shape_agreement(
        [src_documents[id(m)] for m in source_members]
        + [tgt_documents[id(m)] for m in target_members],
        names + [f"target:{name}" for name in names],
        "family sync",
    )
    skeys = [id(m) for m in source_members]
    tkeys = [id(m) for m in target_members]
    digest = _family_digest(
        names,
        {names[i]: tgt_heads[tkeys[i]] for i in range(len(tkeys))},
        {names[i]: tgt_records[tkeys[i]] for i in range(len(tkeys))},
    )
    source = [
        (names[i], src_heads[skeys[i]], src_records[skeys[i]])
        for i in range(len(skeys))
    ]
    target = [
        (names[i], tgt_heads[tkeys[i]], tgt_records[tkeys[i]])
        for i in range(len(tkeys))
    ]
    return _build_sync_artifact(source, target, digest)


def diff_families_memory(source_members, target_members):
    """Pack the incremental difference between two ``MemoryChain`` families.

    Same semantics as :func:`diff_families`: only genuinely differing
    segments are stored, heads, exclusive tails and positions are
    preserved, the result is a deterministic pure function of the two
    families, and neither family is mutated or stepped.  Returns the
    artifact bytes.
    """
    source_members = _check_memory_family(source_members)
    target_members = _check_memory_family(target_members)
    if len(source_members) != len(target_members):
        raise CheckpointError(
            "a family sync needs one target member per source member "
            f"({len(source_members)} source members, "
            f"{len(target_members)} target members)"
        )
    if any(s is t for s in source_members for t in target_members):
        raise CheckpointError(
            "a family sync diffs two distinct families: a member store may "
            "not appear in both families"
        )
    return _build_sync_artifact_memory(
        source_members, target_members, _memory_names(len(source_members))
    )


def apply_family_diff_memory(sync_artifact, target_members):
    """Apply a sync artifact to ``MemoryChain`` stores in place, atomically.

    Same semantics as :func:`apply_family_diff`: full validation and
    target matching precede any mutation, the application lands in one
    critical section, repeating it is a no-op, no optimizer step
    advances and a shared segment stays one shared object.
    """
    target_digest, plan, blobs = parse_sync_artifact(
        _artifact_bytes(sync_artifact)
    )
    target_members = _check_memory_family(target_members)
    if len(target_members) != len(plan):
        raise CheckpointError(
            "the sync artifact and the target family must have the same "
            f"member count ({len(plan)} vs {len(target_members)})"
        )
    names = _memory_names(len(target_members))
    keys = [id(m) for m in target_members]
    ordered = sorted(target_members, key=id)
    for member in ordered:
        member._lock.acquire()
    try:
        heads, records, documents = {}, {}, {}
        for member in target_members:
            head = _read_head_optional(member)
            if head is None:
                raise CheckpointError(
                    "chain has no head pointer (no basis segment committed)"
                )
            document = _load_chain_store(member, head)
            segments = {}
            for index in range(head + 1):
                name = _segment_name(index)
                raw = member._objects[name]
                segments[name] = (raw, id(raw))
            key = id(member)
            heads[key] = head
            records[key] = segments
            documents[key] = document
        if all(
            heads[keys[i]] == plan[i]["head"]
            and all(
                records[keys[i]][_segment_name(slot)][0]
                == _raw_at(records[keys[i]], plan[i], slot, blobs)
                for slot in range(plan[i]["head"] + 1)
            )
            for i in range(len(plan))
        ):
            return None
        actual = _family_digest(
            names,
            {names[i]: heads[keys[i]] for i in range(len(keys))},
            {names[i]: records[keys[i]] for i in range(len(keys))},
        )
        if actual != target_digest:
            raise CheckpointError(
                "family sync artifact does not match the target family: the "
                "target stores have changed since the diff was produced "
                "(their member-reachable state differs from the digest "
                "recorded in the artifact)"
            )
        assembled = [
            _walk_member(member, blobs, records[id(store)])
            for member, store in zip(plan, target_members)
        ]
        _check_family_shape_agreement(
            assembled + [documents[key] for key in keys],
            names + [f"target:{name}" for name in names],
            "family sync",
        )
        # One critical section: each store's segment map and head are
        # replaced together, so a concurrent observer sees one complete
        # chain.  Shared slots reuse the target's exact bytes object;
        # equal carried blocks reuse the same artifact blob object.
        for member, store in zip(plan, target_members):
            member_records = records[id(store)]
            objects = {}
            for slot in range(member["head"] + 1):
                objects[_segment_name(slot)] = _raw_at(
                    member_records, member, slot, blobs
                )
            store._objects = objects
            store.write_head(str(member["head"]).encode("ascii"))
    finally:
        for member in reversed(ordered):
            member._lock.release()
    return None


def full_to_incremental_artifact_memory(full_artifact, target_members):
    """Convert a full family artifact to sync bytes against memory stores."""
    plan, blobs = _parse_family_export(_artifact_bytes(full_artifact))
    target_members = _check_memory_family(target_members)
    if len(target_members) != len(plan):
        raise CheckpointError(
            "the full artifact and the target family must have the same "
            f"member count ({len(plan)} vs {len(target_members)})"
        )
    heads, records, _documents = _capture_memory_family(target_members)
    names = _memory_names(len(target_members))
    keys = [id(m) for m in target_members]
    digest = _family_digest(
        names,
        {names[i]: heads[keys[i]] for i in range(len(keys))},
        {names[i]: records[keys[i]] for i in range(len(keys))},
    )
    return _sync_from_full_plan(plan, blobs, keys, names, heads, records, digest)


def incremental_to_full_artifact_memory(sync_artifact, target_members):
    """Convert sync bytes into a self-contained full artifact (memory)."""
    target_digest, plan, blobs = parse_sync_artifact(
        _artifact_bytes(sync_artifact)
    )
    target_members = _check_memory_family(target_members)
    if len(target_members) != len(plan):
        raise CheckpointError(
            "the sync artifact and the target family must have the same "
            f"member count ({len(plan)} vs {len(target_members)})"
        )
    heads, records, _documents = _capture_memory_family(target_members)
    names = _memory_names(len(target_members))
    keys = [id(m) for m in target_members]
    actual = _family_digest(
        names,
        {names[i]: heads[keys[i]] for i in range(len(keys))},
        {names[i]: records[keys[i]] for i in range(len(keys))},
    )
    if actual != target_digest:
        raise CheckpointError(
            "the sync artifact does not match the given target stores "
            "(their member-reachable state differs from the digest "
            "recorded in the artifact)"
        )
    return _full_from_sync_plan(plan, blobs, keys, names, records)
