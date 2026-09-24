# sequence-engine

Small sequence model engine with explicit truncated backpropagation, written so that truncation points and hidden-state resets are observable from the public API.

## Requirements

Python 3.11 or newer. Standard library only.

## Install

    python3 -m pip install -e .

## Run

    python3 -m sequence_engine --selftest

## Public interface

`sequence_engine.Tensor(data)` and `sequence_engine.Sequential(modules)`.
- `Tensor.tolist() -> list`, `Tensor.shape -> list[int]`.
- `Sequential.forward(batch, hidden=None) -> (output, hidden)`.
- `Sequential.backward(loss) -> None` accumulates gradients.
  `loss` is a scalar/Tensor seed or the tuple form `(output, upstream)`
  carrying the exact Tensor returned by the preceding `forward`.
  Calling `backward` without a preceding `forward`, or twice for one
  `forward`, raises `RuntimeError`. If a layer raises mid-pass the partial
  gradients are rolled back and the same `backward` may be retried; a retry
  is not a second backward.
- `Sequential.zero_grad() -> None` clears accumulated gradients.
- `Sequential.update(learning_rate) -> None` performs one in-place step
  `theta <- theta - learning_rate * grad` on every parameter. Gradients are
  not cleared; parameters without a gradient are unchanged. A non-finite or
  non-numeric learning rate raises `ValueError`.
- `Sequential.parameters() -> list[Tensor]` in registration order.
- `Sequential.save(target) -> None` fixes parameters, accumulated gradients
  (zeros for parameters that have none), the slice-boundary hidden state and
  the layer order in one versioned binary checkpoint. `target` is a
  filesystem path (`str`/`os.PathLike`) or an in-memory `bytearray`; saving
  mutates no existing state and repeated saves of the same state produce
  identical bytes. Saves happen at segment boundaries (no pending
  `backward`); saving mid-segment raises `RuntimeError`.
- `Sequential.load(source) -> hidden | None` restores a checkpoint from a
  path or bytes-like buffer and returns the restored hidden-state list for
  the next `forward(batch, hidden)`. The version, every tensor shape and the
  layer order (count, layer kinds, per-layer parameter shapes) are checked
  first; hidden-state slots are validated on the spot, even on a model that
  has never run a forward. Any mismatch, truncation, corruption or missing
  field rejects the whole checkpoint with `ValueError` -- nothing is
  partially applied or silently filled in. A missing path raises
  `FileNotFoundError`.
- `Sequential.save_incremental(directory) -> int` /
  `Sequential.load_incremental(directory) -> hidden | None` maintain an
  incremental checkpoint *chain* in a directory. Only layers whose
  parameter or gradient content changed relative to the previous chain
  entry are written; unchanged layers are stored as content references
  and copied during reassembly. Loading reassembles the layers into the
  exact full state -- the result is bitwise identical to a full `save`
  at the latest entry, including hidden state and the sign of negative
  zero. `save_incremental` returns the new sequence id (the first entry
  is 1).

## Thread safety

A single `Sequential` may be shared by threads that interleave
`forward`, `backward`, `update`, `zero_grad`, `save`, `load` and the
incremental variants. The whole `forward` ... `backward` window is one
atomic segment (a segment lock is taken by `forward` and released by the
matching `backward`); `update`, `zero_grad` and `load` run at a segment
boundary under the same ordering. Consequently any interleaving is
equivalent to some serial order of the calls: one thread's backward can
never run on another thread's activations, `update` is atomic (no other
thread can read a tensor or a (parameter, gradient) pair halfway through
the step), a `save` always fixes one complete slice boundary (never two
threads' mixed progress), and a `load` is validated fully and committed
in one critical section -- a concurrent `update`/`save`/segment never
observes a half-restored container. A `save` attempted between a
`forward` and its `backward` is refused with `RuntimeError` immediately
(the documented segment-boundary rule) rather than blocking. Two
threads saving full checkpoints to the same path leave one complete
checkpoint on disk; two threads appending to the same incremental
directory are additionally serialized by an advisory file lock and
leave exactly one complete extra entry.

## Checkpoint format versions

Files are self-describing. The current format is version 2; version 1
files are still read. A version 1 file is migrated item by item into
the version 2 document shape while it is read, and every value is
range-checked during that migration: an oversized integer, a non-finite
float, a truncated payload or a missing/added header field fails the
whole migration and rejects the file with `ValueError` before anything
is applied. The loaded document records the version its bytes were
written in (`src`: 1 for a migrated file, 2 for a native file);
re-saving a migrated checkpoint writes format version 2. An unknown
version field is rejected.

## Checkpoint file semantics

Path saves are atomic: bytes go to a temporary file in the destination
directory, which is `os.replace`d over any existing file (no backup is
kept). A process killed mid-write leaves only the rejected temporary file;
the final path either holds the previous complete checkpoint or the new
one. A half-written or corrupted file at the load path is detected by its
trailer and CRC and refused with `ValueError`.

Floats are stored as raw IEEE-754 float64 values, so a save/load round-trip
is bitwise identical, including the sign of negative zero. Integers must
fit in int64 and all serialized numbers must be finite; oversized integers
or non-finite floats raise `ValueError`. Saving into a directory that does
not exist or is not writable, or failing because the disk is full, raises
`OSError`; the caller chooses the destination path.

## Incremental chain semantics

A chain directory is created by the first `save_incremental` call and
contains:

    base.ckp        full checkpoint of the first chain entry
    00000001.delta  first delta (layer references + boundary hidden state)
    00000002.delta  only the layers that changed since entry 1
    ...
    manifest.json   atomically published index (sequence id, digests)

Each delta carries one content digest per layer and chains the CRC of
its predecessor; the manifest is written atomically after the delta.
Loading walks every link from the base, verifying each layer reference,
each changed layer's digest and the predecessor CRC. A truncated,
corrupt, missing or malformed member, or a field added to/removed from
a delta/manifest header, rejects the entire chain with `ValueError`;
reassembly never returns a partially merged document. Reassembling a
healthy chain reproduces the full checkpoint byte for byte. A missing
chain path still raises `FileNotFoundError`; a parent path that is not
a directory, an unwritable directory or a full disk still raises
`OSError`. If the process dies while an entry is being appended, the
manifest still points at the previous complete entry and the
half-written orphan delta is ignored; the next `save_incremental`
replaces the orphan and the chain continues from the last published
entry. Re-saving an unchanged model advances the chain with a delta
that carries no layer bodies; layers without parameters participate
deterministically.

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

CPU only; no GPU backend.
One backward pass per forward pass.
The engine trains nothing on its own and ships no dataset.
