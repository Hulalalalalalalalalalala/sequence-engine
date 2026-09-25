# sequence-engine

Small sequence model engine with explicit truncated backpropagation, written so that truncation points and hidden-state resets are observable from the public API.

## Requirements

Python 3.11 or newer. Standard library only.

## Install

    python3 -m pip install -e .

## Run

    python3 -m sequence_engine --selftest

The built-in self-check runs only in memory (including the incremental
checkpoint path) and writes no files.

## Public interface

`sequence_engine.Tensor(data)` and `sequence_engine.Sequential(modules)`.
- `Tensor.tolist() -> list`, `Tensor.shape -> list[int]`.
- `Sequential.forward(batch, hidden=None) -> (output, hidden)`.
- `Sequential.backward(loss) -> None` accumulates gradients.
  `loss` is a scalar/Tensor seed or the tuple form `(output, upstream)`
  carrying the exact Tensor returned by the preceding `forward`.
  Calling `backward` without a preceding `forward`, or twice for one
  `forward`, raises `RuntimeError`. If a layer raises mid-pass the partial
  gradients are rolled back, the layer caches are restored by replaying
  the segment's forward, and the same `backward` may be retried; a retry
  is not a second backward.
- `Sequential.zero_grad() -> None` clears accumulated gradients.
- `Sequential.update(learning_rate) -> None` performs one in-place step
  `theta <- theta - learning_rate * grad` on every parameter. Gradients are
  not cleared; parameters without a gradient are unchanged. A non-finite or
  non-numeric learning rate raises `ValueError`.
- `Sequential.adam_step(learning_rate) -> None` performs one in-place Adam
  step with coefficients fixed by the engine (they are not parameters of
  the call and the optimizer kind cannot be swapped): moments are updated
  as `m <- 0.9*m + 0.1*g` and `v <- 0.999*v + 0.001*g^2`, the step count
  `t` starts at 1 on the first call and increments by one each call, and
  the parameter update is
  `theta <- theta - lr * (m/(1-0.9^t)) / (sqrt(v/(1-0.999^t)) + 1e-8)`.
  Parameters without a gradient are unchanged but `t` still advances;
  gradients are not cleared. The one-step `update`, gradient accumulation
  and `zero_grad` semantics are unchanged and the two optimizer paths do
  not share any state besides the parameters. A non-finite or non-numeric
  learning rate raises `ValueError`; an unstepped state starts at `t = 0`
  with zero moments.
- `Sequential.set_recompute(enabled) -> None` switches bounded-memory mode
  on or off. With it on, a `forward` keeps only the slice-boundary hidden
  state plus a few anchors (the segment input and the incoming hidden
  values); the matching `backward` recomputes the activations from those
  anchors, so retained activation memory stays bounded by the segment
  size instead of growing with the total sequence length. Forward
  outputs and parameter gradients are bitwise identical with the switch
  on or off. The switch may only be flipped between segments; flipping it
  while a backward is pending raises `RuntimeError` without touching the
  container. If a parameter is rewritten while a recompute segment is in
  flight, `backward` raises `RuntimeError` before any gradient is
  accumulated (no half state; the failed pass does not consume the
  forward and may be retried on a fresh forward).
- `Sequential.parameters() -> list[Tensor]` in registration order.
- `Sequential.save(target) -> None` fixes parameters, accumulated gradients
  (zeros for parameters that have none), the optimizer state (Adam first
  and second moments and step count; zero moments and `t = 0` when
  `adam_step` has never run), the slice-boundary hidden state and the
  layer order in one versioned binary checkpoint. `target` is a
  filesystem path (`str`/`os.PathLike`), an existing directory used as an
  incremental checkpoint chain, a `MemoryChain`, or an in-memory
  `bytearray`; saving mutates no existing state and repeated saves of the
  same state produce identical bytes. A boundary exists before the first
  segment and after every `forward` -- the hidden state returned by that
  forward *is* the boundary -- so saving is allowed even when the latest
  forward has not been back-propagated yet; the snapshot fixes the boundary
  state, not the in-flight activations.
- `Sequential.load(source) -> hidden | None` restores a checkpoint from a
  path, an incremental-chain directory, a `MemoryChain`, or bytes-like
  buffer and returns the restored hidden-state list for the next
  `forward(batch, hidden)`. The format version (the previous format
  versions 1 and 2 are read and migrated item by item; a migrated
  optimizer state starts at `t = 0` with zero moments;
  `Sequential.loaded_from_version` reports the version a successful load
  came from), every tensor shape and the layer order (count, layer
  kinds, per-layer parameter shapes) are checked first; hidden-state
  shapes are verified on the spot, even for a model that has never run a
  forward. Any mismatch, truncation, corruption or missing field rejects
  the whole checkpoint with `ValueError` -- nothing is partially applied
  or silently filled in. A missing path raises `FileNotFoundError`.
- `Sequential.compact(target, up_to=None) -> None` compacts an existing
  incremental checkpoint chain (a chain directory or a `MemoryChain`) in
  place: the basis segment and the deltas through `up_to` (the current
  head when omitted) are folded into one new basis segment and the
  remaining deltas are renumbered after it. The reassembled state --
  parameters, gradients, optimizer moments and step count, hidden state
  -- is bit for bit identical before and after, compaction advances no
  optimizer step, and the segment count decreases deterministically by
  exactly the merged range. Repeating the same compaction is a no-op, as
  is a chain with nothing to merge (basis only, or `up_to=0`).
  Old-version segments participate exactly as on load, and the compacted
  chain is rewritten in the current format version. A process killed
  mid-compaction leaves either the old or the new head reachable -- the
  next open of the chain finishes the roll-forward -- so the directory
  always holds one complete chain. A missing directory raises
  `FileNotFoundError`, an unwritable directory or a full disk raises
  `OSError`, and any corrupt, truncated or shape-inconsistent segment
  rejects the whole compaction with `ValueError` before anything is
  written.

### Threading

All `Sequential` operations are serialised by one per-container lock, so
multiple threads may interleave `forward`, `backward`, `update`,
`adam_step`, `save` and `load`. Under any interleaving each parameter's
final value is one a serial execution of the same calls could have
produced: `update` and `adam_step` each apply as one atomic step, a
`load` is never observed half-applied, and no `save` can read a
half-written tensor (optimizer moments included). Every snapshot
corresponds to one complete slice boundary -- never two passes half
mixed. When two threads save to the same file path the bytes on disk are
always one or the other writer's complete checkpoint (never a blend).

## Checkpoint file semantics

Path saves are atomic: bytes go to a temporary file in the destination
directory, which is `os.replace`d over any existing file (no backup is
kept). A process killed mid-write leaves only the rejected temporary file;
the final path either holds the previous complete checkpoint or the new
one. A half-written or corrupted file at the load path is detected by its
trailer and CRC and refused with `ValueError`.

Floats are stored as raw IEEE-754 float64 values, so a save/load round-trip
is bitwise identical, including the sign of negative zero. Integers must
fit in int64 (the Adam step count included) and all serialized numbers
must be finite; oversized integers or non-finite floats raise
`ValueError` on save -- optimizer moments with such values can never
enter a checkpoint -- and a payload that merely contains an oversized or
non-finite value is refused on load. Saving into a directory that does
not exist or is not writable, or failing because the disk is full,
raises `OSError`; the caller chooses the destination path.

## Checkpoint format versions

The on-disk format is versioned. This build writes **version 3** and
still reads **versions 1 and 2**: an old file is migrated item by item
into the current document shape and the source version is recorded; if
migration fails on any item the whole file is rejected rather than
silently filled in. Version 3 adds the optimizer state (Adam first/second
moments and step count); a version-1 or version-2 checkpoint has no such
state and migrates to `t = 0` with zero moments. A file with an unknown
version, or a header with a field added or removed, is rejected with
`ValueError`. A migrated state re-saves as native version 3. Unknown
(future) versions are never guessed; they are refused.

## Incremental checkpoint chains

Passing an existing directory (or a `MemoryChain`) to `save`/`load` uses an
incremental chain instead of one self-contained file:

- the first save writes a full **basis** segment (`seg-0000000000.seqd`);
- every later save appends a delta segment naming only the tensors that
  changed, compared by encoded leaf identity (so float sign bits such as
  `-0.0` count); unchanged saves append an empty delta deterministically,
  and layers with no parameters participate like any other layer;
- a `head` pointer names the newest committed segment. Each segment file is
  written completely and atomically before the head is advanced, so a crash,
  a full disk or two saves racing into one directory always leave a single
  complete chain reachable from `head`. A training process killed at any
  point therefore leaves a chain that loads to exactly the last complete
  commit, and continuing from it -- appending deltas or taking full saves
  -- stays bit for bit identical to the uninterrupted run.

`Sequential.compact` (or `checkpoint.compact_chain`) folds the basis and a
prefix of the deltas into one new basis segment in place; see its entry in
the interface list above. Compaction commits through a stage-then-roll-
forward protocol guarded by a cross-process directory lock: concurrent
saves, loads and compactions on one directory are serialised, a compaction
killed at any point is completed by the next open, and `head` always
points at one complete chain.

Loading walks the basis and every delta up to the head and reassembles the
state by layer (parameters, gradients, optimizer moments and step count,
then hidden slots). The reassembled state is bit for bit identical to
the full snapshot taken at the same moment, including a chain written by
the previous version-2 engine: old deltas are read, their hidden indices
remapped, and the missing optimizer state filled in from the starting
values. Any segment in the chain that is truncated, corrupt, missing a
field, out of order, changes the layer order or parameter/shapes, or
introduces hidden state incompletely makes the load reject the **whole**
chain with `ValueError`. A missing chain directory or a referenced
segment file that is absent raises `FileNotFoundError`; an unwritable
directory or a full disk raises `OSError`. A chain cannot change the
model's layer order or parameter shapes -- use a fresh full snapshot for
a different model.

`MemoryChain` (exported from `sequence_engine`) is an in-memory chain with
identical commit semantics, useful for tests and long-running processes that
want incremental snapshots without files.

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

CPU only; no GPU backend.
One backward pass per forward pass.
The engine trains nothing on its own and ships no dataset.
