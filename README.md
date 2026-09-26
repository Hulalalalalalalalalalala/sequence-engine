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
  is not a second backward. If that replay itself cannot rebuild the
  caches the rebuild failure is surfaced (not swallowed) as a `ValueError`
  naming the rebuild stage, chained onto the original layer exception,
  which is still the error handed to the caller -- it is neither masked
  nor replaced.
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
  place. Compaction is a streaming online fold: segments are checked one
  by one while the new basis is written out, the remaining deltas are
  converted a segment at a time in the slots the original chain already
  owned, and saves and loads proceed normally throughout -- peak on-disk
  usage never exceeds the original chain plus one new basis segment, and
  the `head` pointer names at every instant one complete chain (the old
  head or the new). The basis segment and the deltas through `up_to` (the
  current head when omitted) are folded into one new basis segment and the
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
- `Sequential.compact_family(members) -> None` folds a whole chain
  family -- all of its member directories handed in at once -- into one
  new basis segment. The basis and the delta prefix every member's head
  can reach (the segments physically shared after the forks) are folded
  together; every member's `head` then names the new history, with that
  member's own, untouched tail deltas following the new basis. Each
  member's reassembled parameters, gradients, optimizer moments and step
  count and hidden state are bit for bit identical afterwards, the fold
  advances no optimizer step (both `t = 0` and stepped chains keep their
  semantics), and every member's segment count drops deterministically by
  exactly the folded range. The folded basis is stored once (hard-linked
  across the member directories, like the prefix it replaces) and each
  member releases only its own old prefix entries, so a shared segment is
  reclaimed by reachability exactly when no member can reach it and is
  never stored twice. Repeating the fold on the same range changes
  nothing, and a family with no foldable common delta is left exactly as
  it was. Members keep saving, loading and appending throughout without
  any member ever appearing as half a chain; a process killed midway
  leaves every member loadable as one complete state and the next family
  operation (or the next open of any member) rolls the fold forward and
  reclaims the residue deterministically, serialised against forks,
  merges, deletions and single-chain compactions by the directory locks
  and leases. A missing member directory or a missing referenced segment
  raises `FileNotFoundError` and leaves the other members untouched; a
  truncated segment, a missing field, an out-of-order segment, or members
  whose parameter shapes or layer order disagree reject the whole fold
  with `ValueError` whose message names the mismatching shapes and layer
  order, before one member byte moves; an unwritable directory or a full
  disk raises `OSError`. `members` may instead be a list of `MemoryChain`
  stores, folded with identical semantics (`checkpoint.compact_family` /
  `checkpoint.compact_family_memory`).
- `Sequential.fork(source, target, up_to=None) -> None` derives a branch
  chain from an existing incremental checkpoint chain at segment `up_to`
  (the source chain's current head when omitted). The branch chain
  directory created at `target` shares every segment up to and including
  the fork point with the source chain -- the prefix files are
  hard-linked, so they are stored once -- and from then on maintains
  only its own `head` and the delta segments it appends. The two chains
  save, load, compact and verify independently and in parallel: a state
  reassembled from either chain is bit for bit identical to the state an
  unforked chain would hold, appends landing on the same segment
  position in both chains leave each other's bytes untouched, and a
  shared segment (immutable by construction) is never observed
  half-written. A shared segment's bytes are reclaimed by reachability,
  exactly when no chain's head can reach it any more -- whether the
  reference went away through a compaction or through the deletion of a
  whole branch directory -- and a process killed at any point neither
  loses a reachable segment nor leaks an unreachable one; every chain
  still standing reopens as one complete state. The fork itself is
  atomic: the branch is staged under a private sibling directory and
  renamed into place, so `target` either appears as one complete chain
  or not at all. A fork point that falls between segment positions (a
  non-integer) or names a segment the chain has not committed, and a
  `target` that already exists, raise `ValueError`; a missing source
  directory or a missing referenced segment raises `FileNotFoundError`;
  an unwritable destination or a full disk raises `OSError`. Forking a
  `MemoryChain` takes no `target` and returns the new `MemoryChain`.
- `Sequential.delete(target) -> None` deletes one branch chain of a
  family: the chain directory, its `head` and the incremental segments
  it alone owned are removed. A shared segment's bytes are reclaimed by
  reachability only -- exactly when no chain's head can reach it any
  more -- and are kept byte for byte intact while any chain still
  references them. The directory is renamed aside in one atomic step
  and then emptied, so a fork or compaction running on the same family
  at any moment observes each surviving chain as one complete state,
  and a process killed at any point leaves every surviving chain
  loadable, with the residue (a detached staging directory, or segments
  no head can reach) swept deterministically by the next fork,
  compaction or deletion. The sweep is a pure reachability function:
  repeating it changes nothing and it advances no chain's optimizer
  step count. A `target` that does not exist or was already deleted
  raises `FileNotFoundError` and leaves every other chain untouched; a
  directory that exists but has no `head`, or holds a chain that cannot
  be parsed, raises `ValueError` and removes not a single shared
  segment; an unwritable directory or a full disk raises `OSError`, and
  an interrupted teardown leaves only whole segment files behind --
  never a half-written one. Deleting a `MemoryChain` drops its head
  and segments in one critical section, and the emptied chain can be
  reused.
- `Sequential.merge(source, target) -> None` merges one chain's current
  state onto another chain (two chain directories, or two
  `MemoryChain` stores). The target keeps every segment it already had
  and exactly one new delta segment is appended after its head,
  carrying only the tensors in which the source chain's head state
  differs from the target's current state (an empty delta when the two
  already agree). Loading the target afterwards is bit for bit
  identical to loading the source at merge time; the source chain --
  its head and its segment files -- is not modified at all, and the two
  chains then evolve independently. Segments the family already shares
  are never stored twice: the appended segment belongs to the target
  alone and holds only the part the source alone owned that genuinely
  differs. Repeating the merge of the same state appends another empty
  delta and changes nothing, and the number of segments grows exactly
  with the difference (always by one segment). The merge advances no
  optimizer step: the target inherits the source's complete state, its
  step count included, and a full save right afterwards lands exactly
  the merged state. A missing source or target directory or a missing
  referenced segment raises `FileNotFoundError` and leaves every other
  chain untouched; merging a chain into itself, two chains whose
  parameter shapes or layer order disagree, or an unparseable chain
  structure rejects the whole merge with `ValueError` before one target
  byte is rewritten; an unwritable directory or a full disk raises
  `OSError`. A process killed mid-merge leaves only the old head or the
  new head reachable (both one complete chain), and the residue -- an
  orphan segment beyond the old head -- is reclaimed deterministically
  by the next fork, compaction, deletion or merge.
- `Sequential.verify(source)` verifies an existing incremental checkpoint
  chain (a chain directory or a `MemoryChain`) strictly read-only. It
  walks the basis and every delta through the `head`, checking each
  segment's completeness (framing and CRC), segment order, the basis
  reference and every tensor/layer shape, and returns a success report
  (`ok`, `head`, `segments`) for an intact chain. A truncated segment,
  missing field, out-of-order segment, or a shape/layer-order mismatch
  rejects the **whole** chain with `ValueError` whose message names the
  first bad segment's position and the reason; the chain directory is not
  modified by a single byte (a compaction interrupted on disk is
  inspected in place, not rolled forward). A missing chain directory
  raises `FileNotFoundError`; an operating-system level read failure
  raises `OSError`. `source` may also be a list of chain directories --
  a chain family whose members share prefix segments after a fork. Every
  member is then verified in turn with the same read-only walk, a sound
  family returns a family report (`ok`, `members`, `chains`), and the
  first bad segment across the family is reported with its position, the
  reason and -- for a shared segment -- every chain whose head reaches
  it.

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
prefix of the deltas into one new basis segment in place as a streaming
online compaction; see its entry in the interface list above. The fold
walks and checks segments as it writes the new basis, converts the tail a
segment at a time in the slots the chain already owned, and commits
through a stage-then-publish protocol guarded by a cross-process directory
lock and lease: concurrent saves, loads, verifies and compactions on one
directory are serialised at each step (a live fold is read straight
through rather than waited on), a compaction killed at any point is
completed by the next open, and `head` always points at one complete
chain. Peak disk usage stays within the original chain plus the single
new basis segment.

`Sequential.compact_family` (or `checkpoint.compact_family`) folds a
whole family in one call: the prefix every member shares becomes one new
basis segment stored once across the member directories (hard links),
each member's exclusive tail follows it unchanged, and every member runs
the same stage-then-publish protocol, so the family-wide fold is
crash-safe and serialised against the other family operations exactly
like a single-chain fold.

`Sequential.fork` (or `checkpoint.fork_chain`) turns one chain into a
chain family: the branch directory shares the segments up to the fork
point with the source chain (hard links, so the prefix is stored once)
and then grows its own tail under its own `head`. Every chain of the
family saves, loads, compacts and verifies independently; a shared
segment's bytes are reclaimed exactly when no chain's head can reach it
any more. `Sequential.delete` (or `checkpoint.delete_chain`) removes one
branch of the family -- its directory, its head and the segments it
alone owned -- atomically with respect to concurrent forks and
compactions, and deterministically sweeps crash residue (staging
directories of killed forks/deletes, segments no head can reach) on
every fork, compaction, deletion and merge. `Sequential.merge` (or
`checkpoint.merge_chains`) lands one chain's current state onto another
inside the family: the target keeps all of its segments and gains one
target-owned delta carrying only the genuine difference from the
source, so the target then loads to the source's state bit for bit
while the source is untouched and the shared prefix is stored once;
the merge uses the same segment-then-head append commit a save does,
so a kill leaves only the old or the new head and its orphan residue
is swept on the next family operation. `checkpoint.verify_chain` (or
`Sequential.verify`) accepts a list of chain directories to verify a
whole family read-only, reporting the first bad segment with every chain
that reaches it.

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
