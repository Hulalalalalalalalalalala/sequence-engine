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
  gradients are rolled back and the same `backward` may be retried; a retry
  is not a second backward.
- `Sequential.set_recompute(enabled) -> None` toggles bounded-memory
  activation recomputation for the streaming truncated backprop. With it
  on, a pending slice keeps only the slice-boundary hidden state, the
  slice input and output boundary and a parameter fingerprint; the
  per-layer activations are rebuilt by replaying the forward pass during
  `backward`, so retained memory stays independent of the total streamed
  sequence length. Forward outputs and parameter gradients are bit for
  bit identical with the switch off. Toggling the switch between a
  `forward` and its `backward`, or rewriting any parameter in that
  window, raises `RuntimeError` and leaves no half-converted state (the
  refused pass may be retried once the parameters are restored).
- `Sequential.zero_grad() -> None` clears accumulated gradients.
- `Sequential.update(learning_rate) -> None` performs one in-place step
  `theta <- theta - learning_rate * grad` on every parameter. Gradients are
  not cleared; parameters without a gradient are unchanged. A non-finite or
  non-numeric learning rate raises `ValueError`.
- `Sequential.adam_step(learning_rate) -> None` performs one in-place,
  bias-corrected Adam step with fixed coefficients (the caller supplies
  only the learning rate; the optimizer kind and every coefficient are
  fixed): `m <- 0.9*m + 0.1*g`, `v <- 0.999*v + 0.001*g*g`, then with the
  step count `t` counted from 1, `m_hat = m/(1 - 0.9**t)`,
  `v_hat = v/(1 - 0.999**t)` and
  `theta <- theta - lr * m_hat / (sqrt(v_hat) + 1e-8)`. The first and
  second moments and the step count are this entry point's own state,
  saved and restored with every checkpoint; a never-stepped container
  starts them at all-zero moments with `t == 0`. Gradients are not
  cleared. The plain `update` formula, gradient accumulation and
  `zero_grad` semantics are unchanged and the two paths never share
  state. A non-finite or non-numeric learning rate raises `ValueError`.
- `Sequential.parameters() -> list[Tensor]` in registration order.
- `Sequential.save(target) -> None` fixes parameters, accumulated gradients
  (zeros for parameters that have none), the optimizer state (step count
  plus both moment tensors), the slice-boundary hidden state and the layer
  order in one versioned binary checkpoint. `target` is a
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
  versions 1 and 2 are read and migrated item by item, with missing
  optimizer state starting at `t == 0` and all-zero moments;
  `Sequential.loaded_from_version`
  reports the version a successful load came from), every tensor and
  optimizer-moment shape and the layer order (count, layer kinds,
  per-layer parameter shapes) are
  checked first; hidden-state shapes are verified on the spot, even for a
  model that has never run a forward. Any mismatch, truncation, corruption
  or missing field rejects the whole checkpoint with `ValueError` -- nothing
  is partially applied or silently filled in. A missing path raises
  `FileNotFoundError`.

### Threading

All `Sequential` operations are serialised by one per-container lock, so
multiple threads may interleave `forward`, `backward`, `update`,
`adam_step`, `save` and `load`. Under any interleaving each parameter's
final value is one a serial execution of the same calls could have
produced: `update`/`adam_step` apply as one atomic step, a `load` is never
observed half-applied, and no `save` can read a half-written tensor or a
half-updated moment. Every snapshot corresponds to one complete slice
boundary -- never two passes half mixed. When two threads save to the same
file path the bytes on disk are always one or the other writer's complete
checkpoint (never a blend).

## Checkpoint file semantics

Path saves are atomic: bytes go to a temporary file in the destination
directory, which is `os.replace`d over any existing file (no backup is
kept). A process killed mid-write leaves only the rejected temporary file;
the final path either holds the previous complete checkpoint or the new
one. A half-written or corrupted file at the load path is detected by its
trailer and CRC and refused with `ValueError`.

Floats are stored as raw IEEE-754 float64 values, so a save/load round-trip
is bitwise identical, including the sign of negative zero. Integers must
fit in int64 (this includes the optimizer step count) and all serialized
numbers must be finite; oversized integers or non-finite floats raise
`ValueError` on save -- so a non-finite moment can never enter a checkpoint
-- and a payload that merely contains an oversized or non-finite value is
refused on load. A step count of zero additionally requires both moment
tensors to be all zeros. Saving into a directory that does not exist or is
not writable, or failing because the disk is full, raises `OSError`; the
caller chooses the destination path.

## Checkpoint format versions

The on-disk format is versioned. This build writes **version 3** and still
reads **versions 1 and 2**: an old file is migrated item by item into the
current document shape and the source version is recorded; version 3 adds
the optimizer state (step count and both moment tensors), which older files
lack and therefore start at `t == 0` with all-zero moments. If migration
fails on any item the whole file is rejected rather than silently filled
in. A file with an unknown version, or a header with a field added or
removed, is rejected with `ValueError`. A migrated state re-saves as
native version 3. Unknown (future) versions are never guessed; they are
refused.

## Incremental checkpoint chains

Passing an existing directory (or a `MemoryChain`) to `save`/`load` uses an
incremental chain instead of one self-contained file:

- the first save writes a full **basis** segment (`seg-0000000000.seqd`);
- every later save appends a delta segment naming only the tensors that
  changed, compared by encoded leaf identity (so float sign bits such as
  `-0.0` count); unchanged saves append an empty delta deterministically,
  and layers with no parameters participate like any other layer;
- each delta also carries the optimizer step count, so the Adam moments
  and `t` travel through the chain exactly like parameters and gradients;
- a `head` pointer names the newest committed segment. Each segment file is
  written completely and atomically before the head is advanced, so a crash,
  a full disk or two saves racing into one directory always leave a single
  complete chain reachable from `head`.

Loading walks the basis and every delta up to the head and reassembles the
state by layer. The reassembled state is bit for bit identical to the full
snapshot taken at the same moment. Any segment in the chain that is
truncated, corrupt, missing a field, out of order, changes the layer order
or parameter/shapes, or introduces hidden state incompletely makes the load
reject the **whole** chain with `ValueError`. A missing chain directory or a
referenced segment file that is absent raises `FileNotFoundError`; an
unwritable directory or a full disk raises `OSError`. A chain cannot change
the model's layer order or parameter shapes -- use a fresh full snapshot for
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
