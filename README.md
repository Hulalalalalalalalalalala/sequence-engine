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
  first; any mismatch, truncation, corruption or missing field rejects the
  whole checkpoint with `ValueError` -- nothing is partially applied or
  silently filled in. A missing path raises `FileNotFoundError`.

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

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

CPU only; no GPU backend.
One backward pass per forward pass.
The engine trains nothing on its own and ships no dataset.
