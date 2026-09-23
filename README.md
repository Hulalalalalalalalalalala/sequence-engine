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
- `Sequential.zero_grad() -> None` clears accumulated gradients.
- `Sequential.parameters() -> list[Tensor]` in registration order.
- `Sequential.update(lr) -> None` applies one in-place step to every
  parameter: `theta <- theta - lr * g` using its accumulated gradient;
  parameters without a gradient are left untouched. Raises `ValueError`
  when `lr` is not a finite number.
- `Sequential.save(target=None)` snapshots the full training state in one
  freeze: every parameter tensor, every accumulated gradient (a parameter
  without a gradient is written as zeros), and the numeric values of the
  segment-boundary hidden state from the most recent forward. With a path
  it commits atomically (temp file plus rename, overwriting any existing
  file without a backup) and returns `None`; with no argument it returns
  the checkpoint bytes for in-memory use.
- `Sequential.load(source) -> list[Tensor | None]` restores a checkpoint
  from a path or from checkpoint bytes and returns the boundary hidden
  state (one slot per layer) to feed into the next `forward`. The restored
  model always starts at a clean segment boundary with no forward in
  flight, so after a crash the caller replays the segment forward before
  calling backward again.

`backward` accepts the scalar upstream seed directly or as a one-element
tuple, `backward((loss,))`.

## Checkpoint format and error contracts

A checkpoint is a small binary envelope: magic bytes, format version,
payload length, a SHA-256 digest, a tagged payload and a trailing length
marker. Floats are stored as raw IEEE-754 float64 bits, so values and
signed zero round-trip bitwise.

- Loading a missing path raises `FileNotFoundError`.
- A truncated, half-written, corrupted, structurally wrong or wrong-version
  checkpoint, or one missing/extra fields, is rejected wholesale with
  `ValueError`; nothing is silently filled in and no state is mutated.
- A checkpoint whose tensor shapes, parameter counts or layer ordering do
  not match the current model is rejected wholesale with `ValueError`.
- State containing an integer outside signed 64-bit range or a non-finite
  float, and an `update` call with a non-finite learning rate, raise
  `ValueError`.
- Saving to a missing directory, an unwritable location, or when the disk
  is full, raises `OSError`. The on-disk location is whatever path the
  caller passed.
- One backward pass per forward pass. Calling `backward` before any
  `forward`, or twice for one forward, raises `RuntimeError`. If a layer
  raises mid-backward the pass is not counted as consumed, so retrying
  `backward` is allowed; the retry completing is the single pass.

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

CPU only; no GPU backend.
One backward pass per forward pass.
The engine trains nothing on its own and ships no dataset.
