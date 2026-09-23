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

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

CPU only; no GPU backend.
One backward pass per forward pass.
The engine trains nothing on its own and ships no dataset.
