"""sequence_engine: a small CPU-only sequence model engine.

Public interface (see README.md):

- ``Tensor(data)`` with ``tolist()`` and ``shape``.
- ``Sequential(modules)`` with ``forward``, ``backward``,
  ``set_recompute`` (bounded-memory activation replay), ``zero_grad``,
  ``parameters``, ``update`` (one in-place gradient step), ``adam_step``
  (fixed-coefficient bias-corrected Adam with checkpointed moments), and
  ``save`` / ``load`` (versioned atomic checkpoints to a path, a chain
  directory, a MemoryChain or a bytearray).
"""

from .checkpoint import MemoryChain
from .sequential import Sequential
from .tensor import Tensor

__all__ = ["Tensor", "Sequential", "MemoryChain"]
__version__ = "0.1.0"
