"""sequence_engine: a small CPU-only sequence model engine.

Public interface (see README.md):

- ``Tensor(data)`` with ``tolist()`` and ``shape``.
- ``Sequential(modules)`` with ``forward``, ``backward``, ``zero_grad``,
  ``parameters``, ``update`` (one in-place gradient step), ``adam_step``
  (one in-place Adam step with fixed coefficients), ``set_recompute``
  (bounded-memory activation recomputation), and ``save`` / ``load``
  (versioned atomic checkpoints to a path, a chain directory, a
  ``MemoryChain`` or a bytearray).
- ``compact_chain(directory, keep_tail=1)`` merges an existing incremental
  chain directory into a fresh basis (plus a rebuilt delta tail) with
  bitwise-identical state and a crash-atomic, generation-tagged commit;
  ``MemoryChain.compact`` provides the same operation in memory.
"""

from .checkpoint import MemoryChain
from .checkpoint import compact_chain
from .sequential import Sequential
from .tensor import Tensor

__all__ = ["Tensor", "Sequential", "MemoryChain", "compact_chain"]
__version__ = "0.1.0"
