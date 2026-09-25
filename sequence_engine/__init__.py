"""sequence_engine: a small CPU-only sequence model engine.

Public interface (see README.md):

- ``Tensor(data)`` with ``tolist()`` and ``shape``.
- ``Sequential(modules)`` with ``forward``, ``backward``, ``zero_grad``,
  ``parameters``, ``update`` (one in-place gradient step), ``adam_step``
  (one in-place Adam step with fixed coefficients), ``set_recompute``
  (bounded-memory activation recomputation), ``save`` / ``load``
  (versioned atomic checkpoints to a path, a chain directory, a
  ``MemoryChain`` or a bytearray), ``compact`` (streaming in-place,
  crash-safe compaction of an incremental checkpoint chain), ``fork``
  (deriving a branch chain that shares the source chain's prefix
  segments and then evolves independently), ``delete_branch``
  (removing one chain of a family while shared segments stay reachable
  through the chains that still reference them) and ``verify``
  (strictly read-only verification of an incremental chain or a chain
  family).
"""

from .checkpoint import MemoryChain
from .sequential import Sequential
from .tensor import Tensor

__all__ = ["Tensor", "Sequential", "MemoryChain"]
__version__ = "0.1.0"
