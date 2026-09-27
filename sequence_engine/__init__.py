"""sequence_engine: a small CPU-only sequence model engine.

Public interface (see README.md):

- ``Tensor(data)`` with ``tolist()`` and ``shape``.
- ``Sequential(modules)`` with ``forward``, ``backward``, ``zero_grad``,
  ``parameters``, ``update`` (one in-place gradient step), ``adam_step``
  (one in-place Adam step with fixed coefficients), ``set_recompute``
  (bounded-memory activation recomputation), ``save`` / ``load``
  (versioned atomic checkpoints to a path, a chain directory, a
  ``MemoryChain`` or a bytearray), ``compact`` (streaming in-place,
  crash-safe compaction of one incremental checkpoint chain, or a single
  family-wide fold over a list of member chains), ``fork``
  (deriving a branch chain that shares the source chain's prefix
  segments and then evolves independently), ``delete`` (removing a
  branch chain and deterministically reclaiming the segments no
  remaining chain can reach), ``merge`` (landing one chain's current
  state onto another by appending one target-owned delta segment),
  ``verify`` (strictly read-only verification of an incremental chain
  or a chain family) and ``export_family`` / ``import_family``
  (packing a whole chain family into one self-contained artifact --
  shared segments stored once -- and restoring it with the same
  members, segment positions, shared layout and bit-for-bit state),
  plus ``sync_family`` / ``apply_sync_family`` (incremental
  cross-family synchronization: producing an artifact that carries
  only the segments two families genuinely differ by and atomically
  landing it on the target family with bit-for-bit state, no
  duplicated shared storage and no optimizer step advanced, with
  ``family_export_to_sync`` / ``sync_to_family_export`` converting
  between the full and incremental artifact shapes).
"""

from .checkpoint import MemoryChain
from .sequential import Sequential
from .tensor import Tensor

__all__ = ["Tensor", "Sequential", "MemoryChain"]
__version__ = "0.1.0"
