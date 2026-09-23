"""sequence_engine: a small CPU-only sequence model engine.

Public interface (see README.md):

- ``Tensor(data)`` with ``tolist()`` and ``shape``.
- ``Sequential(modules)`` with ``forward``, ``backward``, ``zero_grad``
  and ``parameters``.
- ``Sequential.update(lr)`` applies one in-place ``theta <- theta - lr*g``
  step to every parameter.
- ``Sequential.save(target=None)`` / ``Sequential.load(source)`` snapshot
  and restore parameters, accumulated gradients and segment-boundary
  hidden state, to a path (atomic commit) or an in-memory bytes buffer.
"""

from .sequential import Sequential
from .tensor import Tensor

__all__ = ["Tensor", "Sequential"]
__version__ = "0.1.0"
