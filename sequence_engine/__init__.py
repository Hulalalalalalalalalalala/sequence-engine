"""sequence_engine: a small CPU-only sequence model engine.

Public interface (see README.md):

- ``Tensor(data)`` with ``tolist()`` and ``shape``.
- ``Sequential(modules)`` with ``forward``, ``backward``, ``zero_grad``
  and ``parameters``.
"""

from .sequential import Sequential
from .tensor import Tensor

__all__ = ["Tensor", "Sequential"]
__version__ = "0.1.0"
