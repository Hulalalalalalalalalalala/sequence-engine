"""sequence_engine: a tiny standard-library-only sequence model engine.

Public surface (see README):

* :class:`Tensor` -- nested-list CPU tensor with an accumulating ``grad``.
* :class:`Sequential` -- ordered layer container with explicit truncated
  backpropagation.

``sequence_engine.layers`` additionally ships reference layers
(:class:`Linear` and :class:`SimpleRNNCell`) implementing the
``forward`` / ``backward`` / ``parameters`` protocol that caller-supplied
layers follow; they back the built-in self-test.
"""

from .tensor import Tensor
from .sequential import Sequential

__all__ = ["Tensor", "Sequential"]
