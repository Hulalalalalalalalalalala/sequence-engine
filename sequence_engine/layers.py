"""Reference layers implementing the three-capability protocol.

Callers supply their own layers, but the engine ships two small ones so the
self-test and the test suite have a concrete, independently checkable
implementation:

* :class:`Linear` applies ``y = x @ W + b`` over the last dimension for any
  number of leading batch/time axes.
* :class:`SimpleRNNCell` unrolls ``a_t = tanh(x_t @ W + a_{t-1} @ U + b)``
  over a leading time axis of a chunk of shape ``(T, B, in_size)`` and hands
  only the final state back as its hidden cell.

A layer's forward cache is plain state on the layer instance.  It is
overwritten by the next :meth:`forward` and dropped at the end of
:meth:`backward`, which is what keeps backpropagation inside one chunk.
"""

import math

from .tensor import Tensor, product


class Linear:
    """``y = x @ W + b`` over the last dimension."""

    def __init__(self, in_size, out_size):
        if in_size <= 0 or out_size <= 0:
            raise ValueError("Linear sizes must be positive")
        self.in_size = in_size
        self.out_size = out_size
        self.weight = Tensor.zeros((in_size, out_size))
        self.bias = Tensor.zeros((out_size,))
        self._cache = None

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x, h):
        shape = x.shape
        if len(shape) < 1 or shape[-1] != self.in_size:
            raise ValueError(
                f"Linear expected last dimension {self.in_size}, "
                f"got {shape}"
            )
        lead = shape[:-1]
        n_lead = product(lead)
        k, n = self.in_size, self.out_size
        w, b = self.weight._flat, self.bias._flat
        x_flat = x._flat
        y_flat = [0.0] * (n_lead * n)
        for row in range(n_lead):
            x0 = row * k
            y0 = row * n
            for i in range(k):
                xi = x_flat[x0 + i]
                for j in range(n):
                    y_flat[y0 + j] += xi * w[i * n + j]
            for j in range(n):
                y_flat[y0 + j] += b[j]
        y = Tensor._from_flat(y_flat, tuple(shape[:-1]) + (n,))
        # Linear has no recurrent state of its own: the cell is a scalar zero
        # placeholder so the chain stays one-cell-per-layer.
        h_new = Tensor.zeros((1,))
        self._cache = (x, n_lead)
        return y, h_new

    def backward(self, dy):
        if self._cache is None:
            raise RuntimeError("Linear.backward called without a forward pass")
        x, n_lead = self._cache
        self._cache = None
        k, n = self.in_size, self.out_size
        x_flat, dy_flat = x._flat, dy._flat
        dx_flat = [0.0] * (n_lead * k)
        dw_flat = [0.0] * (k * n)
        db_flat = [0.0] * n
        for row in range(n_lead):
            x0, y0 = row * k, row * n
            for j in range(n):
                dyj = dy_flat[y0 + j]
                db_flat[j] += dyj
                for i in range(k):
                    dw_flat[i * n + j] += x_flat[x0 + i] * dyj
            for i in range(k):
                total = 0.0
                for j in range(n):
                    total += dy_flat[y0 + j] * self.weight._flat[i * n + j]
                dx_flat[x0 + i] = total
        self.weight._accumulate_grad(Tensor._from_flat(dw_flat, (k, n)))
        self.bias._accumulate_grad(Tensor._from_flat(db_flat, (n,)))
        return Tensor._from_flat(dx_flat, x._shape)


class SimpleRNNCell:
    """Tanh RNN unrolled over the leading time axis of a ``(T, B, F)`` chunk."""

    def __init__(self, in_size, hidden_size):
        if in_size <= 0 or hidden_size <= 0:
            raise ValueError("RNN sizes must be positive")
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.weight_ih = Tensor.zeros((in_size, hidden_size))
        self.weight_hh = Tensor.zeros((hidden_size, hidden_size))
        self.bias = Tensor.zeros((hidden_size,))
        self._cache = None

    def parameters(self):
        return [self.weight_ih, self.weight_hh, self.bias]

    def forward(self, x, h):
        shape = x.shape
        if len(shape) != 3 or shape[2] != self.in_size:
            raise ValueError(
                f"SimpleRNNCell expects input shape (T, B, {self.in_size}), "
                f"got {shape}"
            )
        t_steps, batch, hidden = shape[0], shape[1], self.hidden_size
        if h is None:
            h_prev = Tensor.zeros((batch, hidden))
        else:
            if h.shape != [batch, hidden]:
                raise ValueError(
                    f"hidden state shape {h.shape} does not match "
                    f"required [{batch}, {hidden}]"
                )
            h_prev = h

        states = []
        activations = []
        inputs = []
        w_ih, w_hh, b = (
            self.weight_ih._flat,
            self.weight_hh._flat,
            self.bias._flat,
        )
        h_flat = list(h_prev._flat)
        y_flat = [0.0] * (t_steps * batch * hidden)
        for t in range(t_steps):
            x_t = x._flat[t * batch * self.in_size:(t + 1) * batch * self.in_size]
            inputs.append(x_t)
            z_flat = [0.0] * (batch * hidden)
            for bi in range(batch):
                for j in range(hidden):
                    total = b[j]
                    x0 = bi * self.in_size
                    for i in range(self.in_size):
                        total += x_t[x0 + i] * w_ih[i * hidden + j]
                    h0 = bi * hidden
                    for i in range(hidden):
                        total += h_flat[h0 + i] * w_hh[i * hidden + j]
                    z_flat[h0 + j] = total
            a_flat = [math.tanh(v) for v in z_flat]
            y_flat[t * batch * hidden:(t + 1) * batch * hidden] = a_flat
            states.append(h_flat)
            activations.append(a_flat)
            h_flat = a_flat

        y = Tensor._from_flat(y_flat, (t_steps, batch, hidden))
        h_new = Tensor._from_flat(h_flat, (batch, hidden))
        # The cache holds this chunk's activations only.  It is replaced on
        # the next forward and consumed on backward, never handed across a
        # chunk boundary.
        self._cache = {
            "inputs": inputs,
            "prev_states": states,
            "activations": activations,
            "t_steps": t_steps,
            "batch": batch,
        }
        return y, h_new

    def backward(self, dy):
        if self._cache is None:
            raise RuntimeError(
                "SimpleRNNCell.backward called without a forward pass"
            )
        cache = self._cache
        self._cache = None
        t_steps = cache["t_steps"]
        batch = cache["batch"]
        in_size, hidden = self.in_size, self.hidden_size
        inputs, prev_states = cache["inputs"], cache["prev_states"]
        activations = cache["activations"]
        w_ih = self.weight_ih._flat
        w_hh = self.weight_hh._flat

        dx_flat = [0.0] * (t_steps * batch * in_size)
        dw_ih = [0.0] * (in_size * hidden)
        dw_hh = [0.0] * (hidden * hidden)
        db = [0.0] * hidden
        dh_next = [0.0] * (batch * hidden)
        outputs_flat = dy._flat

        for t in range(t_steps - 1, -1, -1):
            x_t = inputs[t]
            h_prev = prev_states[t]
            a_t = activations[t]
            # Snapshot the gradient flowing in from step t+1 and start a
            # fresh buffer for the gradient flowing back to step t-1; using
            # one buffer for both would let this step's writes leak back
            # into its own reads.
            dh_in = dh_next
            dh_next = [0.0] * (batch * hidden)
            y0 = t * batch * hidden
            for bi in range(batch):
                base = bi * hidden
                for j in range(hidden):
                    da = outputs_flat[y0 + base + j] + dh_in[base + j]
                    aj = a_t[base + j]
                    dz = da * (1.0 - aj * aj)
                    db[j] += dz
                    for i in range(in_size):
                        dw_ih[i * hidden + j] += x_t[bi * in_size + i] * dz
                    for i in range(hidden):
                        dw_hh[i * hidden + j] += h_prev[base + i] * dz
                    for i in range(in_size):
                        dx_flat[
                            t * batch * in_size + bi * in_size + i
                        ] += dz * w_ih[i * hidden + j]
                    for i in range(hidden):
                        dh_next[base + i] += dz * w_hh[i * hidden + j]

        self.weight_ih._accumulate_grad(Tensor._from_flat(dw_ih, (in_size, hidden)))
        self.weight_hh._accumulate_grad(Tensor._from_flat(dw_hh, (hidden, hidden)))
        self.bias._accumulate_grad(Tensor._from_flat(db, (hidden,)))
        return Tensor._from_flat(dx_flat, (t_steps, batch, in_size))
