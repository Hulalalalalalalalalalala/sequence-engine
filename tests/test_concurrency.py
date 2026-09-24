"""Thread-safety tests: concurrent forward/backward/update/save/load."""

import os
import tempfile
import threading
import unittest

from sequence_engine import Tensor
from sequence_engine import checkpoint as cp
from sequence_engine._selftest import (
    _deep,
    _elementwise_scale_sub,
    _fixed_gradient_state,
    _fresh_stack,
    _find_path_k,
    _subtract_path,
    _SEG1,
    _SEG2,
    _total,
)


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_updates_and_loads_keep_one_consistent_state(self):
        seq, buf, base, grads = _fixed_gradient_state()
        lr, n_updates, n_loads = 0.03, 120, 60
        path = [base]  # path[0] holds the serial reference path

        def build_path():
            current = [_deep(v) for v in base]
            snapshots = [[_deep(v) for v in current]]
            for _ in range(n_updates + 2):
                current = [
                    _elementwise_scale_sub(p, g, lr) for p, g in zip(current, grads)
                ]
                snapshots.append([_deep(v) for v in current])
            return snapshots

        serial_path = build_path()
        errors = []

        def updater():
            try:
                for _ in range(n_updates):
                    seq.update(lr)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def loader():
            try:
                for _ in range(n_loads):
                    seq.load(buf)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=updater), threading.Thread(target=loader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        # Final state sits at one single position k for every parameter;
        # a partially applied update would put parameters at different k.
        observed = [p.tolist() for p in seq.parameters()]
        self.assertIsNotNone(_find_path_k(observed, serial_path))

    def test_concurrent_snapshots_are_always_whole_boundaries(self):
        seq, buf, base, grads = _fixed_gradient_state()
        lr = 0.03
        serial_path = _subtract_path(base, grads, lr, 200)
        snapshots = []
        errors = []

        def saver():
            try:
                for _ in range(300):
                    target = bytearray()
                    seq.save(target)
                    snapshots.append(bytes(target))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=saver),
            threading.Thread(target=saver),
            threading.Thread(target=lambda: [seq.update(lr) for _ in range(100)]),
            threading.Thread(target=lambda: [seq.load(buf) for _ in range(40)]),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(snapshots), 600)
        for raw in snapshots:
            reader, _ = _fresh_stack()
            reader.load(raw)
            observed = [p.tolist() for p in reader.parameters()]
            self.assertIsNotNone(
                _find_path_k(observed, serial_path),
                "a concurrent snapshot mixed two updates",
            )

    def test_load_swap_is_atomic_against_updates(self):
        # A racing (load + update) pair must leave ALL parameters at the
        # same logical state: either update applied to the base snapshot
        # or update applied to the other snapshot -- never parameters
        # taken from two different sources (a half-loaded container).
        seq, _, base, grads = _fixed_gradient_state()
        base_buf = bytearray()
        seq.save(base_buf)

        other, _ = _fresh_stack()
        out, _ = other.forward(Tensor(_SEG2))
        other.backward(_total(out))
        other_params = [p.tolist() for p in other.parameters()]
        other_grads = [p.grad.tolist() for p in other.parameters()]
        other_buf = bytearray()
        other.save(other_buf)

        lr = 0.07

        def path_from(params, grads, max_steps):
            current = [_deep(v) for v in params]
            positions = [[_deep(v) for v in current]]
            for _ in range(max_steps):
                current = [
                    _elementwise_scale_sub(p, g, lr)
                    for p, g in zip(current, grads)
                ]
                positions.append([_deep(v) for v in current])
            return positions

        # Every observed state must be a whole position on one of the two
        # serial paths (source snapshot + k updates). A half-loaded
        # container would mix the two sources and match neither path.
        valid = path_from(base, grads, 400) + path_from(
            other_params, other_grads, 400
        )
        errors = []

        def pair(which_buf):
            # Observe through save(), the documented atomic snapshot:
            # peeking at individual parameters without external locking
            # is not a consistent multi-parameter read (another load can
            # land between two parameters), whereas a snapshot always
            # fixes the whole state at one serial point.
            try:
                for _ in range(200):
                    seq.load(which_buf)
                    seq.update(lr)
                    snap = bytearray()
                    seq.save(snap)
                    reader, _ = _fresh_stack()
                    reader.load(bytes(snap))
                    observed = [p.tolist() for p in reader.parameters()]
                    if observed not in valid:
                        errors.append("observed a half-loaded / half-updated state")
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=pair, args=(bytes(base_buf),)),
            threading.Thread(target=pair, args=(bytes(other_buf),)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_concurrent_full_saves_to_same_path_land_one_complete_file(self):
        seq, _ = _fresh_stack()
        out, _ = seq.forward(Tensor(_SEG1))
        seq.backward(_total(out))
        errors = []
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.ckp")
            seq.save(path)

            def writer():
                try:
                    for _ in range(50):
                        try:
                            seq.save(path)
                        except RuntimeError:
                            pass
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            def trainer():
                try:
                    for k in range(50):
                        out, _ = seq.forward(Tensor(_SEG2 if k % 2 else _SEG1))
                        seq.backward(_total(out))
                        if k % 10 == 0:
                            seq.update(0.01)
                        try:
                            seq.save(path)
                        except RuntimeError:
                            pass
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            threads = [
                threading.Thread(target=writer),
                threading.Thread(target=writer),
                threading.Thread(target=trainer),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            # Whatever landed is a complete checkpoint, not a torn mix.
            doc = cp.load_bytes(path)
            self.assertFalse(doc["pending"])
            reader, _ = _fresh_stack()
            reader.load(path)

    def test_trainer_concurrent_with_savers_and_readers_stays_consistent(self):
        # The engine allows one in-flight segment at a time. A trainer
        # thread runs rapid forward/backward cycles while savers retry on
        # the documented segment-boundary RuntimeError and readers walk
        # parameters; nothing may corrupt and every landed save must be a
        # complete, loadable boundary checkpoint.
        seq, _ = _fresh_stack()
        errors = []

        def trainer():
            try:
                for k in range(120):
                    out, _ = seq.forward(Tensor(_SEG2 if k % 2 else _SEG1))
                    seq.backward(_total(out))
            except Exception as exc:  # pragma: no cover
                errors.append(("trainer", exc))

        def saver():
            saved = 0
            for _ in range(120):
                buf = bytearray()
                try:
                    seq.save(buf)
                    saved += 1
                except RuntimeError:
                    # Mid-segment save is the documented refusal; retry.
                    continue
                # Whatever landed must parse as one complete checkpoint.
                try:
                    reader, _ = _fresh_stack()
                    reader.load(bytes(buf))
                except Exception as exc:  # pragma: no cover
                    errors.append(("saver", exc))
            return saved

        def reader():
            try:
                for _ in range(120):
                    params = seq.parameters()
                    # Every parameter must always be a well-formed tensor.
                    for p in params:
                        self.assertTrue(isinstance(p.shape, list))
            except Exception as exc:  # pragma: no cover
                errors.append(("reader", exc))

        threads = [
            threading.Thread(target=trainer),
            threading.Thread(target=saver),
            threading.Thread(target=saver),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertFalse(seq._pending_backward)
        final = bytearray()
        seq.save(final)  # trainer ended at a boundary: save succeeds
        reader, _ = _fresh_stack()
        reader.load(bytes(final))

    def test_two_trainers_segments_are_atomic_and_serial_equivalent(self):
        # Both threads train the SAME segment, so every completed
        # backward adds one identical gradient vector g. With the segment
        # lock the fwd...bwd window is atomic, so the 2*N additions line
        # up into one deterministic serial sequence of "+ g" steps. The
        # final gradients must equal a single-threaded 2*N run bitwise,
        # no "second backward" RuntimeError may leak, and the container
        # must end at a clean boundary.
        per_thread = 40

        def train():
            for _ in range(per_thread):
                out, _ = seq.forward(Tensor(_SEG1))
                seq.backward(_total(out))

        seq, _ = _fresh_stack()
        threads = [threading.Thread(target=train), threading.Thread(target=train)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(seq._pending_backward)

        reference, _ = _fresh_stack()
        for _ in range(2 * per_thread):
            out, _ = reference.forward(Tensor(_SEG1))
            reference.backward(_total(out))
        self.assertEqual(
            [p.grad.tolist() for p in seq.parameters()],
            [p.grad.tolist() for p in reference.parameters()],
            "interleaved segments must equal one serial order of the same segments",
        )


if __name__ == "__main__":
    unittest.main()
