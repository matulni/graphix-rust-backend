from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np
from graphix.sim.statevec import StatevectorBackend as SB_np
from graphix.transpiler import Circuit

from graphix_rust_backend import StatevectorBackend as SB_rs


def simple_random_circuit(nqubit, depth):
    r"""Generate a test circuit for benchmarking.

    This function generates a circuit with nqubit qubits and depth layers,
    having layers of CNOT and Rz gates with random placements.

    Parameters
    ----------
    nqubit : int
        number of qubits
    depth : int
        number of layers

    Returns
    -------
    circuit : graphix.transpiler.Circuit object
        generated circuit
    """
    qubit_index = list(range(nqubit))
    circuit = Circuit(nqubit)
    for _ in range(depth):
        np.random.shuffle(qubit_index)
        for j in range(len(qubit_index) // 2):
            circuit.cnot(qubit_index[2 * j], qubit_index[2 * j + 1])
        for j in range(len(qubit_index)):
            circuit.rz(qubit_index[j], 2 * np.pi * np.random.random())
    return circuit


def get_perf(f):
    start = time.perf_counter()
    f()
    end = time.perf_counter()
    return end - start


def random_pattern(nqubits, depth):
    circuit = simple_random_circuit(nqubits, depth)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.minimize_space()
    return pattern


class TimeSuite:
    def setup(self, ncircuits, nqubits, depth):
        self.patterns = [random_pattern(nqubits, depth) for _ in range(ncircuits)]

    def test_consistency(self):
        for pattern in self.patterns:
            numpy_result = pattern.simulate_pattern(backend=SB_np())
            rust_result = pattern.simulate_pattern(backend=SB_rs())
            assert np.allclose(numpy_result.flatten(), rust_result.flatten())

    def time_impl(self, backend):
        for pattern in self.patterns:
            pattern.simulate_pattern(backend=backend)


ts = TimeSuite()
ts.setup(20, 16, 2)
ts.test_consistency()


def benchmark(ts, backend):
    pr = cProfile.Profile()
    pr.enable()
    ts.time_impl(backend)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
    print(s.getvalue())


benchmark(ts, SB_np())
benchmark(ts, SB_rs())
