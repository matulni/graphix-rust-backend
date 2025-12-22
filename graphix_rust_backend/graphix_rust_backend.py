"""MBQC state vector backend."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import _statevec_backend_rs as _backend  # noqa: PLC2701
import numpy as np
from graphix.sim.base_backend import DenseState, DenseStateBackend, Matrix
from graphix.states import BasicStates
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from graphix.parameter import ExpressionOrSupportsFloat, Parameter
    from graphix.sim.data import Data


CZ_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, -1]]]],
    dtype=np.complex128,
)
CNOT_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]],
    dtype=np.complex128,
)
SWAP_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 1], [0, 0]], [[0, 0], [0, 1]]]],
    dtype=np.complex128,
)


class Statevec(DenseState):
    """Statevector object."""

    psi: Matrix

    def __init__(
        self,
        data: Data = BasicStates.PLUS,
        nqubit: int | None = None,
    ) -> None:
        """
        Initialize statevector objects.

        `data` can be:
        - a single :class:`graphix.states.State` (classical description of a quantum state)
        - an iterable of :class:`graphix.states.State` objects
        - an iterable of scalars (A 2**n numerical statevector)
        - a *graphix.statevec.Statevec* object

        If *nqubit* is not provided, the number of qubit is inferred from *data* and checked for consistency.
        If only one :class:`graphix.states.State` is provided and nqubit is a valid integer, initialize the statevector
        in the tensor product state.
        If both *nqubit* and *data* are provided, consistency of the dimensions is checked.
        If a *graphix.statevec.Statevec* is passed, returns a copy.

        Parameters
        ----------
        data : Data, optional
            input data to prepare the state. Can be a classical description or a numerical input, defaults to graphix.states.BasicStates.PLUS
        nqubit : int, optional
            number of qubits to prepare, defaults to None

        """
        if nqubit is not None and nqubit < 0:
            raise ValueError("nqubit must be a non-negative integer.")

        # if isinstance(data, Statevec):
        #     # assert nqubit is None or len(state.flatten()) == 2**nqubit
        #     if nqubit is not None and len(data.flatten()) != 2**nqubit:
        #         raise ValueError(
        #             f"Inconsistent parameters between nqubit = {nqubit} and the inferred number of qubit = {len(data.flatten())}."
        #         )
        #     self.psi = data.psi.copy()
        #     return

        # # The type
        # # list[states.State] | list[ExpressionOrSupportsComplex] | list[Iterable[ExpressionOrSupportsComplex]]
        # # would be more precise, but given a value X of type Iterable[A] | Iterable[B],
        # # mypy infers that list(X) has type list[A | B] instead of list[A] | list[B].
        # input_list: list[states.State | ExpressionOrSupportsComplex | Iterable[ExpressionOrSupportsComplex]]
        # if isinstance(data, states.State):
        #     if nqubit is None:
        #         nqubit = 1
        #     input_list = [data] * nqubit
        # elif isinstance(data, Iterable):
        #     input_list = list(data)
        # else:
        #     raise TypeError(f"Incorrect type for data: {type(data)}")

        # if len(input_list) == 0:
        #     if nqubit is not None and nqubit != 0:
        #         raise ValueError("nqubit is not null but input state is empty.")

        #     self.psi = np.array(1, dtype=np.complex128)

        # elif isinstance(input_list[0], states.State):
        #     if nqubit is None:
        #         nqubit = len(input_list)
        #     elif nqubit != len(input_list):
        #         raise ValueError("Mismatch between nqubit and length of input state.")

        #     def get_statevector(
        #         s: states.State | ExpressionOrSupportsComplex | Iterable[ExpressionOrSupportsComplex],
        #     ) -> npt.NDArray[np.complex128]:
        #         if not isinstance(s, states.State):
        #             raise TypeError("Data should be an homogeneous sequence of states.")
        #         return s.get_statevector()

        #     list_of_sv = [get_statevector(s) for s in input_list]

        #     tmp_psi = functools.reduce(lambda m0, m1: np.kron(m0, m1).astype(np.complex128), list_of_sv)
        #     # reshape
        #     self.psi = tmp_psi.reshape((2,) * nqubit)
        # # `SupportsFloat` is needed because `numpy.float64` is not an instance of `SupportsComplex`!
        # elif isinstance(input_list[0], (Expression, SupportsComplex, SupportsFloat)):
        #     if nqubit is None:
        #         length = len(input_list)
        #         if length & (length - 1):
        #             raise ValueError("Length is not a power of two")
        #         nqubit = length.bit_length() - 1
        #     elif nqubit != len(input_list).bit_length() - 1:
        #         raise ValueError("Mismatch between nqubit and length of input state")
        #     psi = np.array(input_list)
        #     # check only if the matrix is not symbolic
        #     if psi.dtype != "O" and not np.allclose(np.sqrt(np.sum(np.abs(psi) ** 2)), 1):
        #         raise ValueError("Input state is not normalized")
        #     self.psi = psi.reshape((2,) * nqubit)
        # else:
        #     raise TypeError(f"First element of data has type {type(input_list[0])} whereas Number or State is expected")
        self.psi = _backend.new_vec(nqubit, _backend.Plus)

    def __str__(self) -> str:
        """Return a string description."""
        return f"Statevec object with statevector {self.psi} and length {self.dims()}."

    @override
    def add_nodes(self, nqubit: int, data: Data) -> None:
        r"""
        Add nodes (qubits) to the state vector and initialize them in a specified state.

        Parameters
        ----------
        nqubit : int
            The number of qubits to add to the state vector.

        data : Data, optional
            The state in which to initialize the newly added nodes.

            - If a single basic state is provided, all new nodes are initialized in that state.
            - If a list of basic states is provided, it must match the length of ``nodes``, and
              each node is initialized with its corresponding state.
            - A single-qubit state vector will be broadcast to all nodes.
            - A multi-qubit state vector of dimension :math:`2^n`, where :math:`n = \mathrm{len}(nodes)`,
              initializes the new nodes jointly.

        Notes
        -----
        Previously existing nodes remain unchanged.

        """
        if data is BasicStates.PLUS:
            state = _backend.Plus
        elif data is BasicStates.ZERO:
            state = _backend.Zero
        else:
            raise NotImplementedError
        _backend.add_nodes(self.psi, nqubit, state)

    @override
    def evolve_single(self, op: Matrix, i: int) -> None:
        """
        Apply a single-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 matrix
        i : int
            qubit index

        """
        _backend.evolve(self.psi, op.astype(np.complex128), i)

    @override
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        """
        Apply a multi-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n matrix
        qargs : list of int
            target qubits' indices

        """
        raise NotImplementedError

    def dims(self) -> tuple[int, ...]:
        """Return the dimensions."""
        raise NotImplementedError

    # Note that `@property` must appear before `@override` for pyright
    @property
    @override
    def nqubit(self) -> int:
        """Return the number of qubits."""
        return _backend.get_nqubits(self.psi)

    @override
    def remove_qubit(self, qarg: int) -> None:
        r"""
        Remove a separable qubit from the system and assemble a statevector for remaining qubits.

        This results in the same result as partial trace, if the qubit *qarg* is separable from the rest.

        For a statevector :math:`\ket{\psi} = \sum c_i \ket{i}` with sum taken over
        :math:`i \in [ 0 \dots 00,\ 0\dots 01,\ \dots,\
        1 \dots 11 ]`, this method returns

        .. math::
            \begin{align}
                \ket{\psi}' =&
                    c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 00}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 00} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 01}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 01} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 10}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 10} \\
                    & + \dots \\
                    & + c_{1 \dots 1_{\mathrm{k-1}}0_{\mathrm{k}}1_{\mathrm{k+1}} \dots 11}
                    \ket{1 \dots 1_{\mathrm{k-1}}1_{\mathrm{k+1}} \dots 11},
           \end{align}

        (after normalization) for :math:`k =` qarg. If the :math:`k` th qubit is in :math:`\ket{1}` state,
        above will return zero amplitudes; in such a case the returned state will be the one above with
        :math:`0_{\mathrm{k}}` replaced with :math:`1_{\mathrm{k}}` .

        .. warning::
            This method assumes the qubit with index *qarg* to be separable from the rest,
            and is implemented as a significantly faster alternative for partial trace to
            be used after single-qubit measurements.
            Care needs to be taken when using this method.
            Checks for separability will be implemented soon as an option.

        Parameters
        ----------
        qarg : int
            qubit index

        """
        _backend.remove_qubit(self.psi, qarg)

    @override
    def entangle(self, edge: tuple[int, int]) -> None:
        """
        Connect graph nodes.

        Parameters
        ----------
        edge : tuple of int
            (control, target) qubit indices

        """
        _backend.entangle(self.psi, edge)

    def tensor(self, other: Statevec) -> None:
        r"""
        Tensor product state with other qubits.

        Results in self :math:`\otimes` other.

        Parameters
        ----------
        other : :class:`graphix.sim.statevec.Statevec`
            statevector to be tensored with self

        """
        _backend.tensor(self.psi, other.flatten())

    def cnot(self, qubits: tuple[int, int]) -> None:
        """
        Apply CNOT.

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices

        """
        _backend.cnot(self.psi, qubits[0], qubits[1])

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """
        Swap qubits.

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices

        """
        _backend.swap(self.psi, qubits)

    def normalize(self) -> None:
        """Normalize the state in-place."""

    def flatten(self) -> Matrix:
        """Return flattened statevector."""
        return _backend.get_vec(self.psi)

    @override
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        """
        Return the expectation value of single-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 operator
        loc : int
            target qubit index

        Returns
        -------
        complex : expectation value.

        """
        return _backend.expectation_value(self.psi, op, loc).real

    def expectation_value(self, op: Matrix, qargs: Sequence[int]) -> complex:
        """
        Return the expectation value of multi-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n operator
        qargs : list of int
            target qubit indices

        Returns
        -------
        complex : expectation value

        """
        raise NotImplementedError

    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> Statevec:
        """Return a copy of the state vector where all occurrences of the given variable in measurement angles are substituted by the given value."""
        raise NotImplementedError

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> Statevec:
        """Return a copy of the state vector where all occurrences of the given keys in measurement angles are substituted by the given values in parallel."""
        raise NotImplementedError


@dataclass(frozen=True)
class StatevectorBackend(DenseStateBackend[Statevec]):
    """MBQC simulator with statevector method."""

    state: Statevec = dataclasses.field(init=False, default_factory=lambda: Statevec(nqubit=0))
