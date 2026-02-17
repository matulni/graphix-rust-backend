"""MBQC state vector backend simulator in Rust."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, NewType

import numpy as np
from graphix.sim.base_backend import DenseState, DenseStateBackend, Matrix
from graphix.sim.statevec import Statevec
from graphix.states import BasicStates
from typing_extensions import override

import graphix_rust_backend._graphix_rust_backend as _backend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphix.sim.data import Data

PyCapsule = NewType("PyCapsule", object)


class StatevecRust(DenseState):
    """Statevector object with Rust backend."""

    psi: PyCapsule

    def __init__(
        self,
        data: Data = BasicStates.PLUS,
        nqubit: int | None = None,
    ) -> None:
        """Initialize statevector objects.

        See :class:`graphix.sim.statevec.Statevec` for additional information.

        Parameters
        ----------
        data : Data, optional
            Input data to prepare the state. Can be a classical description or a numerical input, defaults to 1graphix.states.BasicStates.PLUS`
        nqubit : int, optional
            Number of qubits to prepare, defaults to ``None``
        """
        sv_graphix = Statevec(data, nqubit)
        self.psi = _backend.from_vec(sv_graphix.flatten())

    def __str__(self) -> str:
        """Return a string description."""
        sv = self.flatten()
        return f"Statevec object with statevector {sv} and length {len(sv)}."

    # Note that `@property` must appear before `@override` for pyright
    @property
    @override
    def nqubit(self) -> int:
        """Return the number of qubits."""
        return _backend.get_nqubits(self.psi)

    @override
    def flatten(self) -> Matrix:
        """Return flattened state."""
        return _backend.get_vec(self.psi)

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
            - A multi-qubit state vector of dimension :math:`2^n`, where :math:`n = \mathrm{len}(nodes)`, initializes the new nodes jointly.

        Notes
        -----
        Previously existing nodes remain unchanged.
        """
        sv_to_add = StatevecRust(nqubit=nqubit, data=data)
        self.tensor(sv_to_add)

    @override
    def entangle(self, edge: tuple[int, int]) -> None:
        """Connect graph nodes.

        Parameters
        ----------
        edge : tuple of int
            (control, target) qubit indices
        """
        _backend.entangle(self.psi, edge)

    @override
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        """Apply a multi-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n matrix
        qargs : list of int
            target qubits' indices
        """
        raise NotImplementedError

    @override
    def evolve_single(self, op: Matrix, i: int) -> None:
        """Apply a single-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 matrix
        i : int
            qubit index
        """
        _backend.evolve(self.psi, op.astype(np.complex128), i)

    @override
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        """Return the expectation value of single-qubit operator.

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
        return _backend.expectation_value(self.psi, op, loc)

    @override
    def remove_qubit(self, qarg: int) -> None:
        """Remove a separable qubit from the system."""
        assert not np.isclose(_backend.norm(self.psi), 0)
        _backend.remove_qubit(self.psi, qarg)

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """Swap qubits.

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        _backend.swap(self.psi, qubits)

    def tensor(self, other: StatevecRust) -> None:
        r"""Tensor product state with other qubits.

        Results in ``self`` :math:`\otimes` ``other``.

        Parameters
        ----------
        other : :class:`graphix.sim.statevec.Statevec`
            Statevector to be tensored with ``self``.
        """
        _backend.tensor(self.psi, other.flatten())

    def cnot(self, qubits: tuple[int, int]) -> None:
        """Apply CNOT.

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices.
        """
        _backend.cnot(self.psi, qubits[0], qubits[1])


@dataclass(frozen=True)
class StatevectorRustBackend(DenseStateBackend[StatevecRust]):
    """MBQC state vector backend simulator in Rust."""

    state: StatevecRust = dataclasses.field(init=False, default_factory=lambda: StatevecRust(nqubit=0))
