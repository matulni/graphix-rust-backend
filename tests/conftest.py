"""Pytest configuration for testing Rust backend."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from graphix.transpiler import Circuit
from numpy.random import PCG64, Generator

if TYPE_CHECKING:
    from graphix.pattern import Pattern

SEED = 25


@pytest.fixture
def fx_bg() -> PCG64:
    """Return a random number generator."""
    return PCG64(SEED)


@pytest.fixture
def fx_rng(fx_bg: PCG64) -> Generator:
    """Return a random number generator."""
    return Generator(fx_bg)


@pytest.fixture
def hadamardpattern() -> Pattern:
    circ = Circuit(1)
    circ.h(0)
    return circ.transpile().pattern
