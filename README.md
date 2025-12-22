# Rust backend for Graphix

`graphix-rust-backend` is a plugin for the
[Graphix](https://github.com/TeamGraphix/graphix) library that
efficiently runs statevector simulations.

## Installation

Run `maturin develop --release` on top of a local clone
of this repository (look at the
[`maturin` homepage] for instructions to install `maturin`).

[`maturin` homepage]: https://www.maturin.rs/#usage

## Usage


[`rust_backend` branch]: https://github.com/thierry-martinez/graphix/tree/rust_backend
[`graphix`]: https://github.com/TeamGraphix/graphix/

The graphix `rust_backend` branch adds an `impl` keyword argument to
`graphix.sim.statevec.StatevectorBackend` class, which is set to
`graphix.sim.statevec.RustStatevec` by default
(legacy `graphix.sim.statevec.Statevec` is still available).

There is a simple benchmark in [`benchmarks/simple.py`] in the `graphix` `rust_backend` branch.

[`benchmarks/simple.py`]: https://github.com/thierry-martinez/graphix/blob/rust_backend/benchmarks/simple.py

The benchmark samples 20 random patterns from 16-nqubits 2-depth
circuits.  On my machine (MacBook Pro M2 2022), the rust simulator
takes around 1.6s to execute them, wheras the legacy (numpy-based)
simulator takes around 14.6s (~9x speed-up).



---
