# MFBdG

MFBdG solves self-consistent, spinful Bogoliubov-de Gennes mean-field equations
on square and triangular lattices, with onsite and bond interactions through
third neighbors. It supports charge, spin, normal bond correlations, and
superconducting pairing at fixed filling or fixed chemical potential.

Every calculation uses a periodic unit cell, with one BdG matrix per momentum.
`Nx`, `Ny` set the cell size; `Nkx`, `Nky` set the momentum grid and default to 1.
With `Nkx=Nky=1`, this is the finite periodic-cell calculation. Larger grids
repeat the same cell fields over a larger periodic system. There is no backend
selection. The solvers support local barriers, nonmagnetic constraints, pairing
seeds, and plotting.

## Run an example

Select **Python (MFBdG)** in Jupyter and restart the kernel after a source update.
From the project root, run:

```python
%cd D:/graduate/MFBdG
%matplotlib inline
%run MF_wrap.py Input/Square_pureSC_example.json Local
```

For a terminal using the same environment:

```powershell
D:/miniforge3/python.exe MF_wrap.py Input/Square_pureSC_example.json Local
```

Solver dependencies are listed in [requirements.txt](requirements.txt).
Plotting requires Matplotlib; notebooks require a Jupyter environment.

## Choose the cell and momentum mesh

`Nx` and `Ny` specify the repeated unit cell;
`Nkx` and `Nky` specify the momentum mesh. The corresponding periodic dimensions
are `(Nx * Nkx)` by `(Ny * Nky)`. A 6x6 cell with a 3x3 mesh therefore represents
an 18x18 periodic lattice restricted to fields that repeat every 6x6 sites.
It cannot describe all spatial patterns of an unrestricted 18x18 cell.
For Triangle, dimensions refer to the two oblique primitive directions.

Use `Boundary_Condition: "PBC"` (the default). If `"OBC"` is supplied, the
wrapper prints a message and stops before starting the calculation. With `Nkx=Nky=1`, it solves the finite periodic cell. Increasing
the mesh improves momentum sampling; increasing the cell allows longer-period
order. Mesh convergence need not be monotonic, especially at low temperature.
See the [input guide](Input/INPUT_GUIDE.md) for the examples and parameter table.

## Constraints, pairing, and CPU allocation

`order_ansatz: "full"` permits charge and spin modulation. `"pure_SC"` constrains
only local charge and spin; onsite and bond pairing remain free.
`"nonmagnetic"` constrains only spin.
Exact Lagrange-multiplier constraints and finite local barriers are available.
Local barriers default to `local_max=1e-6` and `tau=1e-5`.
Pure SC still allows both singlet and triplet **bond** pairing. `pairing_sector: "singlet"` enforces singlet pairing throughout the solve;
`pair_seed` alone only chooses an initial state. Neither a d-wave seed nor a
singlet restriction establishes the lowest-energy phase.

`diag_threads` (default `1`) is the only thread input. The solver derives the
descent capacity (two for energy descent, one for Broyden and other solvers)
and the momentum worker count from the available CPU budget. The wrapper prints
the effective allocation first. Remove legacy worker-count inputs. The
local-barrier predictor remains sequential while its momentum diagonalizations
can run in parallel. `solver` is independent of `constraint_method`:
`energy_descent` uses up to ten backtracking attempts and automatic five-update
Broyden recovery for every supported constraint method. One summary after each
recovery block reports its actual update count, total energy/objective changes,
and residual before and after. No recovery switch is required; recovery does
not guarantee convergence or a global minimum.

For constrained momentum runs, all momentum covariance blocks are retained.
Memory grows with `Nkx*Nky*(2*Nx*Ny)^2`. Each worker diagonalizes one k point
at a time and releases its eigenvectors after use. Start with the provided
small-cell examples before
combining a large cell with a dense mesh.

## Results

Results are saved under `Output/<input stem>/`. Filenames include
`_k<Nkx>x<Nky>`. In a notebook, `%run` leaves `output` available for `np.load(output)`,
including on a cache hit. Inspect `converged`, `stop_reason`, and `err`; a saved
file alone does not establish convergence. Energies and free energies are per
physical site. The wrapper reports total wrapper runtime, not time per iteration.
