# MFBdG input guide

The three JSON files in `Input` use the same unit-cell solver; there is no `backend` option.
Set the cell with `Nx`, `Ny` and the momentum grid with `Nkx`, `Nky`.
Use `Boundary_Condition: "PBC"`; `"OBC"` prints a message and stops.
Run from the project root using Python (MFBdG):

```sh
python MF_wrap.py Input/Square_pureSC_example.json Local
```

In Jupyter, select **Python (MFBdG)** and use `%run` in place of `python`.
See the [project introduction](../README.md) for setup and results.

## Example inputs

| Input | Cell and grid | Purpose |
| --- | --- | --- |
| [Square_full_example.json](Square_full_example.json) | 20x20 cell, 1x1 mesh | Square, `full`, `V1 = -1`. |
| [Square_pureSC_example.json](Square_pureSC_example.json) | 4x4 cell, 5x5 mesh | Square, `pure_SC`, `V1 = -1`, local barrier. |
| [Triangle_pureSC_example.json](Triangle_pureSC_example.json) | 6x6 cell, 3x3 mesh | Triangle, `pure_SC`, `V1 = -1`, local barrier. |

All three use `T=0.01`, `n_filling=0.9`, `V1=-1`,
`solver: "energy_descent"`, and `diag_threads: 1`. Pairing starts from a mixed
seed (`pair_noise=0.05`) with `normal_noise=0.01`. The chemical potential is adjusted
to maintain the filling. The pure-SC examples use `local_max=1e-6`, `tau=1e-5`;
the full-order example omits the unused constraint settings.

The square full example is a finite periodic 20x20 cell. The square pure-SC
example represents 20x20 periodic dimensions with fields repeating every 4x4 sites.
The triangle example represents 18x18 periodic dimensions with a 6x6 repeating cell.
Set `Nkx=Nky=1` (the default) for a finite periodic cell.
All examples follow the compact layout of `Square_full_example.json`:
geometry, physics, ansatz/constraints and seeds, then solver/thread controls and plotting.
Each input runs one parameter point; no `run_mode` entry is needed.

For mesh convergence, keep the cell and other settings fixed and vary only
`Nkx`, `Nky`; compare converged branches, filling, free energy and pairing fields.

## Parameters

| Input | Meaning |
| --- | --- |
| `lattice`, `Nx`, `Ny` | Choose `"Square"` or `"Triangle"` and the number of sites along each cell direction. |
| `Boundary_Condition` | `"PBC"` (default). `"OBC"` prints an unsupported-boundary message and stops before the calculation starts. |
| `Nkx`, `Nky` | Momentum mesh dimensions (both default to 1); `Nx`, `Ny` describe the repeated cell. Physical periodic dimensions are `Nx*Nkx` by `Ny*Nky`. |
| `pairing_sector` | `"mixed"` (default) or `"singlet"` for a persistent projection. Pure SC and a singlet seed alone do not enforce this. |
| `k_cache_mb` | Full-order Bloch-cache budget in MB (default 64). Each worker diagonalizes one k point at a time and releases its eigenvectors after use. This budget does not cap total process memory; descent retains the required momentum covariance blocks. |
| `T`, `n_filling` | Temperature and target electrons per site. At `n_filling = 0.9`, the solver adjusts the chemical potential (mu) until the average density reaches 0.9. Use `n_filling: null` and `mu` for fixed chemical potential; the pure-SC local barrier requires fixed filling. |
| `t1`, `t2`, `t3`, `U`, `V1`, `V2`, `V3` | The three hopping amplitudes and three bond interactions follow the first, second, and third neighbor shells. `U` is the onsite interaction. For U and V1, V2, V3, negative values (< 0) are attractive and positive values (> 0) are repulsive. Each nonzero `V` activates normal and pairing feedback on its shell. |
| `order_ansatz` | `"full"` allows local charge, spin, and onsite pairing to vary. `"pure_SC"` constrains only the local normal block: uniform charge and zero local spin within the chosen tolerance or bound. Onsite and bond pairing remain free. `"nonmagnetic"` constrains only local spin, allowing charge modulation and onsite pairing. All three keep all four complex normal and four complex pairing correlations per active bond. |
| `constraint_method` | Omit for `"full"` (explicit `null` is also accepted). For `"pure_SC"` and `"nonmagnetic"`, the default is `"Lagrange_multiplier"` (zero constrained quantities). `"local_barrier"` instead imposes finite bounds using `local_max` for either ansatz. |
| `local_max`, `tau` | For `"nonmagnetic"` with `"local_barrier"`, `local_max` bounds the full spin-vector length at every site (default `1e-6`); charge and onsite pairing are unrestricted. For `"pure_SC"` with `"local_barrier"`, `local_max` bounds only charge deviation and spin length. Pairing is not included in the barrier. `tau` is the positive barrier strength (default `1e-5`). `"full"` also supports an `m_max` spin barrier. `local_max` is unused with exact Lagrange-multiplier constraints. |
| `normal_seed`, `pair_seed` | Normal seeds are `"uniform"`, `"fm"`, or `"afm"` (Square only). Both lattices accept random `"mixed"`, `"singlet"`, and `"triplet"` seeds, plus `"dwave"`, `"extended_s"`, and `"pwave"`. Triplet samples all three spin components and need not be nonmagnetic. Triangle also accepts `"dxy"`, `"d+id"`, `"d-id"`, `"py"`, `"p+ip"`, `"p-ip"`, `"fwave"`, and `"f_y"`. Seeds select only the starting state. |
| `normal_noise`, `pair_noise` | Initial random perturbation amplitudes. Zero disables the corresponding noise. Random pairing seeds start from zero and use `pair_noise` (default `0.05`) before any spin-sector projection. |
| `solver`, `broy_max` | Choose `"broyden"`, `"energy_descent"`, `"linear_mixing"`, or `"dfsane"`. Energy descent uses one shared solver with automatic Broyden recovery for every supported constraint method. `broy_max` sets the Broyden history depth (default `10`). |
| `alpha` | Initial mixing factor, default `0.4`, with `0 < alpha <= 1`. Linear mixing uses 60% old order and 40% newly computed order when `alpha = 0.4`. Broyden adds a history correction to this base update. Energy descent uses `alpha` in Broyden proposals; its line search determines the accepted step. DF-SANE uses its own step rule. |
| `diag_threads` | Only configurable thread count (default `1`). Energy descent derives two trial workers when the CPU budget permits; other solvers derive one. Momentum workers are allocated automatically from the remaining CPU budget, capped by the number of k points. The local-barrier predictor remains sequential. |
| `tolerate`, `maxiterate`, `Plotting` | Residual tolerance, iteration limit, and plotting switch. Set `Plotting` to `true` to plot every 100 iterations and at the end. |

All three examples use these solver, history and thread settings.

```json
"solver": "energy_descent", "alpha": 0.4, "broy_max": 10, "diag_threads": 1
```

All three examples use `alpha: 0.4`, also used when `alpha` is omitted.
To try standalone Broyden, change only `solver` to `"broyden"`.
Changing `alpha` does not change initialization or the Broyden formula.

For linear mixing, the update is `x_next = x + alpha * (F(x) - x)`, where `F(x)` is the order computed from the current BdG Hamiltonian. Smaller `alpha` gives a more damped update. With `"broyden"`, a history reset sets `alpha` to its input value plus `0.1` when the residual is at most `1e-5`, and otherwise restores the input value.

The wrapper prints CPU allocation before the input parameters. Worker counts are derived automatically; remove legacy `descent_workers` and `momentum_workers` keys. `diag_threads` also applies inside every numerical worker. Energy descent retains mandatory Broyden recovery without an extra option.

`solver` and `constraint_method` are independent choices. Every `"energy_descent"` run uses the shared iteration, backtracking and recovery logic. It tries up to **10 step sizes**: the initial step `s0`, then `s0/2`, ..., `s0/512` (at most nine halvings). Constraint callbacks can restrict `s0` and reject infeasible trials. The search stops when an acceptable trial is found.

If descent trials fail, the solver takes up to **5 feasible Broyden recovery updates**, then returns to descent. Small accepted updates or slow residual improvement do not independently trigger recovery. Recovery can increase the objective while retaining the applicable feasibility checks. It stops early if converged, at the iteration limit, or if all feasible recovery attempts fail. No fallback option is needed.

At fixed filling, the objective is `F` plus any barrier penalty `B`; at fixed chemical potential it is `F - mu*n + B`, all per site. The three momentum examples use fixed filling and do not hold `mu` constant. The ten backtracking attempts and five recovery updates are separate counts.

The console prints **one summary after each recovery block**, for example:

```text
Broyden fallback: 5/5 updates, dF/site = -2.18011e-10, dObjective/site = -2.18011e-10, residual = 2.38524e-06 -> 1.4725e-08 (descent trials rejected).
```

The changes are totals across the block; the residuals are measured before its first update and after its last. An early finish reports the actual count, such as `3/5 updates`. Saved `step_modes` and `step_sizes` record each accepted update. Check `converged`, `stop_reason` and `err`; reaching `maxiterate` or saving a file does not establish convergence.

For complete inputs, see the [Square full-order example](Square_full_example.json), [Triangle pairing example](Triangle_pureSC_example.json), and [Square local-barrier example](Square_pureSC_example.json).

To suppress only magnetization while allowing charge separation, use:

```json
"order_ansatz": "nonmagnetic",
"constraint_method": "local_barrier",
"local_max": 1e-6,
"tau": 1e-5
```

For zero local spin within numerical tolerance, use `"constraint_method": "Lagrange_multiplier"`; `local_max` is then unused. The constraint acts on all three components of `m_i = <c_i^dagger sigma c_i>` throughout the iteration, independently of `pair_seed`.

Every calculation uses the unit-cell solver. Set `Nx`, `Ny` for the repeated cell and `Nkx`, `Nky` for the momentum grid (default 1 each). Set `Boundary_Condition: "PBC"`; `"OBC"` prints a message and stops. Remove any old `backend` key. See the [project introduction](../README.md).

The notebook can continue to use `%run MF_wrap.py ...` followed by `np.load(output)`, including when a completed result is reused. Restart an existing kernel after updating the source files.
