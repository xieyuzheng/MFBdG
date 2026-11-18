# MFBdG input guide

Each JSON file in this folder defines one fixed-filling, real-space Bogoliubov-de Gennes calculation. To run the square example from the project root.

```sh
python MF_wrap.py Square_example Local
```

| Input | Meaning |
| --- | --- |
| `lattice`, `Nx`, `Ny`, `Boundary_Condition` | Choose `"Square"` or `"Triangle"` and the number of sites along each lattice direction. Use `"PBC"` for periodic boundaries. |
| `T`, `n_filling` | Temperature and target electrons per site. At `n_filling = 0.9`, the solver adjusts the chemical potential (mu) until the average density reaches 0.9. |
| `t1`, `t2`, `t3`, `U`, `V1`, `V2`, `V3` | The three hopping amplitudes and three bond interactions follow the first, second, and third neighbor shells. `U` is the onsite interaction. For U and V1, V2, V3, negative values (< 0) are attractive and positive values (> 0) are repulsive. Each nonzero `V` activates normal and pairing feedback on its shell. |
| `order_ansatz` | `"full"` allows local charge, spin, and onsite pairing to vary. `"pure_SC"` constrains charge to be uniform and local spin and onsite pairing to vanish within the chosen tolerance or bound. `"nonmagnetic"` constrains only local spin, allowing charge modulation and onsite pairing. All three keep all four complex normal and four complex pairing correlations per active bond. |
| `constraint_method` | Use `null` with `"full"`. For `"pure_SC"` and `"nonmagnetic"`, the default is `"Lagrange_multiplier"` (zero constrained quantities). `"local_barrier"` instead imposes finite bounds using `local_max` for either ansatz. |
| `local_max`, `tau` | For `"nonmagnetic"` with `"local_barrier"`, `local_max` bounds the full spin-vector length at every site (default `1e-6`); charge and onsite pairing are unrestricted. For `"pure_SC"` with `"local_barrier"`, `local_max` bounds charge deviation, spin length, and onsite pair magnitude. `tau` is the positive barrier strength (default `1e-4`). The legacy `"full"` plus `m_max` spin barrier remains supported. `local_max` is unused with exact Lagrange-multiplier constraints. |
| `normal_seed`, `pair_seed` | Normal seeds are `"uniform"`, `"fm"`, or `"afm"` (Square only). Both lattices accept random `"mixed"`, `"singlet"`, and `"triplet"` seeds, plus `"dwave"`, `"extended_s"`, and `"pwave"`. Triplet samples all three spin components and need not be nonmagnetic. Triangle also accepts `"dxy"`, `"d+id"`, `"d-id"`, `"py"`, `"p+ip"`, `"p-ip"`, `"fwave"`, and `"f_y"`. Seeds select only the starting state. |
| `normal_noise`, `pair_noise` | Initial random perturbation amplitudes. Zero disables the corresponding noise. Random pairing seeds start from zero and use `pair_noise` (default `0.05`) before any spin-sector projection. |
| `solver`, `broy_max` | Choose `"broyden"`, `"energy_descent"`, `"linear_mixing"`, or `"dfsane"`. `broy_max` sets the Broyden history depth (default `10`). |
| `alpha` | Initial mixing factor, default `0.2`, with `0 < alpha <= 1`. Linear mixing uses 80% old order and 20% newly computed order when `alpha = 0.2`. Broyden adds a history correction to this base update. Ordinary energy descent uses `alpha` for its Broyden proposal and tests the direct proposal with backtracking. DF-SANE uses its own step rule. |
| `descent_workers` | Number of trial processes for ordinary `"energy_descent"`, either `1` (default) or `2`. Two workers evaluate the direct and Broyden trial directions concurrently at each step size. Other solvers require `1`. `"pure_SC"` with `"local_barrier"` automatically runs serially even when `2` is requested. |
| `tolerate`, `maxiterate`, `Plotting` | Residual tolerance, iteration limit, and plotting switch. Set `Plotting` to `true` to plot every 100 iterations and at the end. |

The square example includes these solver settings.

```json
"solver": "energy_descent",
"alpha": 0.2,
"broy_max": 10,
"descent_workers": 1
```

For linear mixing, the update is `x_next = x + alpha * (F(x) - x)`, where `F(x)` is the order computed from the current BdG Hamiltonian. Smaller `alpha` gives a more damped update. With `"broyden"`, a history reset sets `alpha` to its input value plus `0.1` when the residual is at most `1e-5`, and otherwise restores the input value. Ordinary `"energy_descent"` starts its direct trial with the full residual and backtracks through step factors `1, 1/2, 1/4, ...` until its energy criterion passes.

`descent_workers` controls trial processes. Each process still uses multithreaded numerical libraries for diagonalization. Broyden, DF-SANE, linear mixing, and serial energy descent use the environment's numerical-library thread settings. With two descent workers, each worker automatically receives a thread limit equal to half the logical CPU count, rounded down with a minimum of one. The current descent search has at most two directions at each step size, so two workers cover its available concurrent trials.

For `"pure_SC"` with `"local_barrier"`, energy descent tries up to ten step sizes per candidate: the initially admissible step `s0`, then `s0/2`, ..., `s0/512` (at most nine halvings). It stops halving as soon as a trial is accepted. If those candidates fail, it switches to five feasible Broyden updates and then returns to energy descent. The same fallback starts after three consecutive accepted step factors below `1e-4` when the residual has decreased by at most 1% over those updates. During the fallback, the objective `F+B` may increase. Every accepted state must still satisfy the filling, local bounds, and physical covariance checks. Broyden proposals take priority, with direct mixing available while the history builds or a proposal fails feasibility checks. The console reports both mode changes, and the saved `step_modes` array records the mode for each accepted update.

For complete inputs, see the [Square full-order example](Square_example.json), [Triangle pairing example](Triangle_example.json), and [Square local-barrier example](Square_local_barrier.json).

To suppress only magnetization while allowing charge separation, use:

```json
"order_ansatz": "nonmagnetic",
"constraint_method": "local_barrier",
"local_max": 1e-6,
"tau": 1e-4
```

For zero local spin within numerical tolerance, use `"constraint_method": "Lagrange_multiplier"`; `local_max` is then unused. The constraint acts on all three components of `m_i = <c_i^dagger sigma c_i>` throughout the iteration, independently of `pair_seed`. See [the complete magnetization-only example](Square_nonmagnetic.json).
