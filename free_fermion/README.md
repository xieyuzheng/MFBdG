# Free fermions

Run from the MFBdG project root using an existing JSON input.

```sh
python free_fermion/Plot.py Square_example
python free_fermion/Plot.py Triangle_example
```

For interactive plotting, open `free_fermion/Plot.ipynb` with the **Python (MFBdG)** kernel and run both code cells. Choose Square or Triangle, then adjust sliders for `t1`, `t2`, `t3`, `n_filling` and `Nk`. Plots update when a slider is released; the notebook does not automatically save PNGs. It requires `ipywidgets`.

`bands.py` holds the shared calculation and plotting functions used by both the notebook and `Plot.py`.

The script plots the band along a symmetry path, a fine two-dimensional band map with the Fermi contour in black, and the density of states (DOS). It uses explicit dispersions with the same three neighbor shells, lattice vectors, and negative hopping sign as MFBdG. Square primitive vectors are (1, 0) and (0, 1). Triangle primitive vectors are (1, 0) and (1/2, sqrt(3)/2).

| Input | Use |
| --- | --- |
| `lattice` | `Square` or `Triangle` |
| `t1`, `t2`, `t3` | First-, second-, and third-neighbor hopping |
| `n_filling` | Electrons per site including both spins, with `0 < n_filling < 2` |
| `Nk` | Momentum resolution, default `1200`. The Fermi energy uses `Nk x Nk` points, the contour uses `(Nk+1) x (Nk+1)`, and each band-path segment uses `Nk` samples. |
| `Plotting` | Show the figure when true. The PNG is always saved. |

Both the notebook and script use **T = 0**. The Fermi energy is obtained by filling the lowest k-grid energies, with two spin states per k point. When the occupied levels are completely filled, mu is the midpoint between the last occupied and first empty energies; a partially filled level has mu at that energy. Degenerate levels may be partially occupied. Increasing `Nk` improves the finite-grid approximation to the bulk Fermi surface. The `T` value in an existing mean-field JSON input is ignored.

The calculation uses the bulk hopping band. `Nx`, `Ny`, interaction strengths, boundary-condition settings, and mean-field order choices in the JSON have no role here. The square path is Gamma-X-M-Gamma and the triangular path is Gamma-M-K-Gamma.

Figures are saved in `Output/<input_name>/free_fermion/`. Filenames include lattice, filling, hoppings, `T=0`, and `Nk`.

The DOS uses a 400-bin histogram of the same reciprocal-cell grid used to find the Fermi energy. The energy axes show bare band energy epsilon, with dashed lines marking mu. The Fermi-surface color map also shows bare energy, and its black contour marks epsilon equal to mu. It includes both spins and integrates to two states per site. Histogram bins set the energy resolution and give finite heights to singular peaks.
