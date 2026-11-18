"""Initial real-space BdG order parameters (normal and pairing correlations)."""

import numpy as np


SQUARE_PAIR_SEEDS = ("dwave", "pwave", "extended_s")


def square_pair_seed(name, nx, ny, boundary):
    """Square-lattice initial pair correlator, with amplitude 0.1."""
    site = np.arange(nx * ny)
    x, y = site % nx, site // nx
    dx, dy = x - x[:, None], y - y[:, None]
    if name == "dwave":
        # Keep the axial sign pattern on all active x and y bonds.
        return 0.1 * ((dy == 0).astype(float) - (dx == 0))
    if boundary == "PBC":
        dx %= nx
        dy %= ny
    right = (dx == 1) & (dy == 0)
    if name == "pwave":
        px = right.astype(float) - right.T
        return 0.1 * np.kron(px, [[0, 1], [1, 0]])  # S_z = 0 triplet
    up = (dx == 0) & (dy == 1)
    return 0.1 * (right | right.T | up | up.T)  # extended_s singlet


def random_order_generate(param, terms):
    """Translate the input presets into the initial (rho, pair) for this lattice."""
    nx, ny = param["Nx"], param["Ny"]
    pair_seed = param.get("pair_seed")
    if isinstance(pair_seed, str) and pair_seed in SQUARE_PAIR_SEEDS:
        if param["lattice"] != "Square":
            raise ValueError(f"The {pair_seed} seed uses square-lattice x and y bonds.")
        pair_seed = square_pair_seed(pair_seed, nx, ny, param.get("Boundary_Condition", "PBC"))
    return initial_order(nx * ny, terms, n_filling=param["n_filling"],
                         initial_m=initial_magnetization(param), normal_noise=param.get("normal_noise", 0.0),
                         pair_seed=pair_seed, pair_noise=param.get("pair_noise"))


def initial_magnetization(param):
    """Return m_i = n_up - n_down for the first iteration."""
    seed = param.get("normal_seed", "uniform").lower()
    if seed == "uniform":
        return 0.0
    n = param["n_filling"] if param["n_filling"] is not None else 1.0
    m = min(0.1, n, 2 - n)
    if seed == "fm":
        return m
    if seed == "afm" and param["lattice"] == "Square":
        y, x = np.indices((param["Ny"], param["Nx"]))
        return (m * (-1)**(x + y)).ravel()
    raise ValueError("Use normal_seed = uniform, fm, or afm (square checkerboard).")


def default_pair_seed(terms):
    _, operators = terms
    return "mixed" if np.any(operators[1] // 2 != operators[3] // 2) else 0.1


def initial_order(n_sites, terms, *, n_filling=None, initial_m=0.0, normal_noise=0.0,
                  pair_seed=None, pair_noise=None):
    """Build the spin-orbital density and pair matrices on the interacting sites and bonds."""
    _, operators = terms
    a, b = operators[1], operators[3]
    orbitals = np.unique(np.r_[a, b])
    mask = np.zeros((2*n_sites, 2*n_sites), dtype=bool)
    mask[a, b] = mask[b, a] = True
    if pair_seed is None:
        pair_seed = default_pair_seed(terms)
    pair = initial_pair(mask, pair_seed, pair_noise)
    filling = 1.0 if n_filling is None else n_filling
    rho = initial_normal(n_sites, filling, initial_m, orbitals, a, b, normal_noise)
    return rho, pair


def initial_pair(mask, seed, pair_noise=None):
    """Initialize P, adding Gaussian noise once per independent spin pair."""
    mixed = isinstance(seed, str) and seed == "mixed"
    if pair_noise is None:
        pair_noise = 0.05 if mixed else 0.0
    if not np.isfinite(pair_noise) or pair_noise < 0:
        raise ValueError("pair_noise must be finite and nonnegative.")
    if mixed:
        seed = np.zeros_like(mask, dtype=complex)
    seed = np.asarray(seed, dtype=complex)
    if seed.shape != mask.shape:
        # A scalar or symmetric site seed initializes the singlet sector.
        n_sites = len(mask) // 2
        if seed.ndim and seed.shape != (n_sites, n_sites):
            raise ValueError("Use a scalar, an N x N singlet seed, or a 2N x 2N spin-orbital seed.")
        seed = np.kron(np.broadcast_to(seed, (n_sites, n_sites)), [[0, 1], [-1, 0]])
    pair = np.where(mask, seed, 0j)
    if not np.allclose(pair, -pair.T, rtol=0, atol=1e-14):
        raise ValueError("Pair seed has incorrect symmetry on the selected bonds.")
    if pair_noise:
        a, b = np.nonzero(np.triu(mask, k=1))
        rng = np.random.default_rng(0)
        noise = pair_noise * (rng.normal(size=len(a)) + 1j*rng.normal(size=len(a)))
        pair[a, b] += noise
        pair[b, a] -= noise
    return pair


def initial_normal(n_sites, filling, m, orbitals, a, b, normal_noise):
    """Seed onsite spin densities and perturb independent active normal correlators."""
    m = np.asarray(m, dtype=float)
    rho = np.zeros((2*n_sites, 2*n_sites), dtype=complex)
    up = 2 * np.arange(n_sites)
    rho[up, up] = (filling + m) / 2
    rho[up+1, up+1] = (filling - m) / 2
    if not np.isfinite(normal_noise) or normal_noise < 0:
        raise ValueError("normal_noise must be finite and nonnegative.")
    if normal_noise and orbitals.size:
        rng = np.random.default_rng(1)
        dn = rng.normal(scale=normal_noise, size=orbitals.size)
        rho[orbitals, orbitals] += dn - dn.mean()  # Preserve total filling.
        rho[a, b] = normal_noise * (rng.normal(size=a.size) + 1j*rng.normal(size=a.size))
        rho[b, a] = rho[a, b].conj()
    return rho
