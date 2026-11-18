"""Square-lattice pairing seeds and symmetry channels."""

import numpy as np

from CPU.lattice_structure import SQUARE_BONDS


PAIR_SEEDS = ("dwave", "pwave", "extended_s")


def pair_seed(name, param):
    """Square-lattice initial pair correlator, with amplitude 0.1."""
    nx, ny = param["Nx"], param["Ny"]
    boundary = param.get("Boundary_Condition", "PBC")
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


def pairing_channels(shell, direction, coupling):
    """Resolve the even singlet and odd triplet fields on the two forward bonds."""
    first, second = SQUARE_BONDS[shell-1]
    s1, p1 = direction(*first)
    s2, p2 = direction(*second)
    d_name = "d_xy" if shell == 2 else "d_x2-y2"
    # On the diagonal shell, first and second point NE and SE.
    px, py = ((p1 + p2) / 2, (p1 - p2) / 2) if shell == 2 else (p1, p2)
    return {
        f"{shell}NN extended s": coupling * (s1 + s2) / 2,
        f"{shell}NN {d_name}": coupling * (s1 - s2) / 2,
        f"{shell}NN p_x": coupling * px,
        f"{shell}NN p_y": coupling * py,
    }
