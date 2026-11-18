"""Triangular-lattice bond harmonics in Cartesian axes, with a1 along x."""

import numpy as np

from CPU.lattice_structure import TRIANGLE_BASIS, TRIANGLE_BONDS


PAIR_SEEDS = ("dwave", "pwave", "extended_s", "dxy", "d+id", "d-id",
              "py", "p+ip", "p-ip", "fwave", "f_y")


def triangle_forms(shell):
    """Three even and three odd real harmonics on the forward bonds of a shell."""
    xy = TRIANGLE_BONDS[shell-1] @ TRIANGLE_BASIS
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    forms = {"extended s": np.ones(3), "d_x2-y2": np.cos(2*theta), "d_xy": np.sin(2*theta),
             "p_x": np.cos(theta), "p_y": np.sin(theta)}
    # The three forward bonds have cos(3 theta)=1 on shells 1,3 and sin(3 theta)=1 on 2.
    forms["f_y" if shell == 2 else "f_x"] = np.ones(3)
    for weights in forms.values():
        weights[np.abs(weights) < 1e-14] = 0
    return forms


def pair_seed(name, param):
    """Amplitude 0.1 on active shells, with opposite-spin singlet or t0 triplet structure."""
    nx, ny = param["Nx"], param["Ny"]
    y, x = np.indices((ny, nx))
    i = y*nx + x
    pair = np.zeros((nx*ny, nx*ny), complex)
    count = np.zeros(pair.shape, int)
    odd = name in ("pwave", "py", "p+ip", "p-ip", "fwave", "f_y")
    for shell, offsets in enumerate(TRIANGLE_BONDS, 1):
        if not param.get(f"V{shell}", 0):
            continue
        f = triangle_forms(shell)
        weights = {"extended_s": f["extended s"], "dwave": f["d_x2-y2"], "dxy": f["d_xy"],
                   "d+id": f["d_x2-y2"] + 1j*f["d_xy"], "d-id": f["d_x2-y2"] - 1j*f["d_xy"],
                   "pwave": f["p_x"], "py": f["p_y"], "p+ip": f["p_x"] + 1j*f["p_y"],
                   "p-ip": f["p_x"] - 1j*f["p_y"], "fwave": f.get("f_x", np.zeros(3)),
                   "f_y": f.get("f_y", np.zeros(3))}[name]
        for (dx, dy), value in zip(offsets, weights):
            valid = np.ones(i.shape, bool)
            if param.get("Boundary_Condition", "PBC") == "OBC":
                valid = (x+dx >= 0) & (x+dx < nx) & (y+dy >= 0) & (y+dy < ny)
            a = i[valid]
            b = ((y[valid]+dy) % ny)*nx + (x[valid]+dx) % nx
            pair[a, b] += value
            pair[b, a] += -value if odd else value
            count[a, b] += 1
            count[b, a] += 1
    pair *= 0.1 / np.maximum(count, 1)
    return np.kron(pair, [[0, 1], [1 if odd else -1, 0]])


def pairing_channels(shell, direction, coupling):
    """Resolve the three even and three odd amplitudes using their orthogonal bond vectors."""
    singlet, triplet = zip(*(direction(dx, dy) for dx, dy in TRIANGLE_BONDS[shell-1]))
    channels = {}
    for name, weights in triangle_forms(shell).items():
        fields = triplet if name.startswith(("p_", "f_")) else singlet
        channels[f"{shell}NN {name}"] = coupling * np.tensordot(weights, fields, axes=1) / (weights @ weights)
    return channels


def bond_fields(channels, shells):
    """Reconstruct each plotted s, d, p or f contribution on each oriented bond."""
    fields = {k: v for k, v in channels.items() if k == "onsite s"}
    for shell in shells:
        forms = triangle_forms(shell)
        for name, weights in forms.items():
            group = "s" if name == "extended s" else name[0]
            coefficient = channels[f"{shell}NN {name}"]
            for j, weight in enumerate(weights):
                key = f"{shell}NN {group} {j}"
                fields[key] = fields.get(key, 0) + weight * coefficient
    return fields
