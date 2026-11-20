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
