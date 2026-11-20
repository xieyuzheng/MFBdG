"""Read previously saved real-space order blocks for plotting.

local_block has columns (n_up, n_down, rho_up_down, pair_up_down).
normal_bond_block and pairing_block have shape (number of bonds, 2, 2),
with spin rows at bond_sites[:, 0] and spin columns at bond_sites[:, 1].
Only interacting site pairs are stored, once with i < j.
Normal entries use the solver's rho. Pair entries use its trial pair_order.
The latter defines the mean fields and may differ from the thermal pairing
when the solve is unfinished. Onsite entries are retained for every ansatz
so local constraints can be inspected.
"""

import numpy as np


def plotting_orders(state):
    """Return density, local normal matrix and pairing for new or legacy files.

    Reconstructed matrices contain the saved local and interacting-bond entries.
    Other entries are zero placeholders for the plotting routines.
    """
    if "local_block" not in state:
        return state["density"], state.get("rho"), state["pair_order"]
    local = state["local_block"]
    n = len(local)
    rho = np.zeros((2*n, 2*n), dtype=complex)
    pair = np.zeros_like(rho)
    up = 2 * np.arange(n)
    rho[up, up], rho[up+1, up+1] = local[:, 0], local[:, 1]
    rho[up, up+1], rho[up+1, up] = local[:, 2], local[:, 2].conj()
    pair[up, up+1], pair[up+1, up] = local[:, 3], -local[:, 3]
    i, j = state["bond_sites"].T
    normal, anomalous = state["normal_bond_block"], state["pairing_block"]
    spin_rho = rho.reshape(n, 2, n, 2)
    spin_pair = pair.reshape(n, 2, n, 2)
    spin_rho[i, :, j, :] = normal
    spin_rho[j, :, i, :] = normal.conj().transpose(0, 2, 1)
    spin_pair[i, :, j, :] = anomalous
    spin_pair[j, :, i, :] = -anomalous.transpose(0, 2, 1)
    return (local[:, 0] + local[:, 1]).real, rho, pair
