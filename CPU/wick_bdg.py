"""Unprojected density interactions in the collinear RMFT Nambu convention."""

import numpy as np


def density_terms(V, U):
    """Return couplings and strings (a†, a, b†, b) for the onsite and bond terms."""
    n = len(V)
    i, j = np.triu_indices(n, k=1)
    bonds = V[i, j] != 0
    i, j = i[bonds], j[bonds]
    # Spin indices alternate up, down. Count each interacting bond once.
    a = [2*i + s for s in (0, 1) for t in (0, 1)]
    b = [2*j + t for s in (0, 1) for t in (0, 1)]
    weights = [V[i, j]] * 4
    if U != 0:
        a.append(2*np.arange(n))
        b.append(2*np.arange(n) + 1)
        weights.append(np.full(n, U))
    a, b = np.concatenate(a), np.concatenate(b)
    return np.concatenate(weights), np.stack([a + 2*n, a, b + 2*n, b])


def order_to_Gfull(rho, pair):
    """Build G from spin-orbital rho_ab = <c†_a c_b> and P_ab = <c_b c_a>."""
    # The lower-left covariance is <c_a c_b>.
    anomalous = -pair
    return np.block([[rho, anomalous.conj().T],
                     [anomalous, np.eye(len(rho)) - rho.T]])


def wick_pfaffian_grad_Gfull(G, terms):
    """Sum the weighted four-operator Pfaffians and their derivatives with respect to G."""
    weights, operators = terms
    modes = len(G) // 2
    dagger = (operators + modes) % (2*modes)
    gradient = np.zeros_like(G)
    energy = 0.0
    # Pf(W) = W01 W23 - W02 W13 + W03 W12, Wpq = <Op Oq>.
    for p, q, r, s, sign in [(0, 1, 2, 3, 1), (0, 2, 1, 3, -1), (0, 3, 1, 2, 1)]:
        left = dagger[p], operators[q]
        right = dagger[r], operators[s]
        coefficient = sign * weights
        energy += np.sum(coefficient * G[left] * G[right]).real
        np.add.at(gradient, left, coefficient * G[right])
        np.add.at(gradient, right, coefficient * G[left])
    return energy, gradient


def gradE_to_BdG(h0, gradient):
    """Return Hermitian spin-orbital hopping and the antisymmetric gap."""
    m = 2 * len(h0)
    normal = gradient[:m, :m] - gradient[m:, m:].T
    normal = (normal + normal.conj().T) / 2
    create, destroy = gradient[:m, m:], gradient[m:, :m]
    pairing = (create - create.T + (destroy - destroy.T).conj().T) / 2
    return np.kron(h0, np.eye(2)) + normal, pairing
