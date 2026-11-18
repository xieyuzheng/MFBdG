"""Real-space BdG for U sum_i n_i_up n_i_dn + sum_{i<j} V_ij n_i n_j.

Negative interactions are attractive. All fields can vary in real space.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import root
from scipy.special import entr, expit

if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CPU.lattice_structure import LatticeDict
from CPU.wick_bdg import density_terms, order_to_Gfull, wick_pfaffian_grad_Gfull, gradE_to_BdG
from CPU import MFSolver as mfsolver
from CPU.gap_oneorbital import check_filling, initial_order
from CPU.solver.solver1d import secant_method


def bond_model(nx, ny, *, t=(1, 0, 0), V=(-2, 0, 0), mu=0,
               lattice="Square", boundary="PBC"):
    """Build hopping and bond interactions. Leave mu=0 to set it in solve()."""
    if min(nx, ny) < 1 or boundary not in ("OBC", "PBC"):
        raise ValueError("Use positive lattice sizes and OBC or PBC.")
    lat = LatticeDict(lattice, nx, ny, boundary)
    h0 = -float(mu) * np.eye(lat.Nsite)
    interaction = np.zeros_like(h0)
    shells = (lat.find_nneighbor, lat.find_nextnneighbor, lat.find_tnneighbor)
    for neighbors, hopping, strength in zip(shells, t, V, strict=True):
        if hopping == 0 and strength == 0:
            continue
        for site, i in lat.site_indices.items():
            for other in neighbors(site, nx, ny):
                if other not in lat.site_indices:
                    continue
                j = lat.site_indices[other]
                if i == j:
                    raise ValueError("Periodic bond wraps onto itself. Increase the lattice size.")
                # Each forward link contributes to both orientations.
                h0[i, j] -= hopping
                h0[j, i] -= hopping
                interaction[i, j] += strength
                interaction[j, i] += strength
    return h0, interaction


def thermal_modes(h, delta, temperature, mu):
    """Diagonalize the BdG matrix in the basis of the supplied h and delta."""
    normal = h - mu * np.eye(len(h))
    bdg = np.block([[normal, delta], [delta.conj().T, -normal.T]])
    try:
        energies, vectors = np.linalg.eigh(bdg)
    except np.linalg.LinAlgError:
        # Retry the same Hermitian matrix with LAPACK's QR eigensolver.
        energies, vectors = eigh(bdg, driver="ev")
    return energies, vectors, expit(-energies / temperature)


def contractions(h, delta, temperature, mu=0.0):
    """Return spectrum, particle density matrix, and <c_b c_a> in particle/hole order."""
    n = len(h)
    energies, vectors, occupation = thermal_modes(h, delta, temperature, mu)
    G = (vectors.conj() * occupation) @ vectors.T  # G_ab = <Psi†_a Psi_b>
    return energies, G[:n, :n], G[n:, :n].T


def _wick_energy(h0, terms, rho, pair):
    weights, operators = terms
    a, b = operators[1], operators[3]
    n = rho.diagonal().real
    kinetic = np.sum(h0 * (rho[::2, ::2] + rho[1::2, 1::2])).real
    # Wick expectation of n_a n_b in the thermal BdG state.
    interaction = weights @ (n[a]*n[b] - abs(rho[a, b])**2 + abs(pair[a, b])**2)
    return (kinetic + interaction) / len(h0)


def Energy_total(h0, terms, rho, pair, energies, T):
    """E, S (k_B=1), F=E-TS per site. Only the physical h0 and interactions enter E."""
    energy = _wick_energy(h0, terms, rho, pair)
    # The full Nambu spectrum counts each physical mode twice.
    f = expit(-np.abs(energies) / T)
    entropy = np.sum(entr(f) - (1 - f)*np.log1p(-f)) / (2 * len(h0))
    return dict(energy=energy, entropy=entropy, free=energy - T*entropy)


def print_energy(state):
    print(f"E/site = {state['energy']:.12g}, S/site = {state['entropy']:.12g}, "
          f"F/site = {state['free']:.12g}", flush=True)
    if "objective" in state:
        print(f"B/site = {state['local_barrier']:.12g}, objective/site = {state['objective']:.12g}", flush=True)
    if "max_dn" in state:
        print(f"peak |n_i-n| = {state['max_dn']:.6g}, peak |onsite pair| = {state['max_onsite_pair']:.6g}", flush=True)
    if "max_m" in state:
        print(f"peak |m| = {state['max_m']:.6g}", flush=True)


def _solve_bdg_mu(h, delta, temperature, target, mu, tol, *, normal_first=False):
    """Use a normal-state or previous-mu seed; accept only the paired filling."""
    n_sites = len(h) // 2
    errors = {}
    def filling_error(trial_mu):
        if trial_mu not in errors:
            _, vectors, occupation = thermal_modes(h, delta, temperature, trial_mu)
            particle_weight = np.sum(np.abs(vectors[:len(h)])**2, axis=0)
            errors[trial_mu] = (particle_weight @ occupation) / n_sites - target
        return errors[trial_mu]

    def normal_seed():
        eps = np.linalg.eigvalsh(h)
        return secant_method(lambda x: 2 * expit((x - eps) / temperature).mean() - target,
                             x1=mu, ftol=tol, print_info=False)

    for use_normal in (normal_first, not normal_first):
        seed = normal_seed() if use_normal else mu
        root = secant_method(filling_error, x1=seed, ftol=tol, print_info=False)
        if np.isfinite(root) and abs(filling_error(root)) <= tol:
            return root
    raise RuntimeError(f"Neither chemical-potential guess reached filling {target} within {tol}.")


def _barrier_target(field, width, tau):
    """Invert h = 2*tau*x/(width^2-|x|^2) for a scalar or vector x."""
    scale = tau/width
    norm = np.linalg.norm(field, axis=-1, keepdims=True)
    return width * field / (scale + np.hypot(scale, norm))


def _local_values(rho, pair, filling):
    up = np.arange(0, len(rho), 2)
    dn = up + 1
    nu, nd = rho[up, up].real, rho[dn, dn].real
    n = nu + nd
    spin, kappa = rho[up, dn], pair[up, dn]
    return np.column_stack((n - filling,
                            2*spin.real, 2*spin.imag, nu - nd, 2*kappa.real, 2*kappa.imag))


def add_local_barrier(h, delta, values, local_max, tau, *, interior=False):
    """Add the charge, spin and complex pair log gradients of the old order."""
    fields = np.zeros_like(values)
    # Gap fields are conjugate to 2 Re kappa, 2 Im kappa.
    for cols, cap in zip((slice(0, 1), slice(1, 4), slice(4, 6)),
                         (local_max, local_max, 2*local_max)):
        x = values[:, cols]
        length = np.linalg.norm(x, axis=1, keepdims=True)
        if interior:
            # The accepted covariance is strictly inside the true log domain.
            fields[:, cols] = 2*tau*x / ((cap-length)*(cap+length))
            continue
        # Same reciprocal-slack guard as the RMFT Hamiltonian update.
        plus = cap + length
        minus = np.maximum(cap - length, min(1e-12, cap*1e-6))
        coeff = tau*(1/minus - 1/plus)
        fields[:, cols] = np.divide(x, length, out=np.zeros_like(x), where=length>0)*coeff
    _add_onsite_fields(h, delta, fields)
    return fields


def _add_onsite_fields(h, delta, fields):
    up = np.arange(0, len(h), 2)
    dn = up + 1
    q, bx, by, bz, dr, di = fields.T
    h[up, up] += q + bz
    h[dn, dn] += q - bz
    h[up, dn] += bx - 1j*by
    h[dn, up] += bx + 1j*by
    delta[up, dn] += dr + 1j*di
    delta[dn, up] -= dr + 1j*di


def _local_barrier_energy(rho, pair, filling, local_max, tau):
    values = _local_values(rho, pair, filling)
    magnitudes = np.column_stack((abs(values[:, 0]), np.linalg.norm(values[:, 1:4], axis=1),
                                  np.linalg.norm(values[:, 4:], axis=1)/2))
    ratios = magnitudes / local_max
    inside = bool(np.all(ratios < 1))
    cost = -tau*np.log1p(-ratios**2).sum(axis=1).mean() if inside else np.inf
    dn, m, k = magnitudes.max(axis=0)
    return dict(local_barrier=cost, bounds_satisfied=inside, max_dn=dn, max_m=m, max_onsite_pair=k)


def _field_response(h, delta, T, mu):
    """Factor chi = -d(onsite correlations)/d(onsite fields) at this thermal state."""
    n = len(h) // 2
    energies, vectors, _ = thermal_modes(h, delta, T, mu)
    # Switch from Nambu to four Majorana modes per site. Each onsite source is
    # then i times a real antisymmetric matrix with six independent entries.
    eye = np.eye(2)
    W = np.block([[eye, eye], [-1j*eye, 1j*eye]]) / np.sqrt(2)
    modes = vectors.reshape(2, n, 2, -1).transpose(1, 0, 2, 3).reshape(n, 4, -1)
    modes = (W @ modes).reshape(4*n, -1)
    modes_dagger = modes.conj().T

    # Normal sources are charge and the three Pauli matrices. The two pairing
    # sources are the real and imaginary parts of the onsite singlet field.
    spin = np.array([eye, [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], np.diag([1, -1])])
    sources = np.zeros((6, 4, 4), complex)
    sources[:4, :2, :2] = spin
    sources[:4, 2:, 2:] = -spin.transpose(0, 2, 1)
    sources[4:, :2, 2:] = np.array([1, 1j])[:, None, None] * (1j*spin[2])
    sources[4:, 2:, :2] = sources[4:, :2, 2:].conj().transpose(0, 2, 1)
    a, b = np.triu_indices(4, 1)
    basis = (W @ sources @ W.conj().T).imag[:, a, b]

    # Integrate the static susceptibility in imaginary time. Particle-hole
    # symmetry gives the second half from the real part of the first half.
    beta = 1 / T
    log_denominator = np.logaddexp(0., -beta*energies)
    stop = beta / 2
    edges = [0., min(stop, 1 / max(abs(energies).max(), T))]
    while edges[-1] < stop:
        edges.append(min(3*edges[-1], stop))
    nodes, weights = np.polynomial.legendre.leggauss(6)
    response = np.zeros((n, n, 6, 6))
    for left, right in zip(edges[:-1], edges[1:]):
        for point, weight in zip(nodes, weights):
            time = left + (right-left)*(1+point)/2
            occupation = np.exp(-time*energies - log_denominator)
            K = ((modes*occupation) @ modes_dagger).reshape(n, 4, n, 4).transpose(1, 3, 0, 2)
            # Connected Wick contraction of two onsite bilinears.
            minors = K[a[:, None], a]*K[b[:, None], b] - K[a[:, None], b]*K[b[:, None], a]
            response += (right-left)*weight*minors.real.transpose(2, 3, 0, 1)

    chi = (basis @ response @ basis.T).transpose(0, 2, 1, 3).reshape(6*n, 6*n)
    chi.flat[::6*n+1] += 1e-9  # Regularize the search direction in thermally frozen modes.
    return cho_factor(chi, check_finite=False)


def _constrained_bdg(h, delta, T, filling, mu, fields, tol, *, refine=False,
                     response_cache=None):
    """Enforce uniform thermal charge, zero spin and zero onsite pairing."""
    up = np.arange(0, len(h), 2)
    dn = up + 1
    if fields is None:
        fields = np.zeros((len(up), 6))

    def evaluate(values):
        values = values.reshape(len(up), -1)
        normal, gap = h.copy(), delta.copy()
        _add_onsite_fields(normal, gap, values)
        energies, rho, pair = contractions(normal, gap, T, mu)
        error = _local_values(rho, pair, filling)
        return error.ravel(), (normal, gap, energies, rho, pair)

    error, state = evaluate(fields)
    # Try the last response first. Rebuild it at the current state if needed.
    # Every proposed field is tested against the actual thermal correlations.
    if response_cache is not None and np.max(abs(error)) > tol/2:
        for sweep in range(2):
            if not response_cache or sweep:
                try:
                    response_cache["factor"] = _field_response(state[0], state[1], T, mu)
                except (np.linalg.LinAlgError, MemoryError):
                    response_cache.clear()
                    break
            for _ in range(9):
                step = cho_solve(response_cache["factor"], error, check_finite=False).reshape(fields.shape)
                if not np.all(np.isfinite(step)):
                    break
                trial = fields + step
                trial_error, trial_state = evaluate(trial)
                if not np.max(abs(trial_error)) < np.max(abs(error)):
                    break
                fields, error, state = trial, trial_error, trial_state
                if np.max(abs(error)) <= tol/2:
                    break
            if np.max(abs(error)) <= tol/2:
                break
    # Fall back to the original root solver from the best verified fields.
    if np.max(np.abs(error)) > tol / 2:
        def residual(x):
            return -evaluate(x)[0]  # Raising a local field lowers its conjugate expectation.

        options = {"fatol": tol/2, "ftol": 0, "fnorm": lambda r: np.max(np.abs(r))}
        if refine:
            options["maxfev"] = 40
        solution = root(residual, fields.ravel(), method="df-sane", options=options)
        if refine and np.max(np.abs(solution.fun)) > tol/2:
            solution = root(residual, solution.x, method="krylov",
                            options={"fatol": tol/2, "maxiter": 40, "jac_options": {"rdiff": 1e-6}})
        fields = solution.x.reshape(len(up), -1).copy()
        error, state = evaluate(fields)
        if not np.max(np.abs(error)) <= tol / 2:
            raise RuntimeError(f"pure_SC onsite constraint residual = {np.max(np.abs(error)):.3g}")

    normal, gap, energies, rho, pair = state
    # Absorb the common charge potential into mu at fixed filling.
    shift = fields[:, 0].mean()
    fields = fields.copy()
    fields[:, 0] -= shift
    mu -= shift
    normal[up, up] -= shift
    normal[dn, dn] -= shift
    return rho, pair, normal, gap, energies, mu, fields


def _spin_barrier_bdg(h, delta, T, filling, mu, fields, m_max, tau, tol):
    """Constrain only spin: m_max=None enforces zero, otherwise a log barrier.

    Local density and onsite pairing are free; one global source fixes filling.
    """
    up = np.arange(0, len(h), 2)
    dn = up + 1
    if fields is None:
        fields = np.zeros((len(up), 3))
    # A uniform potential shift enforces the number equation at fixed filling.
    x = np.r_[fields.ravel(), 0.0]

    def evaluate(x):
        field = x[:3*len(up)].reshape(-1, 3)
        bx, by, bz = field.T
        normal = h.copy()
        normal[up, up] += bz
        normal[dn, dn] -= bz
        normal[up, dn] += bx - 1j*by
        normal[dn, up] += bx + 1j*by
        trial_mu = mu - x[-1]
        energies, rho, pair = contractions(normal, delta, T, trial_mu)
        mag = np.column_stack((2*rho[up, dn].real, 2*rho[up, dn].imag,
                               (rho[up, up]-rho[dn, dn]).real))
        # Analytic inverse of b = 2*tau*m/(m_max^2-|m|^2), including b=0.
        target = np.zeros_like(field) if m_max is None else _barrier_target(field, m_max, tau)
        residual = (target - mag).ravel()
        residual = np.r_[residual, filling - rho.trace().real/len(up)]
        return residual, (rho, pair, normal, energies, trial_mu, field), np.linalg.norm(mag, axis=1)

    atol = min(tol/(2*np.sqrt(3)), 1e-12)
    if m_max is not None:
        atol = min(tol/2, 1e-12, m_max*1e-4)
    def spin_ok(magnitude):
        return magnitude.max() <= tol/2 if m_max is None else magnitude.max() < m_max
    err, state, magnitude = evaluate(x)
    if np.max(abs(err)) > atol or not spin_ok(magnitude):
        def residual(x):
            return evaluate(x)[0]
        sol = root(residual, x, method="df-sane",
                   options={"fatol": atol, "ftol": 0, "maxfev": 40, "fnorm": lambda v: np.max(abs(v))})
        if np.max(abs(sol.fun)) > atol:
            sol = root(residual, sol.x, method="krylov",
                       options={"fatol": atol, "maxiter": 50, "jac_options": {"rdiff": 1e-6}})
        err, state, magnitude = evaluate(sol.x)
    if not (np.max(abs(err)) <= atol and spin_ok(magnitude)):
        raise RuntimeError(f"Spin constraint solve failed: peak |m| = {magnitude.max():.6g}, "
                           f"field residual = {np.max(abs(err)):.3g}")
    barrier = 0.0 if m_max is None else -tau*np.mean(np.log1p(-(magnitude/m_max)**2))
    return (*state, float(barrier), float(magnitude.max()))


def onsite_constraint_method(param):
    """Resolve the numerical method for the onsite constraints of this ansatz."""
    ansatz = param.get("order_ansatz", "full")
    method = param.get("constraint_method")
    if ansatz == "nonmagnetic":
        method = "Lagrange_multiplier" if method is None else method
        if method not in ("Lagrange_multiplier", "local_barrier"):
            raise ValueError("nonmagnetic uses constraint_method Lagrange_multiplier or local_barrier.")
        if "charge_max" in param or param.get("m_max") is not None:
            raise ValueError("Use only local_max to set the nonmagnetic local_barrier bound.")
        return method
    if ansatz == "pure_SC":
        if method is None:
            return "Lagrange_multiplier"
        if method == "local_barrier" and ("charge_max" in param or param.get("m_max") is not None):
            raise ValueError("Use only local_max to set the local_barrier bounds.")
        if method in ("Lagrange_multiplier", "local_barrier"):
            return method
        raise ValueError("pure_SC uses constraint_method Lagrange_multiplier or local_barrier.")
    if ansatz == "full":
        return None
    raise ValueError("order_ansatz must be full, pure_SC, or nonmagnetic.")


def solve(h0, V, *, n_filling, mu=0.0, U=0.0, initial_m=0.0, normal_noise=0.0,
          pair_seed=None, pair_noise=None, temperature=0.05, mixing=None, tol=1e-8,
          max_iter=500, solver="linear_mixing", broy_max=10, order_ansatz="full",
          mu_guess_threshold=0.01, plot_iteration=None, m_max=None, tau=1e-4,
          local_max=1e-6, constraint_method=None, descent_workers=1):
    """Solve at fixed filling (electrons/site).

    mu is measured relative to h0 and serves as the initial guess at fixed filling.
    Energy descent defaults to serial trials. Set descent_workers=2 to evaluate
    ordinary descent trials in isolated processes, each using half the logical
    CPUs (at least one). pure_SC local_barrier automatically uses one worker,
    including when descent_workers=2 is requested.
    Each interacting orbital pair contributes its Hartree, exchange and pairing fields.
    A site interaction U couples its two spins. A bond V couples all four spin pairs.
    pure_SC keeps all interacting bond orders and targets uniform spin-balanced
    charge and zero onsite pairing. constraint_method="Lagrange_multiplier"
    (default) enforces equalities and "local_barrier" enforces finite bounds.
    nonmagnetic constrains only all three components of local spin, leaving
    local charge and onsite pairing free. Lagrange_multiplier (default) fixes
    spin to zero; local_barrier uses local_max (default 1e-6) and tau.
    With full order, optional m_max bounds the length of every thermal onsite spin
    using a rotationally invariant log barrier of strength tau (energy units).
    local_barrier shares one tau across charge deviation, spin length and onsite pair magnitude.
    Energy descent mixes full covariances and scores E-TS plus the barrier on each trial.
    local_max bounds all three local quantities for pure_SC, and only spin
    length for nonmagnetic. It is used only with local_barrier.
    initial_m is n_up - n_down (a scalar or one value per site).
    normal_noise sets the Gaussian scale for active occupations and for the real
    and imaginary parts of active exchange correlators, only at initialization.
    A symmetric site pair_seed initializes the correlator in the singlet sector.
    "mixed", "singlet", and "triplet" sample all allowed spatial pairing channels
    in their respective spin sectors, only at initialization. Random triplet
    seeds include all three spin components and need not be nonmagnetic.
    pair_noise is the standard deviation of each real and imaginary perturbation
    before sector projection (default 0.05 for random seeds, zero otherwise).
    pair_order is the trial anomalous correlator used to build the mean fields.
    pair_bdg[a,b] = <c_b c_a> is the full thermal correlator of the final BdG
    Hamiltonian, including site pairs outside the interacting channels.
    Both and the gap matrix delta use indices a=2*i+s (up=0, down=1).
    Start mu from the normal block on the first evaluation and when the last
    order residual is >= mu_guess_threshold. A smaller residual reuses mu.
    """
    terms = density_terms(V, U)
    order_guess = initial_order(len(h0), terms, n_filling=n_filling, initial_m=initial_m,
                                normal_noise=normal_noise, pair_seed=pair_seed, pair_noise=pair_noise)
    param = dict(n_filling=n_filling, T=temperature, solver=solver, tolerate=tol,
                 maxiterate=max_iter, broy_max=broy_max, order_ansatz=order_ansatz,
                 mu_guess_threshold=mu_guess_threshold)
    if descent_workers is not None:
        param["descent_workers"] = descent_workers
    if m_max is not None:
        param.update(m_max=m_max, tau=tau)
    if constraint_method is not None:
        param["constraint_method"] = constraint_method
    if order_ansatz in ("pure_SC", "nonmagnetic") and constraint_method == "local_barrier":
        param.update(local_max=local_max, tau=tau)
    if mixing is not None:
        param["alpha"] = mixing
    return solve_from_order(order_guess, h0, terms, param, mu=mu, Plot_BdG=plot_iteration)


def solve_from_order(order_guess, h0, terms, param, *, mu=0.0, Plot_BdG=None, _trial=None):
    """Iterate from the supplied (rho, pair) using the real-space Wick Hamiltonian."""
    solvername = param["solver"]
    workers = param.get("descent_workers", 1)
    if isinstance(workers, (bool, np.bool_)) or not isinstance(workers, (int, np.integer)) or workers not in (1, 2):
        raise ValueError("descent_workers must be 1 or 2.")
    if workers == 2 and solvername != "energy_descent":
        raise ValueError("descent_workers=2 requires energy_descent.")
    if solvername == "broyden":
        Self_Consistent = mfsolver.Self_Consistent_broyden
    elif solvername == "energy_descent":
        Self_Consistent = mfsolver.Self_Consistent_energy_descent
    elif solvername == "dfsane":
        Self_Consistent = mfsolver.Self_Consistent_dfsane
    elif solvername in ("safe", "linear_mixing"):
        Self_Consistent = mfsolver.Self_Consistent_linear_mixing
    else:
        raise ValueError(f"Unknown solver {solvername!r}")

    n_filling, temperature, tol = param["n_filling"], param["T"], param["tolerate"]
    mu_guess_threshold = param.get("mu_guess_threshold", 0.01)
    check_filling(n_filling)
    if not mu_guess_threshold >= 0:
        raise ValueError("mu_guess_threshold must be nonnegative.")
    if not (temperature > 0 and tol > 0 and param["maxiterate"] >= 1 and 0 < param.get("alpha", 0.2) <= 1):
        raise ValueError("Use T, tolerate, maxiterate > 0 and 0 < alpha <= 1.")
    # Each string is (a†, a, b†, b) with a < b. Its independent contractions
    # are the two real occupations, complex rho[a,b], and complex pair[a,b].
    _, operators = terms
    a, b = operators[1], operators[3]
    orbitals = np.unique(np.r_[a, b])
    rho_seed, pair_seed = order_guess
    rho_seed = np.asarray(rho_seed, dtype=complex)

    order_ansatz = param.get("order_ansatz", "full")
    method = onsite_constraint_method(param)
    exact_onsite = order_ansatz == "pure_SC" and method == "Lagrange_multiplier"
    bond = a // 2 != b // 2
    bond_a, bond_b = a[bond], b[bond]
    m_max, tau = param.get("m_max"), param.get("tau", 1e-4)
    if order_ansatz == "nonmagnetic" and method == "local_barrier":
        m_max = param.get("local_max", 1e-6)
        if m_max is None:
            raise ValueError("local_max must be positive and finite for nonmagnetic local_barrier.")
    if m_max is not None and not (np.isfinite(m_max) and m_max > 0 and np.isfinite(tau) and tau > 0):
        bound_name = "local_max" if order_ansatz == "nonmagnetic" else "m_max"
        raise ValueError(f"The magnetization barrier requires positive finite {bound_name} and tau.")
    local_max = None
    if order_ansatz == "pure_SC" and method == "local_barrier":
        local_max = param.get("local_max", 1e-6)
        if not (np.isfinite(local_max) and local_max > 0 and np.isfinite(tau) and tau > 0):
            raise ValueError("local_max and tau must be positive and finite.")
        # All onsite normal and anomalous spin pairs enter the barrier even when U=0.
        up = np.arange(0, len(rho_seed), 2)
        a, b = np.r_[bond_a, up], np.r_[bond_b, up+1]
        orbitals = np.arange(len(rho_seed))
    magnetic = order_ansatz == "nonmagnetic" or (m_max is not None and order_ansatz == "full")
    n_local = len(orbitals)
    if exact_onsite:
        a, b = bond_a, bond_b
        n_local = 0

    def pack_order(rho, pair):
        if order_ansatz == "pure_SC":
            values = np.r_[rho[bond_a, bond_b], pair[bond_a, bond_b]]
            return np.r_[values.real, values.imag]
        local = rho.diagonal().real[orbitals]
        values = np.r_[rho[a, b], pair[a, b]]
        return np.r_[local, values.real, values.imag]

    if local_max is not None and solvername == "energy_descent":
        from CPU.barrier_descent import mixed_barrier_descent
        return mixed_barrier_descent(order_guess, h0, terms, param, mu, local_max, pack_order, Plot_BdG)

    def pack(rho, pair):
        if local_max is None:
            return pack_order(rho, pair)
        # Explicit fixed-point mixing also iterates the auxiliary onsite correlations.
        values = np.r_[rho[a, b], pair[a, b]]
        return np.r_[rho.diagonal().real[orbitals], values.real, values.imag]

    # Packed residual order: occupations, Re(chi, pair), Im(chi, pair).
    # Onsite spin coherence and onsite pairing belong to the onsite group.
    is_bond = a // 2 != b // 2
    groups = np.r_[np.zeros(n_local, dtype=int), np.tile(np.r_[is_bond.astype(int), 2*is_bond], 2)]
    residual_indices = [np.flatnonzero(groups == group) for group in range(3)]

    # Resolve energy differences per site when comparing descent trials.
    descent = solvername == "energy_descent"
    field_tol = min(tol, 2e-12) if descent else tol
    if exact_onsite and m_max is not None:
        field_tol = min(field_tol, m_max/2)  # pure_SC already fixes all three spin components.
    if local_max is not None:
        field_tol = min(field_tol, 2e-12, 1e-4*local_max)
    response_cache = {} if exact_onsite else None
    err_prev = np.inf
    onsite_fields = None
    spin_fields = None

    def BdG_nextG(Gfull_old):
        """Build the fields at Gfull_old and return the new thermal covariance blocks."""
        nonlocal mu, onsite_fields, spin_fields
        _, gradient = wick_pfaffian_grad_Gfull(Gfull_old, terms)
        h, delta = gradE_to_BdG(h0, gradient)
        spin_info = {}
        if local_max is not None:
            norb = len(h)
            old = _local_values(Gfull_old[:norb, :norb], Gfull_old[norb:, :norb].T, n_filling)
            fields = add_local_barrier(h, delta, old, local_max, tau)
            spin_info["old_barrier_fields"] = fields
        if (not exact_onsite and not magnetic) or (onsite_fields is None and spin_fields is None):
            mu = _solve_bdg_mu(h, delta, temperature, n_filling, mu, field_tol,
                               normal_first=err_prev >= mu_guess_threshold)
        if exact_onsite:
            rho, F, h, delta, energies, mu, onsite_fields = _constrained_bdg(
                h, delta, temperature, n_filling, mu, onsite_fields, field_tol,
                refine=descent, response_cache=response_cache)
            if exact_onsite and m_max is not None:
                nup, ndn = rho.diagonal().real.reshape(-1, 2).T
                spin_info["max_m"] = np.hypot(nup-ndn, 2*abs(rho[::2, 1::2].diagonal())).max()
        elif magnetic:
            rho, F, h, energies, mu, spin_fields, barrier, peak = _spin_barrier_bdg(
                h, delta, temperature, n_filling, mu, spin_fields, m_max, tau, field_tol)
            spin_info = dict(spin_fields=spin_fields, spin_barrier_field=spin_fields,
                             spin_barrier=barrier, max_m=peak, bounds_satisfied=True)
        else:
            energies, rho, F = contractions(h, delta, temperature, mu)
        F = (F - F.T) / 2
        if local_max is not None:
            spin_info.update(_local_barrier_energy(rho, F, n_filling, local_max, tau))
        return rho, F, h, delta, energies, spin_info

    def FGap(order_old, previous=None, previous_err=None):
        nonlocal mu, onsite_fields, spin_fields, err_prev
        if previous is not None:
            spin_fields = previous.get("spin_barrier_field")
            mu, onsite_fields, err_prev = previous["mu"], previous.get("onsite_fields"), previous_err
        real, imag = np.split(order_old[n_local:], 2)
        chi, pairing = np.split(real + 1j * imag, 2)
        if exact_onsite:
            rho_order = np.zeros_like(rho_seed)
            np.fill_diagonal(rho_order, n_filling / 2)
        else:
            rho_order = rho_seed.copy()
            rho_order[orbitals, orbitals] = order_old[:n_local]
        rho_order[a, b] = chi
        rho_order[b, a] = chi.conj()
        pair = np.zeros_like(rho_seed)
        pair[a, b] = pairing
        pair[b, a] = -pairing
        Gfull_old = order_to_Gfull(rho_order, pair)
        rho, new_pair, h, delta, energies, spin_info = BdG_nextG(Gfull_old)
        density = rho.diagonal().real.reshape(-1, 2).sum(axis=1)
        filling = density.mean()
        if abs(filling - n_filling) > tol:
            raise RuntimeError(f"Number equation missed target: filling = {filling:.12g}")
        order = pack_order(rho_order, pair) if local_max is not None else order_old.copy()
        result = dict(order=order, rho=rho, pair_order=pair, delta=delta, density=density,
                      filling=filling, mu=mu, h=h - mu * np.eye(len(h)), energies=energies,
                      pair_bdg=new_pair, **spin_info)
        if exact_onsite:
            result["onsite_fields"] = onsite_fields
        order_new = pack(rho, new_pair)
        Fm = order_new - order_old
        residual = np.abs(Fm)
        result["errset"] = np.array([residual[ix].max(initial=0.0) for ix in residual_indices])
        err_prev = result["errset"].max()
        if descent:
            update_energy(result)
        return Fm, result

    def update_energy(result):
        result.update(Energy_total(h0, terms, result["rho"], result["pair_bdg"],
                                   result["energies"], temperature))

    if _trial is not None:
        return FGap(*_trial)

    history = []
    def report_iteration(iterate, result):
        if "free" not in result:
            update_energy(result)
        history.append((iterate, result["mu"], result["energy"], result["entropy"], result["free"], result["errset"]))
        print_energy(result)
        if Plot_BdG:
            Plot_BdG(iterate, result)

    if workers == 2:
        from CPU.trial_workers import TrialWorkers
        with TrialWorkers(order_guess, h0, terms, param) as trials:
            result, iterate, err = Self_Consistent(
                pack(rho_seed, pair_seed), param, FGap, report_iteration, trial_batch=trials)
    else:
        result, iterate, err = Self_Consistent(pack(rho_seed, pair_seed), param, FGap, report_iteration)
    if local_max is not None:
        result["converged"] = bool(err < tol and result["bounds_satisfied"])
        if not result["bounds_satisfied"]:
            result["stop_reason"] = "Thermal state outside the explicit local barrier"
            print(result["stop_reason"], flush=True)
    if iterate == param["maxiterate"] and err >= tol:
        print("maximum iteration reached, error =", err)
    if iterate == 0 or iterate % 100:
        report_iteration(iterate, result)
    history_dtype = [("iterate", int), ("mu", float), ("energy", float), ("entropy", float), ("free", float), ("errset", float, (3,))]
    return dict(result, iterate=iterate, err=err, history=np.array(history, dtype=history_dtype))


if __name__ == "__main__":
    # Nearest-neighbor bond pairing on a periodic square.
    nx = ny = 6
    h0, V = bond_model(nx, ny, V=(-2.0, 0.0, 0.0))
    result = solve(h0, V, n_filling=0.9, solver="broyden", max_iter=1500)
    print(f"Filling {result['filling']:.9f}, mu {result['mu']:.8f}")
    print(f"Iterations {result['iterate']}, residual {result['err']:.2e}")
    i = (nx - 1) // 2 + nx * ((ny - 1) // 2)
    delta = result["delta"]
    print("Central x bond gap (spin rows and columns):")
    print(delta[2*i:2*i+2, 2*(i+1):2*(i+1)+2])
