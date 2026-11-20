"""Onsite constraints, barrier fields, and the single-Gamma susceptibility."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import root
from scipy.special import expit

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


def print_energy(state):
    print(f"E/site = {state['energy']:.12g}, S/site = {state['entropy']:.12g}, "
          f"F/site = {state['free']:.12g}", flush=True)
    if "objective" in state:
        print(f"B/site = {state.get('local_barrier', state.get('spin_barrier', 0.)):.12g}, objective/site = {state['objective']:.12g}", flush=True)
    if "max_dn" in state:
        print(f"peak |n_i-n| = {state['max_dn']:.6g}, peak |onsite pair| = {state['max_onsite_pair']:.6g}", flush=True)
    if "max_m" in state:
        print(f"peak |m| = {state['max_m']:.6g}", flush=True)


def _barrier_target(field, width, tau):
    """Invert h = 2*tau*x/(width^2-|x|^2) for a scalar or vector x."""
    scale = tau/width
    norm = np.linalg.norm(field, axis=-1, keepdims=True)
    return width * field / (scale + np.hypot(scale, norm))


def _local_values(rho, pair, filling):
    """Normal onsite coordinates: charge deviation and the three spin components."""
    up = np.arange(0, len(rho), 2)
    dn = up + 1
    nu, nd = rho[up, up].real, rho[dn, dn].real
    n = nu + nd
    spin = rho[up, dn]
    return np.column_stack((n - filling,
                            2*spin.real, 2*spin.imag, nu - nd))


def add_local_barrier(h, delta, values, local_max, tau, *, interior=False):
    """Add charge and spin log-barrier gradients to the local normal block."""
    fields = np.zeros_like(values)
    for cols in (slice(0, 1), slice(1, 4)):
        cap = local_max
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
    q, bx, by, bz = fields.T
    h[up, up] += q + bz
    h[dn, dn] += q - bz
    h[up, dn] += bx - 1j*by
    h[dn, up] += bx + 1j*by


def _local_barrier_energy(rho, pair, filling, local_max, tau):
    values = _local_values(rho, pair, filling)
    magnitudes = np.column_stack((abs(values[:, 0]), np.linalg.norm(values[:, 1:4], axis=1)))
    ratios = magnitudes / local_max
    inside = bool(np.all(ratios < 1))
    cost = -tau*np.log1p(-ratios**2).sum(axis=1).mean() if inside else np.inf
    dn, m = magnitudes.max(axis=0)
    up = np.arange(0, len(rho), 2)
    k = np.max(abs(pair[up, up+1]), initial=0.)  # Observable, not a constrained quantity.
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
    del vectors
    modes_dagger = modes.conj().T

    # Only normal sources: charge and the three Pauli matrices.
    spin = np.array([eye, [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], np.diag([1, -1])])
    sources = np.zeros((4, 4, 4), complex)
    sources[:, :2, :2] = spin
    sources[:, 2:, 2:] = -spin.transpose(0, 2, 1)
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

    chi = (basis @ response @ basis.T).transpose(0, 2, 1, 3).reshape(4*n, 4*n)
    chi.flat[::4*n+1] += 1e-9  # Regularize the search direction in thermally frozen modes.
    return cho_factor(chi, check_finite=False)


def _constrained_bdg(h, delta, T, filling, mu, fields, tol, *, refine=False,
                     response_cache=None, ops):
    """Enforce uniform thermal charge and zero spin; pairing is unconstrained."""
    up = np.arange(0, h.shape[-1], 2)
    dn = up + 1
    if fields is None:
        fields = np.zeros((len(up), 4))

    thermal = ops.contractions
    add_fields = ops.add_fields
    local_values = ops.local_values
    response = ops.field_response

    def evaluate(values):
        values = values.reshape(len(up), -1).copy()
        if filling is None:
            values[:,0] -= values[:,0].mean()
        normal, gap = h.copy(), delta.copy()
        add_fields(normal, gap, values)
        energies, rho, pair = thermal(normal, gap, T, mu)
        target = filling
        if target is None:
            target = ops.filling(rho)
        error = local_values(rho, pair, target)
        return error.ravel(), (normal, gap, energies, rho, pair)

    error, state = evaluate(fields)
    # Try the last response first. Rebuild it at the current state if needed.
    # Every proposed field is tested against the actual thermal correlations.
    if filling is not None and response_cache is not None and np.max(abs(error)) > tol/2:
        for sweep in range(2):
            if not response_cache or sweep:
                try:
                    response_cache["factor"] = response(state[0], state[1], T, mu)
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
    if filling is not None:
        mu -= shift
        normal[..., up, up] -= shift
        normal[..., dn, dn] -= shift
    return rho, pair, normal, gap, energies, mu, fields


def _normal_barrier_bdg(h, delta, T, filling, mu, fields, m_max, tau, tol, *, ops, constrain_charge=False, normal_feedback=None):
    """Solve local normal fields with exact spin or finite barrier constraints.

    constrain_charge also bounds charge deviations; pairing remains free.
    One global source fixes the filling when specified.
    """
    up = np.arange(0, h.shape[-1], 2)
    dn = up + 1
    channels = 4 if constrain_charge else 3
    if fields is None:
        fields = np.zeros((len(up), channels))
    # A uniform potential shift enforces the number equation at fixed filling.
    x = np.r_[fields.ravel(), 0.0] if filling is not None else fields.ravel().copy()

    def evaluate(x):
        field = x[:channels*len(up)].reshape(-1, channels)
        bx, by, bz = field[:, -3:].T
        if constrain_charge:
            target = np.column_stack([_barrier_target(field[:,sl],m_max,tau)
                                      for sl in (slice(0,1),slice(1,4))])
        normal = h.copy() if normal_feedback is None else normal_feedback(target)
        normal[..., up, up] += bz
        normal[..., dn, dn] -= bz
        normal[..., up, dn] += bx - 1j*by
        normal[..., dn, up] += bx + 1j*by
        if constrain_charge:
            normal[..., up, up] += field[:, 0]
            normal[..., dn, dn] += field[:, 0]
        trial_mu = mu - x[-1] if filling is not None else mu
        thermal = ops.contractions
        energies, rho, pair = thermal(normal, delta, T, trial_mu)
        local_rho = rho.mean(axis=0)
        mag = np.column_stack((2*local_rho[up, dn].real, 2*local_rho[up, dn].imag,
                               (local_rho[up, up]-local_rho[dn, dn]).real))
        # Analytic inverse of b = 2*tau*m/(m_max^2-|m|^2), including b=0.
        if constrain_charge:
            local = np.column_stack(((local_rho[up,up]+local_rho[dn,dn]).real-filling, mag))
        else:
            local = mag
            target = np.zeros_like(field) if m_max is None else _barrier_target(field, m_max, tau)
        residual = target - local
        if constrain_charge:
            # The number equation below fixes total charge. Use its otherwise
            # redundant local mean to fix the charge-field gauge in field units.
            residual[:,0] += -residual[:,0].mean() + target[:,0].mean()*(2*tau/m_max**2)
        residual = residual.ravel()
        if filling is not None:
            residual = np.r_[residual, filling - local_rho.trace().real/len(up)]
        magnitude = np.linalg.norm(mag, axis=1)
        if constrain_charge:
            magnitude = np.column_stack((abs(local[:,0]),magnitude))
        return residual, (rho, pair, normal, energies, trial_mu, field), magnitude

    atol = min(tol/(2*np.sqrt(3)), 1e-12)
    if m_max is not None:
        atol = min(tol/2, 1e-12, m_max*1e-4)
        if constrain_charge:
            # Resolve bounded coordinates relative to the outer tolerance.
            atol = min(atol, m_max*tol/2)
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
        raise RuntimeError(f"Normal constraint solve failed: peak deviation = {magnitude.max():.6g}, "
                           f"field residual = {np.max(abs(err)):.3g}")
    costs = np.zeros_like(magnitude) if m_max is None else np.log1p(-(magnitude/m_max)**2)
    barrier = 0.0 if m_max is None else -tau*np.mean(costs.sum(axis=1) if constrain_charge else costs)
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
