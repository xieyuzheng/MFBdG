"""Feasible full-covariance energy descent for the local log barrier."""
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import brentq
from scipy.special import entr

from CPU import MFSolver as mfsolver
from CPU.bond_bdg import (add_local_barrier, contractions, print_energy, _add_onsite_fields,
                          _barrier_target, _field_response, _local_barrier_energy,
                          _local_values, _solve_bdg_mu, _wick_energy)
from CPU.wick_bdg import gradE_to_BdG, order_to_Gfull, wick_pfaffian_grad_Gfull


def _covariance_errset(dr, dk):
    """Split the full covariance residual into onsite, normal-bond and pairing-bond maxima."""
    n = len(dr) // 2
    sites = np.arange(n)
    errset = np.zeros(3)
    for group, change in enumerate((dr, dk), start=1):
        blocks = np.abs(change).reshape(n, 2, n, 2)
        errset[0] = np.maximum(errset[0], blocks[sites, :, sites, :].max())
        blocks[sites, :, sites, :] = 0
        errset[group] = blocks.max()
    return errset


def _barrier_step(values, change, local_max):
    """Largest trial fraction inside every local ball, with margin at a crossed boundary."""
    step = 1.0
    for cols, cap in zip((slice(0, 1), slice(1, 4), slice(4, 6)),
                         (local_max, local_max, 2*local_max)):
        x, d = values[:, cols], change[:, cols]
        outside = np.linalg.norm(x + d, axis=1) >= cap
        if np.any(outside):
            x, d = x[outside], d[outside]
            a = np.sum(d*d, axis=1)
            b = np.sum(x*d, axis=1)
            slack = cap**2 - np.sum(x*x, axis=1)
            root = np.sqrt(b*b + a*slack)
            fraction = np.empty_like(root)
            outward = b >= 0
            fraction[outward] = slack[outward] / (root[outward] + b[outward])
            fraction[~outward] = (root[~outward] - b[~outward]) / a[~outward]
            step = min(step, .9*fraction.min())
    return step


def _barrier_trial(r, k, local, filling):
    """Match the trial's onsite coordinates and retain a physical Gaussian covariance."""
    n = len(r)//2
    current = _local_values(r, k, filling)
    change = local-current
    dn = change[:, 0]
    norm = np.max(abs(dn) + np.linalg.norm(change[:, 1:4], axis=1)
                  + np.linalg.norm(change[:, 4:], axis=1))/2
    # The local correction changes Nambu eigenvalues by at most norm.
    # Mixing with the uniform covariance by this fraction restores 0 <= G <= I.
    margin = min(filling/2, 1-filling/2)
    weight = margin/(margin+norm)
    r, k = weight*r, weight*k
    up = 2*np.arange(n)
    dn, mx, my, mz, re, im = (weight*local).T
    r[up, up] = filling/2 + (dn+mz)/2
    r[up+1, up+1] = filling/2 + (dn-mz)/2
    r[up, up+1] = (mx+1j*my)/2
    r[up+1, up] = r[up, up+1].conj()
    k[up, up+1] = (re+1j*im)/2
    k[up+1, up] = -k[up, up+1]
    return r, k


def _predict_barrier_state(h, gap, T, mu, target, local_max, tau, cache):
    """Take one safeguarded susceptibility step for the barrier search direction."""
    def inverse_field(b):
        groups = ((slice(0, 1), local_max), (slice(1, 4), local_max), (slice(4, 6), 2*local_max))
        return np.column_stack([_barrier_target(b[:, cols], cap, tau) for cols, cap in groups])

    n = len(h)//2
    field = cache.get("field")
    if field is None:
        field = np.zeros((n, 6))
    mu = cache.get("mu", mu)
    # A cached mu belongs to the previous predictor Hamiltonian. In particular,
    # the initial barrier fields can leave it on a nearly incompressible filling
    # plateau after those fields are removed. Restore the target filling before
    # using the local susceptibility to update the onsite fields.
    normal, pairing = h.copy(), gap.copy()
    _add_onsite_fields(normal, pairing, field)
    mu = _solve_bdg_mu(normal, pairing, T, target, mu, 1e-12, normal_first=True)

    def evaluate(b):
        local = inverse_field(b)
        normal, pairing = h.copy(), gap.copy()
        _add_onsite_fields(normal, pairing, b)
        _, r, k = contractions(normal, pairing, T, mu)
        k = (k-k.T)/2
        return _local_values(r, k, target)-local, r, k, normal, pairing

    residual, r, k, normal, pairing = evaluate(field)
    peak = np.max(abs(residual))
    if peak > 5e-14:
        if "response" not in cache or cache["age"] >= 12:
            try:
                cache["response"] = _field_response(normal, pairing, T, mu)
            except (np.linalg.LinAlgError, MemoryError):
                cache["response"] = None
            cache["age"] = 0
        factor = cache["response"]
        step = residual if factor is None else cho_solve(factor, residual.ravel()).reshape(n, 6)
        if not np.all(np.isfinite(step)):
            step = residual
        for backtrack in range(5):
            trial = field + 0.5**backtrack * step
            error, rt, kt, _, _ = evaluate(trial)
            if np.max(abs(error)) < peak:
                field, r, k = trial, rt, kt
                cache["age"] += 1
                break
        else:
            cache["age"] = 12
    # Fix the charge-field gauge and preserve its BdG Hamiltonian by shifting mu.
    q = field[:, 0]
    shift = q[0] if q.min()==q.max() else brentq(
        lambda s: _barrier_target(q[:, None]-s, local_max, tau).mean(), q.min(), q.max(), xtol=1e-15)
    field = field.copy()
    field[:, 0] -= shift
    cache.update(field=field, mu=mu-shift)
    local = inverse_field(field)
    return _barrier_trial(r, k, local, target)


def _bond_proposal(rho, pair, proposed, indices, memory, param, h0, terms):
    """Extrapolate the bond correlations using the existing RMFT Broyden update."""
    a, b = indices
    def vector(r, k):
        z = np.r_[r[a, b], k[a, b]]
        return np.r_[z.real, z.imag]
    v = vector(rho, pair)
    residual = vector(*proposed) - v
    last = memory.get("last")
    memory["last"] = (v, residual)
    if last is None or not a.size:
        return None
    dv, df = v-last[0], residual-last[1]
    if np.linalg.norm(df) <= 1e-14 or np.linalg.norm(dv) <= 1e-14:
        return None
    depth = mfsolver.broyden_depth(param)
    if "df" not in memory or len(memory["df"]) == depth:
        memory.update(df=[], du=[], gram=np.zeros((depth, depth)))
    count = len(memory["df"])+2
    alpha = param.get("alpha", .2)
    D, U, gram = mfsolver.Broyden_update(df, dv, memory["df"], memory["du"], memory["gram"], count, alpha)
    new = mfsolver.Broyden_mixing(v, residual, D, U, gram[:count-1, :count-1], alpha, count)
    real, imag = np.split(new, 2)
    chi, anomalous = np.split(real+1j*imag, 2)
    r, k = rho.copy(), pair.copy()
    r[a, b], r[b, a] = chi, chi.conj()
    k[a, b], k[b, a] = anomalous, -anomalous
    _, gradient = wick_pfaffian_grad_Gfull(order_to_Gfull(r, k), terms)
    return gradE_to_BdG(h0, gradient)


def mixed_barrier_descent(order_guess, h0, terms, param, mu, local_max, pack, plot):
    """Line search on the full Gaussian covariance at fixed filling."""
    n, T, target = len(h0), param["T"], param["n_filling"]
    tau, tol = param.get("tau", 1e-4), param["tolerate"]
    rho, pair = (np.array(v, dtype=complex, copy=True) for v in order_guess)
    uniform = np.eye(2*n)*target/2
    density0 = rho.diagonal().real.reshape(n, 2).sum(axis=1)
    rho[np.diag_indices(2*n)] += (target - density0.mean()) / 2
    # Repair infeasible onsite coordinates without scaling every bond by the
    # tiny barrier width. The covariance repair retains bond amplitudes up to
    # the mixing needed for positivity; the physical-state check below also
    # handles user seeds that were not valid covariances to begin with.
    values = _local_values(rho, pair, target)
    factor = _barrier_step(np.zeros_like(values), values, local_max)
    if factor < 1:
        rho, pair = _barrier_trial(rho, pair, factor*values, target)

    def score(r, k):
        barrier = _local_barrier_energy(r, k, target, local_max, tau)
        if (not barrier["bounds_satisfied"]
                or abs(r.trace().real/n - target) > tol):
            return None
        occupations = np.linalg.eigvalsh(order_to_Gfull(r, k))
        if occupations[0] < -2e-12 or occupations[-1] > 1+2e-12:
            return None
        f = occupations.clip(0, 1)
        entropy = np.sum(entr(f) + entr(1-f))/(2*n)
        energy = _wick_energy(h0, terms, r, k)
        free = energy - T*entropy
        objective = free + barrier["local_barrier"]
        return dict(**barrier, energy=energy, entropy=entropy, free=free, objective=objective)

    def try_step(trial_r, trial_k, *, allow_increase=False):
        change = _local_values(trial_r, trial_k, target) - old
        step = _barrier_step(old, change, local_max)
        dr, dk = trial_r-rho, trial_k-pair
        objective_limit = state["objective"] + 4*np.finfo(float).eps
        for _ in range(10):
            next_r, next_k = (trial_r, trial_k) if step == 1 else (rho + step*dr, pair + step*dk)
            trial = score(next_r, next_k)
            if trial is not None and (allow_increase or trial["objective"] <= objective_limit):
                return next_r, next_k, trial, step
            step /= 2
        return None

    state = score(rho, pair)
    for _ in range(64):
        if state is not None:
            break
        rho, pair = (rho + uniform)/2, pair/2
        state = score(rho, pair)
    if state is None:
        raise ValueError("Could not initialize a physical covariance inside the local barrier.")

    history, steps, step_modes = [], [], []
    broyden_remaining, small_steps = 0, 0
    recent_errors = []
    peaks = np.zeros(3)
    predictor, memory = {}, {}
    a, b = terms[1][1], terms[1][3]
    bonds = a//2 != b//2
    indices = a[bonds], b[bonds]
    bond_weights = terms[0][bonds]
    rng = np.random.default_rng(0)
    for iteration in range(param["maxiterate"] + 1):
        _, gradient = wick_pfaffian_grad_Gfull(order_to_Gfull(rho, pair), terms)
        h, gap = gradE_to_BdG(h0, gradient)
        old = _local_values(rho, pair, target)
        physical_h, physical_gap = h.copy(), gap.copy()
        fields = add_local_barrier(h, gap, old, local_max, tau, interior=True)
        mu = _solve_bdg_mu(h, gap, T, target, mu, min(tol/10, 1e-12),
                           normal_first=iteration == 0)
        energies, r, k = contractions(h, gap, T, mu)
        k = (k-k.T)/2
        dr, dk = r-rho, k-pair
        errset = _covariance_errset(dr, dk)
        err = errset.max()
        recent_errors.append(float(err))
        recent_errors = recent_errors[-4:]
        peaks = np.maximum(peaks, [state["max_dn"], state["max_m"], state["max_onsite_pair"]])
        result = dict(state, rho=rho, pair_order=pair, pair_bdg=k, order=pack(rho, pair),
                      filling=rho.trace().real/n, density=rho.diagonal().real.reshape(n, 2).sum(axis=1),
                      h=h-mu*np.eye(len(h)), delta=gap, energies=energies, mu=mu,
                      old_barrier_fields=fields, iterate=iteration, err=err, errset=errset, converged=err < tol)
        if iteration and iteration % 100 == 0:
            print(mfsolver.iteration_info(iteration, err, result), flush=True)
            print_energy(result)
            if plot:
                plot(iteration, result)
        history.append((iteration, mu, state["energy"], state["entropy"], state["free"],
                        state["local_barrier"], state["objective"], err, errset))
        if err < tol or iteration == param["maxiterate"]:
            thermal = _local_barrier_energy(r, k, target, local_max, tau)
            result["thermal_local_peaks"] = np.array([thermal["max_dn"], thermal["max_m"],
                                                      thermal["max_onsite_pair"]])
            result["converged"] = bool(err < tol and thermal["bounds_satisfied"])
            if result["converged"] or iteration == param["maxiterate"]:
                break
        if iteration == 0:
            predictor["mu"] = mu
        if (broyden_remaining == 0 and small_steps >= 3 and len(recent_errors) == 4
                and err >= 0.99*recent_errors[0]):
            broyden_remaining = 5
            print("Tiny descent steps with stalled residual; starting five feasible Broyden updates.", flush=True)
        candidates = []
        if iteration:
            direct = _predict_barrier_state(physical_h, physical_gap, T, mu, target, local_max, tau, predictor)
            proposal = _bond_proposal(rho, pair, direct, indices, memory, param, h0, terms)
            if proposal is not None:
                candidates.append(_predict_barrier_state(*proposal, T, mu, target, local_max, tau, predictor.copy()))
            candidates.append(direct)
        candidates.append((r, k))
        accepted = None
        for candidate in candidates:
            accepted = try_step(*candidate, allow_increase=broyden_remaining > 0)
            if accepted is not None:
                break
        if accepted is None and iteration == 0:
            # Give the usual predictor a chance if the initial thermal step failed.
            direct = _predict_barrier_state(physical_h, physical_gap, T, mu, target, local_max, tau, predictor)
            accepted = try_step(*direct, allow_increase=broyden_remaining > 0)
        if accepted is None and broyden_remaining == 0:
            # Prefer the Broyden proposal, using direct mixing while history builds.
            # The feasibility checks in score still apply when energy can increase.
            broyden_remaining = 5
            print("Descent trials rejected; starting five feasible Broyden updates.", flush=True)
            for candidate in candidates:
                accepted = try_step(*candidate, allow_increase=True)
                if accepted is not None:
                    break
        if accepted is None and bond_weights.size:
            # Perturb only active bond fields. The accepted covariance is still the line-search origin.
            print(f"{mfsolver.iteration_info(iteration, err, result)}: ordinary steps rejected; trying random bond fields",
                  flush=True)
            i, j = indices
            scale = abs(bond_weights) * min(0.01, err)
            for attempt in (1, 2):
                # Refine the cached onsite fields at the unperturbed state before adding noise.
                direct = _predict_barrier_state(physical_h, physical_gap, T, mu, target, local_max, tau, predictor)
                normal, gap_trial = physical_h.copy(), physical_gap.copy()
                noise = (0.1**attempt) * scale * (rng.standard_normal((2, len(i))) + 1j*rng.standard_normal((2, len(i)))) / np.sqrt(2)
                normal[i, j] += noise[0]
                normal[j, i] += noise[0].conj()
                gap_trial[i, j] += noise[1]
                gap_trial[j, i] -= noise[1]
                candidate = _predict_barrier_state(normal, gap_trial, T, mu, target, local_max, tau, predictor.copy())
                accepted = try_step(*candidate, allow_increase=broyden_remaining > 0)
                if accepted is not None:
                    change = accepted[2]["objective"] - state["objective"]
                    print(f"random retry {attempt}/2 accepted, step = {accepted[3]:.3g}, "
                          f"change in (F+B)/site = {change:.6g}", flush=True)
                    break
                print(f"random retry {attempt}/2 rejected", flush=True)
                accepted = try_step(*direct, allow_increase=broyden_remaining > 0)
                if accepted is not None:
                    print("refined predictor accepted", flush=True)
                    break
        if accepted is None:
            result["stop_reason"] = "no_acceptable_step"
            break
        mode = "broyden" if broyden_remaining else "energy_descent"
        if broyden_remaining:
            print(f"Broyden fallback update {6-broyden_remaining}/5, step = {accepted[3]:.6g}, "
                  f"change in (F+B)/site = {accepted[2]['objective']-state['objective']:.6g}", flush=True)
        rho, pair, state, step = accepted
        steps.append(step)
        step_modes.append(mode)
        small_steps = small_steps + 1 if step < 1e-4 else 0
        if broyden_remaining:
            broyden_remaining -= 1
            if broyden_remaining == 0:
                small_steps = 0
                recent_errors.clear()
                print("Five Broyden fallback updates completed; returning to energy descent.", flush=True)
    if not result["converged"]:
        result.setdefault("stop_reason", "Maximum iterations reached")
        print(result["stop_reason"], flush=True)
    if iteration % 100 or iteration == 0:
        print_energy(result)
        if plot:
            plot(iteration, result)
    names = ["iterate", "mu", "energy", "entropy", "free", "barrier", "objective", "err"]
    result.update(history=np.array(history, dtype=[(names[0], int)]+[(s, float) for s in names[1:]]+[("errset", float, (3,))]),
                  step_sizes=np.asarray(steps), step_modes=np.asarray(step_modes, dtype="U14"),
                  max_accepted_local=peaks)
    return result
