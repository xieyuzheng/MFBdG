"""Feasible full-covariance energy descent for the local log barrier."""
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import brentq
from scipy.special import entr

from CPU import MFSolver as mfsolver
from CPU.bond_bdg import (add_local_barrier, contractions, print_energy, _add_onsite_fields,
                          _barrier_target, _constrained_bdg, _field_response, _local_barrier_energy,
                          _local_values, _solve_bdg_mu, _wick_energy)
from CPU.wick_bdg import gradE_to_BdG, order_to_Gfull, wick_pfaffian_grad_Gfull


def _barrier_step(values, change, widths):
    """Largest trial fraction inside every local ball, with margin at a crossed boundary."""
    step = 1.0
    for cols, cap in zip((slice(0, 1), slice(1, 4), slice(4, 6)),
                         (widths[0], widths[1], 2*widths[2])):
        if cap == 0:
            continue
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


def _predict_barrier_state(h, gap, T, mu, target, widths, tau, cache):
    """Take one safeguarded susceptibility step for the barrier search direction."""
    def inverse_field(b):
        groups = ((slice(0, 1), widths[0]), (slice(1, 4), widths[1]), (slice(4, 6), 2*widths[2]))
        return np.column_stack([_barrier_target(b[:, cols], cap, tau) for cols, cap in groups])

    n = len(h)//2
    field = cache.get("field")
    if field is None:
        field = np.zeros((n, 6))
    mu = cache.get("mu", mu)

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
        lambda s: _barrier_target(q[:, None]-s, widths[0], tau).mean(), q.min(), q.max(), xtol=1e-15)
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


def mixed_barrier_descent(order_guess, h0, terms, param, mu, widths, pack, plot):
    """Line search on the full Gaussian covariance at fixed filling or fixed mu."""
    n, T, target = len(h0), param["T"], param["n_filling"]
    tau, tol = param.get("tau", 1e-4), param["tolerate"]
    rho, pair = (np.array(v, dtype=complex, copy=True) for v in order_guess)
    filling = rho.trace().real/n if target is None else target
    uniform = np.eye(2*n)*filling/2
    density0 = rho.diagonal().real.reshape(n, 2).sum(axis=1)
    shift = filling-density0 if widths[0] == 0 else np.full(n, filling-density0.mean())
    rho[np.diag_indices(2*n)] += np.repeat(shift/2, 2)
    # Bring an infeasible user seed into the open domain along the segment
    # from the uniform spin-balanced state. Feasible seeds are used as supplied.
    values = _local_values(rho, pair, target)
    factor = _barrier_step(np.zeros_like(values), values, widths)
    rho, pair = uniform + factor*(rho-uniform), factor*pair

    def score(r, k):
        barrier = _local_barrier_energy(r, k, target, widths, tau)
        if (not barrier["bounds_satisfied"]
                or (widths[0] == 0 and barrier["max_dn"] > tol)
                or (target is not None and abs(r.trace().real/n - target) > tol)):
            return None
        occupations = np.linalg.eigvalsh(order_to_Gfull(r, k))
        if occupations[0] < -2e-12 or occupations[-1] > 1+2e-12:
            return None
        f = occupations.clip(0, 1)
        entropy = np.sum(entr(f) + entr(1-f))/(2*n)
        energy = _wick_energy(h0, terms, r, k)
        free = energy - T*entropy
        objective = free + barrier["local_barrier"]
        if target is None:
            objective -= mu*r.trace().real/n
        return dict(**barrier, energy=energy, entropy=entropy, free=free, objective=objective)

    state = score(rho, pair)
    for _ in range(64):
        if state is not None:
            break
        rho, pair = (rho + uniform)/2, pair/2
        state = score(rho, pair)
    if state is None:
        raise ValueError("Could not initialize a physical covariance inside the local barrier.")

    history, steps = [], []
    peaks = np.zeros(3)
    charge_fields = None
    predictor, memory = {}, {}
    a, b = terms[1][1], terms[1][3]
    bonds = a//2 != b//2
    indices = a[bonds], b[bonds]
    for iteration in range(param["maxiterate"] + 1):
        _, gradient = wick_pfaffian_grad_Gfull(order_to_Gfull(rho, pair), terms)
        h, gap = gradE_to_BdG(h0, gradient)
        old = _local_values(rho, pair, target)
        physical_h, physical_gap = h.copy(), gap.copy()
        fields = add_local_barrier(h, gap, old, widths, tau, interior=True)
        if target is None:
            h -= fields[:, 0].mean()*np.eye(len(h))
        if target is not None and (widths[0] > 0 or charge_fields is None):
            mu = _solve_bdg_mu(h, gap, T, target, mu, min(tol/10, 1e-12),
                               normal_first=iteration == 0)
        if widths[0] == 0:
            r, k, h, gap, energies, mu, charge_fields = _constrained_bdg(
                h, gap, T, target, mu, charge_fields, min(tol, 2e-12), refine=True, charge_only=True)
        else:
            energies, r, k = contractions(h, gap, T, mu)
        dr, dk = r-rho, (k-k.T)/2-pair
        err = max(np.max(abs(dr)), np.max(abs(dk)))
        peaks = np.maximum(peaks, [state["max_dn"], state["max_m"], state["max_onsite_pair"]])
        result = dict(state, rho=rho, pair_order=pair, order=pack(rho, pair),
                      filling=rho.trace().real/n, density=rho.diagonal().real.reshape(n, 2).sum(axis=1),
                      h=h-mu*np.eye(len(h)), delta=gap, energies=energies, mu=mu,
                      old_barrier_fields=fields, iterate=iteration, err=err, converged=err < tol)
        if charge_fields is not None:
            result["onsite_fields"] = charge_fields
        if iteration and iteration % 100 == 0:
            print(f"iterate = {iteration}, err = {err:.6g}, mu = {mu:.8f}", flush=True)
            print_energy(result)
            if plot:
                plot(iteration, result)
        history.append((iteration, mu, state["energy"], state["entropy"], state["free"],
                        state["local_barrier"], state["objective"], err))
        if err < tol or iteration == param["maxiterate"]:
            thermal = _local_barrier_energy(r, (k-k.T)/2, target, widths, tau)
            result["thermal_local_peaks"] = np.array([thermal["max_dn"], thermal["max_m"],
                                                      thermal["max_onsite_pair"]])
            result["converged"] = bool(err < tol and thermal["bounds_satisfied"]
                                       and (widths[0] > 0 or thermal["max_dn"] <= tol))
            if result["converged"] or iteration == param["maxiterate"]:
                break
        if iteration == 0 and target is not None and widths[0] > 0:
            predictor["mu"] = mu
        candidates = []
        if iteration and target is not None and widths[0] > 0:
            direct = _predict_barrier_state(physical_h, physical_gap, T, mu, target, widths, tau, predictor)
            proposal = _bond_proposal(rho, pair, direct, indices, memory, param, h0, terms)
            if proposal is not None:
                candidates.append(_predict_barrier_state(*proposal, T, mu, target, widths, tau, predictor.copy()))
            candidates.append(direct)
        candidates.append((r, (k-k.T)/2))
        accepted = False
        for trial_r, trial_k in candidates:
            change = _local_values(trial_r, trial_k, target) - old
            step = _barrier_step(old, change, widths)
            dr, dk = trial_r-rho, trial_k-pair
            for _ in range(5):
                next_r, next_k = (trial_r, trial_k) if step == 1 else (rho + step*dr, pair + step*dk)
                trial = score(next_r, next_k)
                if trial is not None and trial["objective"] <= state["objective"] + 4*np.finfo(float).eps:
                    rho, pair, state = next_r, next_k, trial
                    steps.append(step)
                    accepted = True
                    break
                step /= 2
            if accepted:
                break
        if not accepted and not (target is not None and widths[0] > 0):
            result["stop_reason"] = "No feasible downhill mixed-state step"
            break
    if not result["converged"]:
        result.setdefault("stop_reason", "Maximum iterations reached")
        print(result["stop_reason"], flush=True)
    if iteration % 100 or iteration == 0:
        print_energy(result)
        if plot:
            plot(iteration, result)
    names = ["iterate", "mu", "energy", "entropy", "free", "barrier", "objective", "err"]
    result.update(history=np.array(history, dtype=[(names[0], int)]+[(s, float) for s in names[1:]]),
                  step_sizes=np.asarray(steps), max_accepted_local=peaks)
    return result
