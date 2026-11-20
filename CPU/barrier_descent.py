"""Local-barrier covariance callbacks for the shared energy-descent solver."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq
from scipy.special import entr

from CPU import MFSolver as mfsolver
from CPU.bond_bdg import print_energy, _barrier_target


def _barrier_step(values, change, local_max):
    """Largest trial fraction inside every local ball, with margin at a crossed boundary."""
    step = 1.0
    for cols in (slice(0, 1), slice(1, 4)):
        cap = local_max
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


def _predict_barrier_state(h, gap, T, mu, target, local_max, tau, cache, ops):
    """Take one safeguarded susceptibility step for the barrier search direction."""
    local_values = ops.local_values
    thermal = ops.contractions
    add_fields = ops.add_fields
    solve_mu = ops.solve_mu
    response = ops.field_response
    def inverse_field(b):
        groups = ((slice(0, 1), local_max), (slice(1, 4), local_max))
        return np.column_stack([_barrier_target(b[:, cols], cap, tau) for cols, cap in groups])

    n = ops.nsite
    field = cache.get("field")
    if field is None:
        field = np.zeros((n, 4))
    mu = cache.get("mu", mu)
    # A cached mu belongs to the previous predictor Hamiltonian. In particular,
    # the initial barrier fields can leave it on a nearly incompressible filling
    # plateau after those fields are removed. Restore the target filling before
    # using the local susceptibility to update the onsite fields.
    normal, pairing = h.copy(), gap.copy()
    add_fields(normal, pairing, field)
    mu = solve_mu(normal, pairing, T, target, mu, 1e-12, normal_first=True)

    def evaluate(b):
        local = inverse_field(b)
        normal, pairing = h.copy(), gap.copy()
        add_fields(normal, pairing, b)
        _, r, k = thermal(normal, pairing, T, mu)
        k = ops.project_pair(k)
        return local_values(r, k, target)-local, r, k, normal, pairing

    residual, r, k, normal, pairing = evaluate(field)
    peak = np.max(abs(residual))
    if peak > 5e-14:
        if "response" not in cache or cache["age"] >= 12:
            try:
                cache["response"] = response(normal, pairing, T, mu)
            except (np.linalg.LinAlgError, MemoryError):
                cache["response"] = None
            cache["age"] = 0
        factor = cache["response"]
        if factor is not None:
            # The residual is thermal_local(field) - inverse_barrier(field).
            # Its negative derivative includes both susceptibilities. In gapped
            # modes the inverse-barrier term can dominate the thermal response.
            values, lower = factor
            triangle = np.tril(values) if lower else np.triu(values)
            jacobian = triangle @ triangle.T if lower else triangle.T @ triangle
            for site in range(n):
                for start, stop in ((0, 1), (1, 4)):
                    b = field[site, start:stop]
                    scale = tau/local_max
                    length = np.hypot(scale, np.linalg.norm(b))
                    alpha = local_max/(scale + length)
                    block = alpha*(np.eye(len(b)) - np.outer(b/(scale + length), b/length))
                    indices = slice(4*site + start, 4*site + stop)
                    jacobian[indices, indices] += block
            factor = cho_factor(jacobian, check_finite=False)
        step = residual if factor is None else cho_solve(factor, residual.ravel()).reshape(n, 4)
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
    return ops.repair(r, k, local, target)


def _bond_proposal(rho, pair, proposed, indices, memory, param, ops):
    """Extrapolate normal bonds and all pairing using the existing Broyden update."""
    a, _ = indices
    vector = ops.vector
    v = vector(rho, pair)
    residual = vector(*proposed) - v
    last = memory.get("last")
    memory["last"] = (v, residual)
    if last is None or not v.size:
        return None
    dv, df = v-last[0], residual-last[1]
    if np.linalg.norm(df) <= 1e-14 or np.linalg.norm(dv) <= 1e-14:
        return None
    depth = mfsolver.broyden_depth(param)
    if "df" not in memory or len(memory["df"]) == depth:
        memory.update(df=[], du=[], gram=np.zeros((depth, depth)))
    count = len(memory["df"])+2
    alpha = param.get("alpha", .4)
    D, U, gram = mfsolver.Broyden_update(df, dv, memory["df"], memory["du"], memory["gram"], count, alpha)
    new = mfsolver.Broyden_mixing(v, residual, D, U, gram[:count-1, :count-1], alpha, count)
    real, imag = np.split(new, 2)
    chi, anomalous = np.split(real+1j*imag, [len(a)])
    return ops.proposal(rho, pair, chi, anomalous)


def mixed_barrier_descent(order_guess, param, mu, local_max, pack, plot, *, ops):
    """Adapt feasible Gaussian covariances to the shared energy-descent solver."""
    n, T, target = ops.nsite, param["T"], param["n_filling"]
    local_values = ops.local_values
    thermal = ops.contractions
    add_barrier = ops.add_barrier
    solve_mu = ops.solve_mu
    barrier_energy = ops.barrier
    density = ops.density
    filling = ops.filling
    residual = ops.residual
    tau, tol = param.get("tau", 1e-5), param["tolerate"]
    rho, pair = (np.array(v, dtype=complex, copy=True) for v in order_guess)
    uniform = np.eye(2*n)*target/2
    rho, pair = ops.initial(rho, pair, target)
    # Repair infeasible onsite coordinates without scaling every bond by the
    # tiny barrier width. The covariance repair retains bond amplitudes up to
    # the mixing needed for positivity; the physical-state check below also
    # handles user seeds that were not valid covariances to begin with.
    values = local_values(rho, pair, target)
    factor = _barrier_step(np.zeros_like(values), values, local_max)
    if factor < 1:
        rho, pair = ops.repair(rho, pair, factor*values, target)

    def score(r, k):
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(k))):
            return None
        barrier = barrier_energy(r, k, target, local_max, tau)
        if (not barrier["bounds_satisfied"]
                or abs(filling(r) - target) > tol):
            return None
        occupations = ops.occupations(r, k)
        if (not np.all(np.isfinite(occupations))
                or occupations.min() < -2e-12 or occupations.max() > 1+2e-12):
            return None
        f = occupations.clip(0, 1)
        entropy = np.sum(entr(f) + entr(1-f))/(2*n*ops.nk)
        energy = ops.energy(r, k)
        free = energy - T*entropy
        objective = free + barrier["local_barrier"]
        if not np.isfinite(objective):
            return None
        return dict(**barrier, energy=energy, entropy=entropy, free=free, objective=objective)

    def predict(normal, pairing, cache):
        # A failed optional predictor must not discard the accepted state or
        # prevent the thermal direction and bounded recovery attempts.
        trial_cache = cache.copy()
        try:
            candidate = _predict_barrier_state(normal, pairing, T, mu, target,
                                               local_max, tau, trial_cache, ops)
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            print(f"Barrier predictor rejected: {exc}", flush=True)
            return None
        cache.update(trial_cache)
        return candidate

    def try_step(trial_r, trial_k, *, allow_increase=False):
        change = local_values(trial_r, trial_k, target) - old
        step = _barrier_step(old, change, local_max)
        dr, dk = trial_r-rho, trial_k-pair
        objective_limit = state["objective"] + 4*np.finfo(float).eps
        def trial_at(fraction):
            next_r, next_k = (trial_r, trial_k) if fraction == 1 else (rho + fraction*dr, pair + fraction*dk)
            try:
                trial = score(next_r, next_k)
            except (RuntimeError, np.linalg.LinAlgError):
                trial = None
            if trial is not None and (allow_increase or trial["objective"] <= objective_limit):
                return next_r, next_k, trial, fraction
            return None
        return mfsolver._backtrack(trial_at, step)

    state = score(rho, pair)
    for _ in range(64):
        if state is not None:
            break
        rho, pair = (rho + uniform)/2, pair/2
        state = score(rho, pair)
    if state is None:
        raise ValueError("Could not initialize a physical covariance inside the local barrier.")

    history = []
    peaks = np.zeros(3)
    predictor, memory = {}, {}
    indices, bond_weights = ops.indices, ops.bond_weights
    rng = np.random.default_rng(0)
    old = physical_h = physical_gap = r = k = result = None
    current_iteration, err = 0, np.inf

    def evaluate(iteration):
        nonlocal mu, old, physical_h, physical_gap, r, k, err, peaks, current_iteration, result
        current_iteration = iteration
        h, gap = ops.build(rho, pair)
        old = local_values(rho, pair, target)
        physical_h, physical_gap = h.copy(), gap.copy()
        fields = add_barrier(h, gap, old, local_max, tau, interior=True)
        mu = solve_mu(h, gap, T, target, mu, min(tol/10, 1e-12),
                           normal_first=iteration == 0)
        energies, r, k = thermal(h, gap, T, mu)
        unprojected_pair = k
        k = ops.project_pair(k)
        dr, dk = r-rho, k-pair
        errset = residual(dr, dk)
        err = errset.max()
        peaks = np.maximum(peaks, [state["max_dn"], state["max_m"], state["max_onsite_pair"]])
        result = dict(state, rho=rho, pair_order=pair, pair_bdg=k, order=pack(rho, pair),
                      filling=filling(rho), density=density(rho),
                      h=h-mu*np.eye(2*n), delta=gap, energies=energies, mu=mu,
                      old_barrier_fields=fields, iterate=iteration, err=err, errset=errset, converged=err < tol)
        measured = ops.order(r, unprojected_pair)[2]
        trial = ops.order(rho, pair)[2]
        result['unprojected_pair_residual'] = np.max(abs(measured-trial),initial=0.)
        if iteration and iteration % 100 == 0:
            print(mfsolver.iteration_info(iteration, err, result), flush=True)
            print_energy(result)
            if plot:
                plot(iteration, result)
        history.append((iteration, mu, state["energy"], state["entropy"], state["free"],
                        state["local_barrier"], state["objective"], err, errset))
        feasible = True
        if err < tol or iteration == param["maxiterate"]:
            thermal_bounds = barrier_energy(r, k, target, local_max, tau)
            result["thermal_local_peaks"] = np.array([thermal_bounds["max_dn"], thermal_bounds["max_m"]])
            result["converged"] = bool(err < tol and thermal_bounds["bounds_satisfied"])
            feasible = thermal_bounds['bounds_satisfied']
        if iteration == 0:
            predictor["mu"] = mu
        return result, err, feasible

    candidates, initial_predictor_tried = [], False
    def prepare(iteration):
        nonlocal candidates, initial_predictor_tried
        initial_predictor_tried = False
        candidates = []
        if iteration:
            direct = predict(physical_h, physical_gap, predictor)
            if direct is not None:
                try:
                    proposal = _bond_proposal(rho, pair, direct, indices, memory, param, ops)
                except (RuntimeError, np.linalg.LinAlgError) as exc:
                    print(f"Broyden proposal rejected; resetting history: {exc}", flush=True)
                    memory.clear()
                    proposal = None
                if proposal is not None:
                    candidate = predict(*proposal, predictor.copy())
                    if candidate is not None:
                        candidates.append(candidate)
                candidates.append(direct)
        candidates.append((r, k))

    def search(recovery):
        nonlocal initial_predictor_tried
        accepted = None
        for candidate in candidates:
            accepted = try_step(*candidate, allow_increase=recovery)
            if accepted is not None:
                break
        if accepted is None and current_iteration == 0 and not initial_predictor_tried:
            # Give the usual predictor a chance if the initial thermal step failed.
            initial_predictor_tried = True
            direct = predict(physical_h, physical_gap, predictor)
            if direct is not None:
                candidates.append(direct)
                accepted = try_step(*direct, allow_increase=recovery)
        return accepted

    def retry():
        accepted = None
        if bond_weights.size:
            # Perturb only active bond fields. The accepted covariance is still the line-search origin.
            print(f"{mfsolver.iteration_info(current_iteration, err, result)}: ordinary steps rejected; trying random bond fields",
                  flush=True)
            i, j = indices
            scale = abs(bond_weights) * min(0.01, err)
            for attempt in (1, 2):
                # Refine the cached onsite fields at the unperturbed state before adding noise.
                direct = predict(physical_h, physical_gap, predictor)
                normal, gap_trial = physical_h.copy(), physical_gap.copy()
                noise = (0.1**attempt) * scale * (rng.standard_normal((2, len(i))) + 1j*rng.standard_normal((2, len(i)))) / np.sqrt(2)
                ops.perturb(normal, gap_trial, noise)
                candidate = predict(normal, gap_trial, predictor.copy())
                if candidate is not None:
                    accepted = try_step(*candidate, allow_increase=True)
                if accepted is not None:
                    change = accepted[2]["objective"] - state["objective"]
                    print(f"random retry {attempt}/2 accepted, step = {accepted[3]:.3g}, "
                          f"change in (F+B)/site = {change:.6g}", flush=True)
                    break
                print(f"random retry {attempt}/2 rejected", flush=True)
                if direct is not None:
                    accepted = try_step(*direct, allow_increase=True)
                if accepted is not None:
                    print("refined predictor accepted", flush=True)
                    break
        return accepted

    def accept(accepted):
        nonlocal rho, pair, state
        rho, pair, state, step = accepted
        return step

    result, iteration, err = mfsolver.energy_descent(param, evaluate, search, accept,
        prepare=prepare, retry=retry, failure=lambda state,error: 'no_acceptable_step')
    if not result["converged"]:
        result.setdefault("stop_reason", "Maximum iterations reached")
        print(result["stop_reason"], flush=True)
    if iteration % 100 or iteration == 0:
        print_energy(result)
        if plot:
            plot(iteration, result)
    names = ["iterate", "mu", "energy", "entropy", "free", "barrier", "objective", "err"]
    result.update(history=np.array(history, dtype=[(names[0], int)]+[(s, float) for s in names[1:]]+[("errset", float, (3,))]),
                  max_accepted_local=peaks)
    return result
