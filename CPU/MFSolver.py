"""Shared BdG self-consistency with the RMFT CPU solvers."""

import numpy as np

from .solver.broyden_solver import Broyden_mixing, Broyden_update
from .solver.dfsane_solver import dfsane_one_iteration

ENERGY_DESCENT_VERSION = 5

def print_broyden_summary(updates, start, state, objective, residual):
    free0, objective0, residual0, reason = start
    print(f"Broyden fallback: {updates}/5 updates, dF/site = {state['free']-free0:+.6g}, "
          f"dObjective/site = {objective-objective0:+.6g}, "
          f"residual = {residual0:.6g} -> {residual:.6g} ({reason}).", flush=True)

def stop_reason(state, param):
    """Classify a solver result or saved NPZ using the same completion criteria."""
    err = float(state["err"])
    if not np.isfinite(err) or err < 0:
        return "invalid_residual" if np.isfinite(err) else "nonfinite_residual"

    ansatz = param.get("order_ansatz", "full")
    barrier = param.get("constraint_method") == "local_barrier"
    if not bool(state.get("bounds_satisfied", True)):
        return "constraint_failed"
    if barrier and ansatz == "pure_SC":
        cap = param.get("local_max", 1e-6)
        peaks = [state.get(key, np.inf) for key in ("max_dn", "max_m")]
        if not np.all(np.asarray(peaks) < cap):
            return "constraint_failed"
        if "thermal_local_peaks" in state and not np.all(np.asarray(state["thermal_local_peaks"])[:2] < cap):
            return "constraint_failed"
    spin_bound = param.get("m_max")
    if ansatz == "nonmagnetic":
        if barrier:
            spin_bound = param.get("local_max", param.get("m_max", 1e-6))
        elif not float(state.get("max_m", np.inf)) <= param["tolerate"]:
            return "constraint_failed"
    if spin_bound is not None and not float(state.get("max_m", np.inf)) < spin_bound:
        return "constraint_failed"

    recorded = str(state.get("stop_reason", ""))
    if recorded and recorded != "converged":
        return "iteration_limit" if recorded in ("Maximum iterations reached", "maxiterate") else recorded
    if err < param["tolerate"] and bool(state.get("converged", True)):
        return "converged"
    if int(state["iterate"]) >= param["maxiterate"]:
        return "iteration_limit"
    return "stopped"

def iteration_info(iterate, err, state):
    if "errset" in state:
        errors = state["errset"]
        values = ", ".join(f"{v:.6g}" for v in errors)
        residual = f"err = {err:.6g}, errset = [{values}]"
    else:
        residual = f"err = {err:.6g}"  # Older saved results.
    return f"iterate = {iterate}, {residual}, mu = {state['mu']:.8f}"

def broyden_depth(param):
    Broy_max = param.get("broy_max", param.get("broyden_max", 10))
    if isinstance(Broy_max, (bool, np.bool_)) or not isinstance(Broy_max, (int, np.integer)) or Broy_max < 1:
        raise ValueError("broy_max must be a positive integer.")
    return Broy_max


def Self_Consistent_dfsane(order_init, param, FGap, Plot_BdG=None):
    order = order_init.copy()
    Fm, result = FGap(order)
    order_prev, Fprev = order, Fm
    fmval = fmval_init = np.linalg.norm(Fm)**2
    Cnum, Qnum = fmval_init, 1.0
    iterate = 0
    err = np.max(np.abs(Fm), initial=0.0)

    while err >= param["tolerate"] and iterate < param["maxiterate"]:
        order_new, Fm_new, fmval, Cnum, Qnum, extrainfo = dfsane_one_iteration(
            FGap, order, Fm, order_prev, Fprev, fmval, fmval_init, iterate, Cnum, Qnum)
        order_prev, Fprev = order, Fm
        order, Fm = order_new, Fm_new
        result, = extrainfo
        iterate += 1
        err = np.max(np.abs(Fm), initial=0.0)
        if iterate % 100 == 0:
            print(f"iterate = {iterate}, err = {err:.6g}, mu = {result['mu']:.8f}", flush=True)
            if Plot_BdG:
                Plot_BdG(iterate, result)

    return result, iterate, err


def Self_Consistent_broyden(order_init, param, FGap, Plot_BdG=None):
    order = order_init.copy()
    Broy_max = broyden_depth(param)
    alpha = alpha_init = param.get("alpha", 0.4)
    broy_start, broyiter = 10, 0
    DFm_list, Um_list = [], []
    Am_max = np.zeros((Broy_max, Broy_max))
    iterate = 0
    Fm, result = FGap(order)
    err = np.max(np.abs(Fm), initial=0.0)

    while err >= param["tolerate"] and iterate < param["maxiterate"]:
        order_new = order + alpha * Fm
        if iterate >= broy_start and broyiter > 1:
            Fdiff = Fm - Fprev
            Vdiff = order - order_prev
            # An unchanged residual gives no direction to normalize.
            if np.linalg.norm(Fdiff) > np.finfo(float).eps:
                DFmat, Umat, Am_max = Broyden_update(
                    Fdiff, Vdiff, DFm_list, Um_list, Am_max, broyiter, alpha)
                Amat = Am_max[:broyiter-1, :broyiter-1]
                order_new = Broyden_mixing(order, Fm, DFmat, Umat, Amat, alpha, broyiter)
                broyiter += 1
                if broyiter - 2 >= Broy_max:
                    broyiter = 0
                    DFm_list.clear()
                    Um_list.clear()
                    alpha = alpha_init + 0.1 if err <= 1e-5 else alpha_init
        elif iterate >= broy_start:
            broyiter += 1

        order_prev, Fprev = order, Fm
        order = order_new
        Fm, result = FGap(order)
        iterate += 1
        err = np.max(np.abs(Fm), initial=0.0)
        if iterate % 100 == 0:
            print(f"iterate = {iterate}, err = {err:.6g}, mu = {result['mu']:.8f}", flush=True)
            if Plot_BdG:
                Plot_BdG(iterate, result)

    return result, iterate, err


def Self_Consistent_linear_mixing(order_init, param, FGap, Plot_BdG=None):
    order = order_init.copy()
    alpha = param.get("alpha", 0.4)
    iterate = 0
    Fm, result = FGap(order)
    err = np.max(np.abs(Fm), initial=0.0)

    while err >= param["tolerate"] and iterate < param["maxiterate"]:
        order = order + alpha * Fm
        Fm, result = FGap(order)
        iterate += 1
        err = np.max(np.abs(Fm), initial=0.0)
        if iterate % 100 == 0:
            print(f"iterate = {iterate}, err = {err:.6g}, mu = {result['mu']:.8f}", flush=True)
            if Plot_BdG:
                Plot_BdG(iterate, result)

    return result, iterate, err


def _descent_objective(state, param):
    value = state['free'] + state.get('local_barrier', state.get('spin_barrier', 0.))
    return value if param.get('n_filling') is not None else value-state['mu']*state['filling']


def _backtrack(trial, initial_step=1.):
    """Try at most ten successively halved, representation-specific steps."""
    step = initial_step
    for _ in range(10):
        accepted = trial(step)
        if accepted is not None:
            return accepted
        step *= .5
    return None


def energy_descent(param, evaluate, search, accept, *, prepare, failure, retry=None):
    """One descent/recovery loop for every constraint and representation.

    evaluate(i) returns (state, residual, feasible) at the accepted state.
    prepare(i) constructs directions; search(recovery) line-searches them.
    accept(trial) commits a feasible trial and returns the accepted step fraction.
    Constraint callbacks own their state, feasibility checks and predictors;
    only this loop owns stopping and five-update recovery after rejected descent trials.
    """
    remaining = 0
    summary_pending = False
    steps, modes = [], []
    for iteration in range(param['maxiterate']+1):
        result, err, feasible = evaluate(iteration)
        objective = _descent_objective(result, param)
        if summary_pending:
            print_broyden_summary(5, fallback_start, result, objective, err)
            summary_pending = False
        if (err < param['tolerate'] and feasible) or iteration == param['maxiterate']:
            break
        prepare(iteration)
        trial = search(remaining > 0)
        if trial is None and remaining == 0:
            remaining = 5
            fallback_start = (result['free'], objective, err, 'descent trials rejected')
            trial = search(True)
        if trial is None and retry is not None:
            trial = retry()
        if trial is None:
            result['stop_reason'] = failure(result, err)
            break
        modes.append('broyden' if remaining else 'energy_descent')
        step = accept(trial)
        steps.append(step)
        if remaining:
            remaining -= 1
            if remaining == 0:
                summary_pending = True
    if remaining:
        print_broyden_summary(5-remaining, fallback_start, result, objective, err)
    result.update(step_sizes=np.asarray(steps), step_modes=np.asarray(modes, dtype='U14'))
    return result, iteration, err


def Self_Consistent_energy_descent(order_init, param, FGap, Plot_BdG=None, *, evaluate_trials=None):
    """Adapt order-parameter trials to the shared energy-descent solver.

    Each FGap trial resets auxiliary fields to the accepted state. Independent
    directions may be evaluated concurrently without changing their ordering.
    """
    alpha = param.get("alpha", .4)
    depth = param.get("broy_max", 10)
    if isinstance(depth, (bool, np.bool_)) or not isinstance(depth, (int, np.integer)) or depth < 1:
        raise ValueError("broy_max must be a positive integer.")
    DFm_list, Um_list = [], []
    Am_max = np.zeros((depth, depth))
    order = order_init.copy()
    Fm, result = FGap(order)
    err = np.max(np.abs(Fm), initial=0.)
    energy_tol = 2e-12

    def potential(state):
        return _descent_objective(state, param)

    if not np.isfinite(potential(result)) or not np.isfinite(err):
        raise ValueError("Energy descent requires a finite initial objective and residual.")
    history = []
    directions = []
    evaluated = finite_trial = False

    def evaluate(iterate):
        history.append((iterate, potential(result), err))
        if iterate and iterate % 100 == 0:
            print(f"iterate = {iterate}, err = {err:.6g}, objective = {potential(result):.12g}", flush=True)
            if Plot_BdG:
                Plot_BdG(iterate, result)
        return result, err, True

    def prepare(iterate):
        nonlocal directions, Am_max, evaluated, finite_trial
        directions = [Fm]
        if iterate and np.linalg.norm(Fm-Fprev) > 1e-14:
            if len(DFm_list) == depth:
                DFm_list.clear()
                Um_list.clear()
            broyiter = len(DFm_list)+2
            try:
                DFmat, Umat, Am_max = Broyden_update(
                    Fm-Fprev, order-order_prev, DFm_list, Um_list, Am_max, broyiter, alpha)
                proposed = Broyden_mixing(order, Fm, DFmat, Umat, Am_max[:broyiter-1, :broyiter-1], alpha, broyiter)
                directions.append(proposed-order)
            except (RuntimeError, np.linalg.LinAlgError) as exc:
                print(f"Broyden proposal rejected; resetting history: {exc}", flush=True)
                DFm_list.clear()
                Um_list.clear()
                Am_max.fill(0.)
        evaluated = finite_trial = False

    def search(recovery):
        current_energy = potential(result)
        # Prefer the Broyden proposal in recovery, with direct mixing as backup.
        directions_to_try = (directions[1:] + [alpha*Fm]) if recovery else directions
        def trial_at(step):
            nonlocal evaluated, finite_trial
            trials = [order + step*direction for direction in directions_to_try]
            trials = [trial for trial in trials if np.all(np.isfinite(trial))]
            if evaluate_trials is not None:
                outcomes = evaluate_trials(trials, result, err)
            else:
                outcomes = []
                for trial in trials:
                    try:
                        outcomes.append(FGap(trial, result, err))
                    except (RuntimeError, np.linalg.LinAlgError):
                        outcomes.append(None)
            best = None
            for trial, outcome in zip(trials, outcomes):
                if outcome is None:
                    continue
                Ftrial, trial_result = outcome
                evaluated = True
                trial_err = np.max(np.abs(Ftrial), initial=0.)
                value = potential(trial_result)
                if not np.isfinite(value) or not np.isfinite(trial_err):
                    continue
                finite_trial = True
                if not trial_result.get('bounds_satisfied', True):
                    continue
                if not recovery and value > current_energy+energy_tol:
                    continue
                if (best is None or value < best[3]-energy_tol or
                        (abs(value-best[3]) <= energy_tol and trial_err < best[4])):
                    best = trial, Ftrial, trial_result, value, trial_err, step
                if recovery:
                    return best  # first feasible proposal, in Broyden-first order
            return best
        return _backtrack(trial_at)

    order_prev, Fprev = order, Fm
    def accept(best):
        nonlocal order, Fm, result, err, order_prev, Fprev
        order_prev, Fprev = order, Fm
        order, Fm, result, _, err, step = best
        return step

    def failure(state, error):
        reason = ("no_acceptable_step" if finite_trial else "nonfinite_trial") if evaluated else "trial_solve_failed"
        print(f"energy_descent stopped: {reason}, err = {error:.6g}", flush=True)
        return reason

    result, iterate, err = energy_descent(param, evaluate, search, accept,
                                         prepare=prepare, failure=failure)
    result.setdefault("stop_reason", "converged" if err < param["tolerate"] else "maxiterate")
    result["descent_history"] = np.array(history, dtype=[("iterate", int), ("objective", float), ("residual", float)])
    result['descent_policy_version'] = ENERGY_DESCENT_VERSION
    return result, iterate, err
