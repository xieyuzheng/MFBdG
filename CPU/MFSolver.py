"""Real-space BdG self-consistency with the RMFT CPU solvers."""

import numpy as np

from .solver.broyden_solver import Broyden_mixing, Broyden_update
from .solver.dfsane_solver import dfsane_one_iteration


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
        peaks = [state.get(key, np.inf) for key in ("max_dn", "max_m", "max_onsite_pair")]
        if not np.all(np.asarray(peaks) < cap):
            return "constraint_failed"
        if "thermal_local_peaks" in state and not np.all(state["thermal_local_peaks"] < cap):
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
        return "iteration_limit" if recorded == "Maximum iterations reached" else recorded
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
            print(iteration_info(iterate, err, result), flush=True)
            if Plot_BdG:
                Plot_BdG(iterate, result)

    return result, iterate, err


def broyden_depth(param):
    Broy_max = param.get("broy_max", param.get("broyden_max", 10))
    if isinstance(Broy_max, (bool, np.bool_)) or not isinstance(Broy_max, (int, np.integer)) or Broy_max < 1:
        raise ValueError("broy_max must be a positive integer.")
    return Broy_max


def Self_Consistent_broyden(order_init, param, FGap, Plot_BdG=None):
    order = order_init.copy()
    Broy_max = broyden_depth(param)
    alpha = alpha_init = param.get("alpha", 0.2)
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
            print(iteration_info(iterate, err, result), flush=True)
            if Plot_BdG:
                Plot_BdG(iterate, result)

    return result, iterate, err


def Self_Consistent_energy_descent(order_init, param, FGap, Plot_BdG=None, trial_batch=None):
    """Backtrack the thermal and Broyden steps using the trial state's free energy."""
    alpha, Broy_max = param.get("alpha", 0.2), broyden_depth(param)
    DFm_list, Um_list = [], []
    Am_max = np.zeros((Broy_max, Broy_max))
    order = order_init.copy()
    Fm, result = FGap(order)
    err = np.max(np.abs(Fm), initial=0.0)
    iterate = 0

    def potential(state):
        return state["free"] + state.get("local_barrier", state.get("spin_barrier", 0.0))

    while err >= param["tolerate"] and iterate < param["maxiterate"]:
        directions = [Fm]
        if iterate and np.linalg.norm(Fm - Fprev) > 1e-14:
            if len(DFm_list) == Broy_max:
                DFm_list.clear()
                Um_list.clear()
            broyiter = len(DFm_list) + 2
            DFmat, Umat, Am_max = Broyden_update(
                Fm - Fprev, order - order_prev, DFm_list, Um_list, Am_max, broyiter, alpha)
            Amat = Am_max[:broyiter-1, :broyiter-1]
            proposed = Broyden_mixing(order, Fm, DFmat, Umat, Amat, alpha, broyiter)
            directions.append(proposed - order)

        # Compare physical thermal states at the same accepted starting point.
        best = None
        evaluated = False
        finite_trial = False
        energy_tol = 2e-12  # per site
        current_energy = potential(result)
        for backtrack in range(5):
            trials = [order + 0.5**backtrack * direction for direction in directions]
            outcomes = trial_batch(trials, result, err) if trial_batch is not None else None
            for index, trial in enumerate(trials):
                try:
                    outcome = outcomes[index] if outcomes is not None else FGap(trial, result, err)
                    if outcome is None:
                        continue
                    Ftrial, trial_result = outcome
                except (RuntimeError, np.linalg.LinAlgError):
                    continue  # A failed trial constraint solve can retry at a shorter step.
                evaluated = True
                trial_err = np.max(np.abs(Ftrial), initial=0.0)
                value = potential(trial_result)
                if not np.isfinite(value) or not np.isfinite(trial_err):
                    continue
                finite_trial = True
                if value > current_energy + energy_tol:
                    continue
                if (best is None or value < best_energy - energy_tol or
                        (abs(value - best_energy) <= energy_tol and trial_err < best_err)):
                    best = trial, Ftrial, trial_result
                    best_energy, best_err = value, trial_err
            if best is not None:
                break
        if best is None:
            reason = "no_downhill_step" if finite_trial else "nonfinite_trial"
            if not evaluated:
                reason = "trial_solve_failed"
            result["stop_reason"] = reason
            print(f"energy_descent stopped: {reason}, err = {err:.6g}", flush=True)
            break

        order_prev, Fprev = order, Fm
        order, Fm, result = best
        err = best_err
        iterate += 1
        if iterate % 100 == 0:
            print(iteration_info(iterate, err, result), flush=True)
            if Plot_BdG:
                Plot_BdG(iterate, result)

    return result, iterate, err


def Self_Consistent_linear_mixing(order_init, param, FGap, Plot_BdG=None):
    order = order_init.copy()
    alpha = param.get("alpha", 0.2)
    iterate = 0
    Fm, result = FGap(order)
    err = np.max(np.abs(Fm), initial=0.0)

    while err >= param["tolerate"] and iterate < param["maxiterate"]:
        order = order + alpha * Fm
        Fm, result = FGap(order)
        iterate += 1
        err = np.max(np.abs(Fm), initial=0.0)
        if iterate % 100 == 0:
            print(iteration_info(iterate, err, result), flush=True)
            if Plot_BdG:
                Plot_BdG(iterate, result)

    return result, iterate, err
