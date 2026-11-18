"""Real-space BdG self-consistency with the RMFT CPU solvers."""

import numpy as np

from .solver.broyden_solver import Broyden_mixing, Broyden_update
from .solver.dfsane_solver import dfsane_one_iteration


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
            print(f"iterate = {iterate}, err = {err:.6g}, mu = {result['mu']:.8f}", flush=True)
            if Plot_BdG:
                Plot_BdG(iterate, result)

    return result, iterate, err


def Self_Consistent_energy_descent(order_init, param, FGap, Plot_BdG=None):
    """Backtrack the thermal and Broyden steps using the trial state's free energy."""
    alpha, Broy_max = param.get("alpha", 0.2), broyden_depth(param)
    DFm_list, Um_list = [], []
    Am_max = np.zeros((Broy_max, Broy_max))
    order = order_init.copy()
    Fm, result = FGap(order)
    err = np.max(np.abs(Fm), initial=0.0)
    iterate = 0

    def potential(state):
        value = state["free"] + state.get("local_barrier", state.get("spin_barrier", 0.0))
        if param["n_filling"] is None:
            value -= state["mu"] * state["filling"]
        return value

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
        energy_tol = 2e-12  # per site
        current_energy = potential(result)
        for backtrack in range(5):
            for direction in directions:
                trial = order + 0.5**backtrack * direction
                try:
                    Ftrial, trial_result = FGap(trial, result, err)
                except (RuntimeError, np.linalg.LinAlgError):
                    continue  # A failed trial constraint solve can retry at a shorter step.
                trial_err = np.max(np.abs(Ftrial), initial=0.0)
                value = potential(trial_result)
                if not np.isfinite(value) or not np.isfinite(trial_err):
                    continue
                if value > current_energy + energy_tol:
                    continue
                if (best is None or value < best_energy - energy_tol or
                        (abs(value - best_energy) <= energy_tol and trial_err < best_err)):
                    best = trial, Ftrial, trial_result
                    best_energy, best_err = value, trial_err
            if best is not None:
                break
        if best is None:
            result["stop_reason"] = "no downhill step"
            print(f"energy_descent stopped: no downhill step, err = {err:.6g}", flush=True)
            break

        order_prev, Fprev = order, Fm
        order, Fm, result = best
        err = best_err
        iterate += 1
        if iterate % 100 == 0:
            print(f"iterate = {iterate}, err = {err:.6g}, mu = {result['mu']:.8f}", flush=True)
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
            print(f"iterate = {iterate}, err = {err:.6g}, mu = {result['mu']:.8f}", flush=True)
            if Plot_BdG:
                Plot_BdG(iterate, result)

    return result, iterate, err
