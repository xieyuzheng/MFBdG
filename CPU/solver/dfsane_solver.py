import numpy as np
import warnings

def spectral_coeff_outlier(fabs):
    """Handle extreme values of the residual norm to stabilize the step size."""
    if fabs < 1e-5:
        return 1e5
    elif fabs > 1e5:
        return 1
    else:
        return 1 / fabs

def spectral_coeff(sk, yk):
    """Compute spectral coefficient using Barzilai–Borwein formula."""
    numer = np.vdot(sk, sk).real
    denom = np.vdot(sk, yk).real

    if abs(denom) < 1e-14:
        return 1.0
        
    return numer / denom

def find_new_alpha(alpha, fold, fnew, tau_min = 0.1, tau_max = 0.5):
    alpha_t = alpha**2*fold / (fnew+ (2*alpha-1)*fold)
    alpha_new = np.clip(alpha_t, tau_min * alpha, tau_max * alpha)
    return alpha_new

def fmerit(Fvec):
    return np.linalg.norm(Fvec)**2

def dfsane_one_iteration(func, V, Fk, Vprev, Fprev, fkval, fkval_init, k, Cnum, Qnum,
                         gamma=1e-4, nu=0.85, tau_min = 0.1, tau_max = 0.5, is_feasible = None):
    """
    One iteration step for dfsane; assumes func(V) returns (F(V), extra_info).
    
    References:
    La Cruz, William, José Martínez, and Marcos Raydan. Mathematics of computation 75.255 (2006): 1429-1448.
    Cheng W, Li D H. IMA Journal of Numerical Analysis, 2009, 29(3): 814-825.
    
    """
    Vdiff = V - Vprev
    Fdiff = Fk - Fprev
    fknorm = np.sqrt(fkval)
    sigmak = spectral_coeff(Vdiff, Fdiff) if k > 0 else 1.0
    if not (1e-10 <= np.abs(sigmak) <= 1e10):
        sigmak = spectral_coeff_outlier(fknorm)

    epskfac = fkval_init / (1+k)**2
    dvec = -sigmak * Fk

    alpha_plus = alpha_minus = 1.0
    update_iter = 0

    while True:
        update_iter += 1
        for direction, alpha in [("plus", alpha_plus), ("minus", alpha_minus)]:
            sign = 1 if direction == "plus" else -1
            Vstep = V + sign * alpha * dvec

            if is_feasible is not None:
                if not is_feasible(Vstep):
                    if sign == 1:
                        alpha_plus *= 0.5
                    else:
                        alpha_minus *= 0.5
                    continue
            
            result = func(Vstep)
            if isinstance(result, tuple):
                Fkstep, *extra = result
            else:
                Fkstep, extra = result, None

            fkstep = fmerit(Fkstep)
            fk_crit = Cnum + epskfac - gamma * (alpha**2) * fkval
            if sign == 1:
                alpha_plus = find_new_alpha(alpha_plus, fkval, fkstep, tau_min = tau_min, tau_max = tau_max)
            elif sign == -1:
                alpha_minus = find_new_alpha(alpha_minus, fkval, fkstep, tau_min = tau_min, tau_max = tau_max)
            #print(f"k= {k}, C = {Cnum}, Q = {Qnum}, epsk = {epskfac}, fkval = {fkval} ", fkstep, fk_crit, fkstep<=fk_crit)
            
            if fkstep <= fk_crit:
                Qnum_new = nu * Qnum + 1
                Cnum_new = (nu * Qnum * (Cnum + epskfac) + fkstep) / Qnum_new
                
                return Vstep, Fkstep, fkstep, Cnum_new, Qnum_new, extra

        if update_iter >= 30:
            warnings.warn("Weird convergence during line search for df-sane method.")
            Cnum_new = fkstep
            Qnum_new = 1.0
            return Vstep, Fkstep, fkstep, Cnum_new, Qnum_new, extra
