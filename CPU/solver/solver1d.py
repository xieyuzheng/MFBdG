import numpy as np


def secant_method(f,x1, xrtol=5e-3, ftol = 1e-8, max_iter = 100, print_info = True):
    """Secant with bracket recovery for an increasing density residual.

    Small steps trigger recovery only without residual progress. Outside a
    bracket, expand toward the target; inside, fall back to bisection.
    Convergence requires abs(f) <= ftol. On exhaustion return the best sample.
    """
    if x1 >= 0:
        x0 = x1*(1 - 1e-2) - 1e-3
    else:
        x0 = x1*(1 + 1e-2) + 1e-3

    f1 = f(x1)
    if abs(f1) <= ftol:
        return x1
    f0 = f(x0)
    if abs(f0) <= ftol:
        return x0
    xsave, fsave = [x1, x0], [f1, f0]
    bracket = None
    if np.isfinite(f0) and np.isfinite(f1) and np.signbit(f0) != np.signbit(f1):
        (lo, flo), (hi, fhi) = sorted([(x0, f0), (x1, f1)])
        bracket = (lo, flo, hi, fhi)
    step = max(abs(x1-x0), 1e-3)
    best_error = min(abs(f0), abs(f1))
    stalled = 0
    for seci in range(max_iter+1):
        if np.abs(f1) <= ftol:
            return x1
        if seci == max_iter or not np.isfinite(f1):
            errors = np.where(np.isfinite(fsave), np.abs(fsave), np.inf)
            bestindex = np.argmin(errors)
            if print_info:
                reason = "maximum iter for mu" if seci == max_iter else "non-finite density for mu"
                print(reason, "; returning best mu; density residual =", errors[bestindex])
            return xsave[bestindex]

        fdiff = f1-f0
        scale = max(abs(f0), abs(f1), np.finfo(float).tiny)
        xnew = (x1-f1*(x1-x0)/fdiff if np.isfinite(fdiff)
                and abs(fdiff) > 8*np.finfo(float).eps*scale else np.nan)
        small_step = abs(xnew-x1) <= xrtol*max(abs(x1), .01)
        recover = stalled >= 2 or (stalled > 0 and small_step)
        if bracket is not None:
            lo, flo, hi, fhi = bracket
            if not np.isfinite(xnew) or not lo < xnew < hi or recover:
                xnew = lo+.5*(hi-lo)
        else:
            # Escape a plateau by expanding, rather than averaging same-sign points.
            direction = -np.sign(f1)
            if (not np.isfinite(xnew) or recover
                    or not 0 < direction*(xnew-x1) <= 4*step):
                xnew = x1+direction*step
                step *= 2
            else:
                step = max(step, abs(xnew-x1))

        fnew = f(xnew)
        if np.isfinite(fnew):
            if bracket is not None:
                if np.signbit(fnew) == np.signbit(flo):
                    bracket = (xnew, fnew, hi, fhi)
                else:
                    bracket = (lo, flo, xnew, fnew)
            elif np.isfinite(f1) and np.signbit(fnew) != np.signbit(f1):
                (lo, flo), (hi, fhi) = sorted([(x1, f1), (xnew, fnew)])
                bracket = (lo, flo, hi, fhi)
        stalled = 0 if abs(fnew) < .9*best_error else stalled+1
        best_error = min(best_error, abs(fnew))
        x0, f0, x1, f1 = x1, f1, xnew, fnew
        xsave.append(x1)
        fsave.append(f1)
