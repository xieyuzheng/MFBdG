import numpy as np

def Broyden_update(Fdiff, Vdiff, DFm_list, Um_list, Am_max, broyiter, alpha):
    Fdnorm = np.linalg.norm(Fdiff)
    DFm = Fdiff/Fdnorm
    DVm = (Vdiff)/Fdnorm
    Um = alpha*DFm + DVm
    DFm_list.append(DFm)
    Um_list.append(Um)
    DFmat = np.column_stack(DFm_list) # DFmat is (m-1)x(4N), where broyiter = (m-1)
    Umat = np.column_stack(Um_list)
    new_vector = np.conjugate(DFmat.T) @ DFm
    # Update last row and column directly in A (no new allocation)
    uindex = broyiter -1 -1 #since python starts from 0
    Am_max[:uindex+1, uindex] =  new_vector[:uindex+1]
    Am_max[uindex, :uindex+1] = np.conjugate(Am_max[:uindex+1, uindex])
    return DFmat, Umat, Am_max

def Broyden_mixing(Vm, Fm,DFmat,Umat, Am, alpha, broyiter, is_feasible = None):
    Vnext = Vm+ alpha*Fm
    I = np.eye(Am.shape[0]) # Am is (m-1)x(m-1)
    #print("det = ", np.linalg.det(0.05*I+ Am))
    Bmat = np.linalg.inv(0.05*I+ Am)
    cvec = DFmat.conj().T @ Fm
    Bmat = Bmat* cvec.T # all rows of B multiply by c
    Vmodmat = Umat@Bmat
    Vmod = np.sum(Vmodmat,axis=1) # sum all columns
    Vnext = Vnext - Vmod

    if is_feasible is not None:
        if not is_feasible(Vnext):
            dV = Vnext - Vm
            fac = 1.0

            for _ in range(30):
                fac *= 0.5
                Vtrial = Vm + fac * dV
                if is_feasible(Vtrial):
                    return Vtrial

            print("[broyden] WARNING: no feasible step after 30 reductions; "
                  "keeping previous order (no update).", flush=True)

            return Vm.copy()
    return Vnext
