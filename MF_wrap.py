"""Single-point periodic unit-cell BdG. Run: python MF_wrap.py Square_full_example Local"""

import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from Filepath import filepath
from CPU.bond_bdg import print_energy, onsite_constraint_method
from CPU.cell_bdg import model_from_input, solve, validate_boundary_condition
from CPU.parallel import allocation_from_input, print_allocation
from CPU.MFSolver import stop_reason, iteration_info


def write_progress(fileprog, param, iterate, err):
    progress = [param["n_filling"], param.get("U", 0), param["V1"], param.get("V2", 0),
                param.get("V3", 0), int(iterate), float(err)]
    fileprog.write_text(f"n_filling, U, V1, V2, V3, iterate, err\n{progress}\n", encoding="utf-8")


def pair_seed_label(param):
    seed = param.get("pair_seed")
    if seed is None:
        seed = "mixed" if any(param.get(f"V{i}", 0) for i in (1, 2, 3)) else 0.1
    if isinstance(seed, list):
        digest = hashlib.sha256(json.dumps(seed).encode()).hexdigest()
        label = "custom_" + digest[:12]
    elif isinstance(seed, str):
        label = seed
    else:
        label = str(float(seed))
    return label


def print_finished(state, param, folder_prog, start_time):
    reason = stop_reason(state,param)
    status = 'Finished' if reason=='converged' else f'Saved ({reason})'
    print(f"{status} n = {state['filling']:.9f}, T = {param['T']}, "
          f"{iteration_info(state['iterate'],state['err'],state)}, tolerance = {param['tolerate']:g}")
    print(f"Output files are saved to {folder_prog.resolve()}")
    print(f"total wrapper runtime = {perf_counter()-start_time:.2f} s")


def main():
    start_time = perf_counter()
    inputname = sys.argv[1]
    servername = sys.argv[2] if len(sys.argv) > 2 else "Local"
    location, outlocation = filepath(servername)

    inputfile = Path(location) / inputname
    if inputfile.suffix != ".json":
        inputfile = Path(location) / "Input" / (inputname + ".json")
    elif inputname == inputfile.name and not inputfile.is_file():
        inputfile = Path(location) / "Input" / inputname
    with inputfile.open(encoding="utf-8") as f:
        param = json.load(f)
    param.setdefault("Boundary_Condition", "PBC")
    try:
        validate_boundary_condition(param)
    except ValueError as error:
        print(f"Stopped: {error}", flush=True)
        return None
    param.setdefault("order_ansatz", "full")
    param.setdefault("tau", 1e-5)
    param.setdefault("alpha", 0.4)
    if "backend" in param:
        raise ValueError("Remove backend: every calculation uses the unit-cell solver. Set Nkx=Nky=1 for a finite periodic cell.")
    param.setdefault("Nkx", 1)
    param.setdefault("Nky", 1)
    plan = allocation_from_input(param)
    print_allocation(plan)
    model = model_from_input(param)
    print(param, flush=True)

    if "orders" in param:
        raise ValueError("Remove 'orders' from the input. U and V determine all Wick fields.")
    if param.get("run_mode", "single_point") != "single_point":
        raise ValueError("This wrapper runs one parameter point at a time.")

    folder_prog = Path(outlocation) / inputfile.stem
    seed_label = pair_seed_label(param)
    seed_label += f"_k{model.nkx}x{model.nky}"
    output = folder_prog / f"{inputfile.stem}_gapdata_{seed_label}.npz"
    fileprog = folder_prog / f"{inputfile.stem}_progress_{seed_label}.dat"
    if output.is_file():
        with np.load(output) as saved:
            saved_param = json.loads(saved["parameters"].item())
            saved_param.setdefault("order_ansatz", "full")
            method = onsite_constraint_method(param)
            constrained = True
            if param['order_ansatz'] == 'pure_SC':
                constrained = ('old_barrier_fields' if method == 'local_barrier' else 'onsite_fields') in saved
            if param['order_ansatz'] == 'nonmagnetic' or param.get('m_max') is not None:
                constrained = constrained and 'max_m' in saved
            current_descent = True
            if param['solver'] == 'energy_descent':
                from CPU.MFSolver import ENERGY_DESCENT_VERSION
                current_descent = ('descent_policy_version' in saved and
                                   saved['descent_policy_version'].item() == ENERGY_DESCENT_VERSION)
            if param['order_ansatz'] != 'full' or param.get('m_max') is not None:
                constrained = constrained and saved.get('momentum_constraint_version',0)==2
            if saved_param == param and constrained and current_descent and stop_reason(saved,param)=='converged':
                if not fileprog.is_file():
                    write_progress(fileprog, param, saved["iterate"], saved["err"])
                print(f"Data already exist: {output}")
                if "free" in saved:
                    print_energy(saved)
                else:
                    raise ValueError("Result is missing energy data; remove the incomplete output and rerun.")
                print_finished(saved, param, folder_prog, start_time)
                return output

    fileprog.unlink(missing_ok=True)
    print(f"Starting at T = {param['T']}, n = {param['n_filling']}", flush=True)
    Plot_BdG = None
    if param.get('Plotting',False):
        from CPU.Plot import plot_iteration_state
        Plot_BdG = partial(plot_iteration_state,param)
    result = solve(model, param, _allocation=plan, plot_iteration=Plot_BdG)

    result['stop_reason'] = stop_reason(result,param)
    result['converged'] = result['stop_reason']=='converged'
    folder_prog.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".tmp.npz")
    np.savez(pending, **result, parameters=json.dumps(param))
    pending.replace(output)
    write_progress(fileprog, param, result["iterate"], result["err"])
    print_finished(result, param, folder_prog, start_time)
    return output


if __name__ == '__main__':
    output = main()  # Keep the %run notebook interface for inspecting saved results.
