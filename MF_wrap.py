"""Single-point real-space BdG wrapper. Run: python MF_wrap.py bond_sc Local"""

import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from Filepath import filepath
from CPU.bond_bdg import bond_model, contractions, Energy_total, onsite_constraint_method, print_energy, solve_from_order
from CPU.MFSolver import iteration_info, stop_reason
from CPU.gap_oneorbital import RANDOM_PAIR_SEEDS, PAIRING, check_filling, default_pair_seed, random_order_generate
from CPU.wick_bdg import density_terms
from CPU.result_io import compact_result


def write_progress(fileprog, param, iterate, err):
    progress = [param["n_filling"], param.get("U", 0), param["V1"], param.get("V2", 0),
                param.get("V3", 0), int(iterate), float(err)]
    fileprog.write_text(f"n_filling, U, V1, V2, V3, iterate, err\n{progress}\n", encoding="utf-8")


def model_from_input(param):
    h0, V = bond_model(param["Nx"], param["Ny"], lattice=param["lattice"],
                       boundary=param.get("Boundary_Condition", "PBC"),
                       t=(param["t1"], param.get("t2", 0), param.get("t3", 0)),
                       V=(param["V1"], param.get("V2", 0), param.get("V3", 0)))
    return h0, density_terms(V, param.get("U", 0))


def pair_seed_label(param):
    seed = param.get("pair_seed")
    if seed is None:
        _, terms = model_from_input(param)
        seed = default_pair_seed(terms)
    if isinstance(seed, list):
        digest = hashlib.sha256(json.dumps(seed).encode()).hexdigest()
        label = "custom_" + digest[:12]
    elif seed in RANDOM_PAIR_SEEDS or seed in PAIRING[param["lattice"]].PAIR_SEEDS:
        label = seed
    else:
        label = str(float(seed))
    return label


def parameter_label(param, keys=("t1", "t2", "t3", "U", "V1", "V2", "V3", "T")):
    parts = []
    for key in keys:
        value = float(param.get(key, 0))
        number = "0" if value == 0 else str(value).removesuffix(".0")
        parts.append(f"{key}={number}")
    return "_".join(parts)


def print_finished(state, param, folder_prog, start_time):
    reason = stop_reason(state, param)
    status = "Finished" if reason == "converged" else f"Saved ({reason})"
    print(f"{status} n = {state['filling']:.9f}, T = {param['T']}, "
          f"{iteration_info(state['iterate'], state['err'], state)}, tolerance = {param['tolerate']:g}")
    print(f"Output files are saved to {folder_prog.resolve()}")
    print(f"time taken = {perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
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
    check_filling(param.get("n_filling"))
    param.setdefault("order_ansatz", "full")
    param["constraint_method"] = onsite_constraint_method(param)
    if param["order_ansatz"] == "nonmagnetic" and param["constraint_method"] == "local_barrier":
        param.setdefault("local_max", 1e-6)
        param.setdefault("tau", 1e-4)
    print(param, flush=True)
    if param["order_ansatz"] == "full":
        reminder = "order_ansatz = full, constraint_method = None"
        if param.get("m_max") is not None:
            reminder += f", spin bound m_max = {param['m_max']:g}"
        print(reminder, flush=True)
    elif param["order_ansatz"] == "nonmagnetic":
        bound = "zero" if param["constraint_method"] == "Lagrange_multiplier" else f"|m_i| < {param['local_max']:g}"
        print(f"Local spin constrained to {bound}; local charge and onsite pairing remain free.", flush=True)

    if "orders" in param:
        raise ValueError("Remove 'orders' from the input. U and V determine all Wick fields.")
    if param.get("run_mode", "single_point") != "single_point":
        raise ValueError("This wrapper runs one parameter point at a time.")

    folder_prog = Path(outlocation) / inputfile.stem
    seed_label = pair_seed_label(param)
    legacy_parameters = parameter_label(param, ("t1", "t2", "t3", "V1", "V2", "V3", "T"))
    previous_label = f"{legacy_parameters}_{seed_label}"
    filling = str(float(param["n_filling"])).removesuffix(".0")
    legacy_filling_label = f"n={filling}_{previous_label}"
    legacy_u_prefix = parameter_label(param, ("U",))
    legacy_u_label = f"n={filling}_{legacy_u_prefix}_{previous_label}"
    legacy_seed_label = f"n={filling}_{parameter_label(param)}_{seed_label}"
    label = f"{legacy_seed_label}_{param['order_ansatz']}"
    output = folder_prog / f"{inputfile.stem}_gapdata_{label}.npz"
    fileprog = folder_prog / f"{inputfile.stem}_progress_{label}.dat"

    # Reuse an older matching run in place. New runs always use the parameter label.
    for suffix in (f"_{label}", f"_{legacy_seed_label}", f"_{legacy_u_label}", f"_{legacy_filling_label}",
                   f"_{previous_label}", f"_{seed_label}", ""):
        candidate = folder_prog / f"{inputfile.stem}_gapdata{suffix}.npz"
        if not candidate.is_file():
            continue
        with np.load(candidate) as saved:
            saved_param = json.loads(saved["parameters"].item())
            saved_param.setdefault("order_ansatz", "full")
            if saved_param["order_ansatz"] == "full" or (
                    saved_param["order_ansatz"] == "pure_SC" and saved_param.get("constraint_method") is None):
                saved_param["constraint_method"] = onsite_constraint_method(saved_param)
            if saved_param["order_ansatz"] == "nonmagnetic":
                # Normalize historical metadata without rewriting saved results.
                if saved_param.get("constraint_method") == "local_barrier" and "m_max" in saved_param:
                    old_bound = saved_param.pop("m_max")
                    saved_param.setdefault("local_max", 1e-6 if old_bound is None else old_bound)
                saved_param["constraint_method"] = onsite_constraint_method(saved_param)
                if saved_param["constraint_method"] == "local_barrier":
                    saved_param.setdefault("local_max", 1e-6)
                    saved_param.setdefault("tau", 1e-4)
            constrained = param["constraint_method"] != "Lagrange_multiplier" or "onsite_fields" in saved
            if param["order_ansatz"] == "nonmagnetic":
                constrained = "spin_fields" in saved and "max_m" in saved
            if param["order_ansatz"] == "pure_SC" and param["constraint_method"] == "local_barrier":
                marker = "objective" if param["solver"] == "energy_descent" else "old_barrier_fields"
                constrained = marker in saved and ("stop_reason" in saved or "converged" in saved)
            if param.get("m_max") is not None:
                constrained = constrained and "max_m" in saved
            if saved_param == param and constrained:
                reason = stop_reason(saved, param)
                if reason != "converged":
                    print(f"Skipping unfinished cache entry ({reason}): {candidate}")
                    continue
                output = candidate
                fileprog = folder_prog / f"{inputfile.stem}_progress{suffix}.dat"
                if not fileprog.is_file():
                    write_progress(fileprog, param, saved["iterate"], saved["err"])
                print(f"Data already exist: {output}")
                if "free" in saved:
                    print_energy(saved)
                else:
                    # Older files store the trial pairing. Recover the thermal pairing
                    # from the saved Hamiltonian (its normal block already includes -mu).
                    h0, terms = model_from_input(param)
                    energies, rho, pair = contractions(saved["h"], saved["delta"], param["T"])
                    print_energy(Energy_total(h0, terms, rho, pair, energies, param["T"]))
                print_finished(saved, param, folder_prog, start_time)
                sys.exit(0)

    fileprog.unlink(missing_ok=True)
    print(f"Starting at T = {param['T']}, n = {param['n_filling']}", flush=True)
    h0, terms = model_from_input(param)
    order_guess = random_order_generate(param, terms)

    Plot_BdG = None
    if param.get("Plotting", False):
        from CPU.Plot import plot_iteration_state
        Plot_BdG = partial(plot_iteration_state, param)

    result = solve_from_order(order_guess, h0, terms, param, Plot_BdG=Plot_BdG)

    result["stop_reason"] = stop_reason(result, param)
    result["converged"] = result["stop_reason"] == "converged"
    folder_prog.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".tmp.npz")
    np.savez(pending, **compact_result(result, terms), parameters=json.dumps(param))
    pending.replace(output)
    write_progress(fileprog, param, result["iterate"], result["err"])
    print_finished(result, param, folder_prog, start_time)
