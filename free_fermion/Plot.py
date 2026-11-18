"""Plot free-fermion bands from an MFBdG JSON input."""

import json
import sys
from pathlib import Path

import numpy as np

from bands import number_solution, plot_bands


inputfile = Path(sys.argv[1])
if inputfile.suffix != ".json":
    inputfile = Path("Input") / (str(inputfile) + ".json")
elif not inputfile.is_file():
    inputfile = Path("Input") / inputfile
with inputfile.open(encoding="utf-8") as f:
    param = json.load(f)

filling = param["n_filling"]
if not (np.isfinite(filling) and 0 < filling < 2):
    raise ValueError("Use 0 < n_filling < 2.")
lattice = param["lattice"]
nk = param.get("Nk", 1200)
if type(nk) is not int or nk < 2:
    raise ValueError("Nk must be an integer of at least 2.")
hopping = np.array([param["t1"], param.get("t2", 0), param.get("t3", 0)], dtype=float)
if not np.isfinite(hopping).all():
    raise ValueError("Hopping amplitudes must be finite.")

mu, band = number_solution(lattice, hopping, filling, nk)
folder = Path("Output") / inputfile.stem / "free_fermion"
folder.mkdir(parents=True, exist_ok=True)
label = f"n={filling}_t1={hopping[0]}_t2={hopping[1]}_t3={hopping[2]}_T=0"
output = folder / f"bands_{lattice}_{label}_Nk={nk}.png"
plot_bands(lattice, hopping, filling, mu, nk, band, output, param.get("Plotting", True))
print(f"Free fermions: n = {filling:.10f}, T = 0, mu = {mu:.10f}")
print(f"Plot saved to {output.resolve()}")
