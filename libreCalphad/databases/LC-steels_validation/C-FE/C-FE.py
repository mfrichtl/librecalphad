from datetime import date
from espei.datasets import load_datasets, recursive_glob
from espei.plot import dataplot
from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad.mapping import BinaryStrategy, plot_binary
import pycalphad.variables as v
import yaml

db = load_database("LC-steels-thermo.tdb")
disabled_phases = ["CEMENTITE_D011"]  # stable first
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
with open("../../run_param_gen.yaml", "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
datasets = load_datasets(recursive_glob(dataset_folder))

fig, ax = plt.subplots(figsize=(6, 4))
comps = ["C", "FE", "VA"]
conditions = {v.T: (300, 4000, 20), v.P: 101325, v.N: 1, v.X("C"): (0, 1, 0.01)}
binary = BinaryStrategy(db, comps, phases, conditions)
binary.initialize()
binary.do_map()
plot_binary(binary, ax=ax)
dataplot(comps, phases, conditions, datasets, ax=ax)
ax.set_xlim((0, 0.3))
plt.tight_layout()
plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
plt.savefig(f"./{comps[0]}-{comps[1]}-stable_phase_diagram.png")

disabled_phases = ["GRAPHITE"]  # metastable next
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
with open("../../run_param_gen.yaml", "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
datasets = load_datasets(recursive_glob(dataset_folder))

fig, ax = plt.subplots(figsize=(6, 4))
comps = ["C", "FE", "VA"]
conditions = {v.T: (300, 4000, 20), v.P: 101325, v.N: 1, v.X("C"): (0, 1, 0.01)}
binary = BinaryStrategy(db, comps, phases, conditions)
binary.initialize()
binary.do_map()
plot_binary(binary, ax=ax)
dataplot(comps, phases, conditions, datasets, ax=ax)
ax.set_xlim((0, 0.3))
plt.tight_layout()
plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
plt.savefig(f"./{comps[0]}-{comps[1]}-metastable_phase_diagram.png")
