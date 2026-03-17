from datetime import date
from espei.datasets import load_datasets, recursive_glob
from espei.plot import dataplot, plot_endmember, plot_interaction
from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import Workspace, variables as v
from pycalphad.mapping import BinaryStrategy, plot_binary
from pycalphad.property_framework.metaproperties import IsolatedPhase
import yaml

db = load_database("LC-steels-thermo.tdb")
with open("../../run_param_gen.yaml", "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
datasets = load_datasets(recursive_glob(dataset_folder))
comps = ["CU", "MG", "VA"]
disabled_phases = [
    "FCC_L10",
    "FCC2_L10",
    "FCC_L12",
    "BCC_B2",
    "BCC_4SL",
    "HCP_L12",
    "IONIC_LIQ",
]
# phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
# conditions = {v.T: (300, 2000, 15), v.P: 101325, v.N: 1, v.X("MN"): (0, 1, 0.005)}
# fig, ax = plt.subplots()
# binary = BinaryStrategy(db, comps, phases, conditions)
# binary.do_map()
# plot_binary(binary, ax=ax)
# dataplot(comps, phases, conditions, datasets, ax=ax)
# fig.tight_layout()
# plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
# plt.savefig(f"./{comps[0]}-{comps[1]}-phase_diagram.png")
# plt.close()

# plot HM_MIX for liquid (ESPEI example for plotting thermochemical data)
fig, ax = plt.subplots()
# ax.plot(x, wks.get(prop_HM), label=prop_HM.display_name)
plot_interaction(
    db, comps, "LIQUID", (("CU", "MG"),), "HM_MIX", datasets=datasets, ax=ax
)
ax.legend()
fig.savefig("./LIQUID-HM_MIX.png")
plt.close()
