from datetime import date
from espei.datasets import load_datasets, recursive_glob
from espei.plot import dataplot, plot_endmember, plot_interaction
from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import Workspace, as_property, calculate, variables as v
from pycalphad.mapping import BinaryStrategy, plot_binary
from pycalphad.property_framework.metaproperties import IsolatedPhase
from tinydb import Query
import yaml

db = load_database("LC-steels-thermo.tdb")
with open("../../run_param_gen.yaml", "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
datasets = load_datasets(recursive_glob(dataset_folder))
comps = ["FE", "MN", "VA"]
disabled_phases = []
enabled_phases = ["BCC_A2", "CBCC_A12", "CUB_A13", "FCC_A1", "LIQUID"]
# phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
phases = enabled_phases
conditions = {v.T: (300, 2000, 15), v.P: 101325, v.N: 1, v.X("MN"): (0, 1, 0.01)}
fig, ax = plt.subplots(figsize=(8, 6))
binary = BinaryStrategy(db, comps, phases, conditions)
binary.do_map()
plot_binary(binary, ax=ax)
dataplot(comps, phases, conditions, datasets, ax=ax)
fig.tight_layout()
plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
plt.savefig(f"./{comps[0]}-{comps[1]}-phase_diagram.png")
plt.close()

# plot HM_MIX
phase_dict = {
    "LIQUID": (("FE", "MN"),),
    "FCC_A1": (("FE", "MN"), "VA"),
    "CUB_A13": (("FE", "MN"), "VA"),
}
for phase, configuration in phase_dict.items():
    fig, ax = plt.subplots()
    plot_interaction(
        db, comps, phase, configuration, "HM_MIX", datasets=datasets, ax=ax
    )
    ax.legend()
    fig.savefig(f"./{phase}-HM_MIX.png")
    plt.close()

# Plot CPM of BCC_A2 (Wituziewicz2003 Figure 3b)
fig, ax = plt.subplots()
# wks = Workspace(
#     db, comps, "BCC_A2", {v.N: 1, v.P: 101325, v.T: (350, 1105, 1), v.X("MN"): 0.0203}
# )
# heat_capacity = as_property("HM.T")
# heat_capacity.display_name = "Isobaric Heat Capacity"
# heat_capacity.display_units = "J/mol/K"
# ax.plot(wks.get(v.T), wks.get(heat_capacity))
# ax.set_xlabel(f"{v.T.display_name} [{v.T.display_units}]")
# ax.set_ylabel(f"{heat_capacity.display_name} [{heat_capacity.display_units}]")
# ax.set_title("Fe-0.0203Mn BCC_A2")
cpm_res = calculate(
    db, comps, "BCC_A2", T=(350, 1105, 2), P=101325, N=1, output="heat_capacity"
)
ax.plot(cpm_res.T, cpm_res.heat_capacity.squeeze())
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Isobaric Heat Capacity (J/mol-formula-K)")
fig.tight_layout()
plt.savefig("./CPM-FE-0.0203MN-BCC_A2.png")
plt.close()
