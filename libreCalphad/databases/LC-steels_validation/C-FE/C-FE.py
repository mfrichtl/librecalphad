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

# Plot some energy calculations
temps = [800, 1200, 1600]
for temp in temps:
    wks = Workspace(
        db,
        ["C", "FE", "VA"],
        [phase for phase in list(db.phases)],
        {v.X("C"): (0, 0.5, 0.01), v.T: temp, v.P: 101325, v.N: 1},
    )
    x = wks.get(v.X("C"))
    fig, ax = plt.subplots()

    for phase in wks.phases:
        metastable_wks = wks.copy()
        metastable_wks.phases = [phase]
        prop_GM = IsolatedPhase(phase, metastable_wks)(f"GM({phase})")
        prop_GM.display_name = f"GM({phase})"
        prop_HM = IsolatedPhase(phase, metastable_wks)(f"HM({phase})")
        prop_HM.display_name = f"HM({phase})"
        prop_SM = IsolatedPhase(phase, metastable_wks)(f"SM({phase})")
        prop_SM.display_name = f"SM({phase})"
        ax.plot(x, wks.get(prop_GM), label=prop_GM.display_name)
        ax.plot(x, wks.get(prop_HM), linestyle=":", label=prop_HM.display_name)
        ax.plot(x, wks.get(prop_SM), linestyle="--", label=prop_SM.display_name)
    ax.legend()
    plt.savefig(f"./energies-{temp}.png")

# specific interaction plots based on input data
fig, ax = plt.subplots()
plot_endmember(
    db, comps, "CEMENTITE_D011", ("FE", "C"), "CPM_FORM", datasets=datasets, ax=ax
)
fig.tight_layout()
plt.savefig("./CPM_FORM-CEMENTITE_D011.png")
