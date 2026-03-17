from datetime import date
from espei.datasets import load_datasets, recursive_glob
from espei.plot import dataplot, plot_endmember, plot_interaction
from libreCalphad.databases.db_utils import load_database
from libreCalphad.models.thermodynamics import calculate_energy_from_activity
import matplotlib.pyplot as plt
from pycalphad import Workspace, variables as v
from pycalphad.mapping import BinaryStrategy, plot_binary
from pycalphad.property_framework.metaproperties import IsolatedPhase
import yaml

lc_db = load_database("LC-steels-thermo.tdb")
with open("../../run_param_gen.yaml", "r") as f:
    dataset_folder = yaml.safe_load(f)["system"]["datasets"]
datasets = load_datasets(recursive_glob(dataset_folder))
# Use the LC-steels phases

# First plot stable phase diagrams
fig, ax = plt.subplots(figsize=(6, 4))
comps = ["C", "FE", "VA"]
conditions = {v.T: (300, 4000, 20), v.P: 101325, v.N: 1, v.X("C"): (0, 1, 0.01)}
disabled_phases = ["CEMENTITE_D011"]
phases = [phase for phase in list(lc_db.phases.keys()) if phase not in disabled_phases]
print(f"Plotting stable phase diagram with.")
binary = BinaryStrategy(lc_db, comps, phases, conditions)
binary.do_map()
plot_binary(binary, ax=ax)
dataplot(comps, phases, conditions, datasets, ax=ax)
ax.set_xlim((0, 0.3))
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
fig.tight_layout()
plt.savefig(f"./{comps[0]}-{comps[1]}-stable_phase_diagram.png")
plt.close()

# now plot metastable phase diagrams
disabled_phases = ["GRAPHITE"]  # metastable next
phases = [phase for phase in list(lc_db.phases.keys()) if phase not in disabled_phases]
print(f"Plotting metastable phase diagram with.")
fig, ax = plt.subplots(figsize=(6, 4))
binary = BinaryStrategy(lc_db, comps, phases, conditions)
binary.do_map()
plot_binary(binary, ax=ax)
dataplot(comps, phases, conditions, datasets, ax=ax)
ax.set_xlim((0, 0.3))
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
fig.tight_layout()
plt.savefig(f"./{comps[0]}-{comps[1]}-metastable_phase_diagram.png")
plt.close()

# Plot some energy calculations
temps = [800, 1200, 1600]
phases = [phase for phase in list(lc_db.phases)]
for temp in temps:
    print(f"Plotting energies at {temp} C.")
    wks = Workspace(
        lc_db,
        comps,
        phases,
        {v.X("C"): (0, 1, 0.01), v.T: temp, v.P: 101325, v.N: 1},
    )
    x = wks.get(v.X("C"))
    figGM, axGM = plt.subplots()
    figHM, axHM = plt.subplots()
    figSM, axSM = plt.subplots()

    for phase in wks.phases:
        metastable_wks = wks.copy()
        metastable_wks.phases = [phase]
        prop_GM = IsolatedPhase(phase, metastable_wks)(f"GM({phase})")
        prop_GM.display_name = f"GM({phase})"
        prop_HM = IsolatedPhase(phase, metastable_wks)(f"HM({phase})")
        prop_HM.display_name = f"HM({phase})"
        prop_SM = IsolatedPhase(phase, metastable_wks)(f"SM({phase})")
        prop_SM.display_name = f"SM({phase})"
        axGM.plot(x, wks.get(prop_GM), label=prop_GM.display_name)
        axHM.plot(x, wks.get(prop_HM), label=prop_HM.display_name)
        axSM.plot(x, wks.get(prop_SM), label=prop_SM.display_name)
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    axGM.legend()
    axHM.legend()
    axSM.legend()
    figGM.tight_layout()
    figHM.tight_layout()
    figSM.tight_layout()
    figGM.savefig(f"./GM-{temp}.png")
    figHM.savefig(f"./HM-{temp}.png")
    figSM.savefig(f"./SM-{temp}.png")
    plt.close()
    # plt.savefig(f"./energies-{temp}.png")

# specific interaction plots based on input data
fig, ax = plt.subplots()
plot_endmember(
    lc_db, comps, "CEMENTITE_D011", ("FE", "C"), "CPM_FORM", datasets=datasets, ax=ax
)
fig.tight_layout()
plt.savefig("./CPM_FORM-CEMENTITE_D011.png")
plt.close()
