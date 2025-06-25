from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import Workspace, variables as v
from pycalphad.mapping import BinaryStrategy, plot_binary
from pycalphad.property_framework.metaproperties import IsolatedPhase
from datetime import date

db = load_database("mf-steel.tdb")
disabled_phases = [
    "BCC_B2",
    "BCC_4SL",
    "FCC_L10",
    "FCC_L12",
    "HCP_L12",
    "GAS",
    "IONIC_LIQ",
]
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

fig, ax = plt.subplots(figsize=(6, 4))
comps = ["C", "FE", "VA"]
conditions = {v.T: (300, 2000, 20), v.P: 101325, v.N: 1, v.X("C"): (0, 1, 0.01)}
binary = BinaryStrategy(db, comps, phases, conditions)
binary.initialize()
binary.do_map()
plot_binary(binary, ax=ax)
plt.tight_layout()
plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
plt.savefig(f"./{comps[0]}-{comps[1]}-phase_diagram.png")

# Plot some energy calculations
temps = [800, 1200, 1600]
for temp in temps:
    wks = Workspace(
        db,
        ["C", "FE", "VA"],
        ["BCC_A2", "CEMENTITE_D011", "FCC_A1", "GRAPHITE_A9", "LIQUID"],
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
        ax.plot(x, wks.get(prop_GM), label=prop_GM.display_name)
        ax.plot(x, wks.get(prop_HM), linestyle=":", label=prop_HM.display_name)
    ax.legend()
    plt.savefig(f"./energies-{temp}.png")
