from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad.mapping import BinaryStrategy, plot_binary
import pycalphad.variables as v
from datetime import date

db = load_database('mf-steel.tdb')
comps = ['AL', 'FE', 'VA']
disabled_phases = ['BCC_B2', 'FCC_L12', 'HCP_L12', 'FCC_L10', 'TAU2_ALFEMO_A2', 'ALCRFE_O1', 'LIQUID']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

fig, ax = plt.subplots(figsize=(8,6))

conditions = {v.X('AL'): (0, 1, 0.01), v.T: (300, 2000, 20), v.P: 101325, v.N: 1}

binary = BinaryStrategy(db, comps, phases, conditions)
binary.initialize()
binary.do_map()
plot_binary(binary, ax=ax)
fig.tight_layout()
plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
plt.savefig(f"./{comps[0]}-{comps[1]}-phase_diagram.png")