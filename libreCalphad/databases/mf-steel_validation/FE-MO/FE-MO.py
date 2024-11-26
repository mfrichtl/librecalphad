from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad.mapping import BinaryStrategy, plot_binary
import pycalphad.variables as v
from datetime import date

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_B2', 'BCC_4SL', 'FCC_L10', 'FCC2_L10', 'FCC_L12', 'HCP_L12', 'IONIC_LIQ', 'SIGMA_D8B', 'MU2_D85', 'TAU2_ALFEMO_A2']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

fig, ax = plt.subplots(figsize=(6,4))
comps = ['FE', 'MO', 'VA']
conditions = {v.T: (300, 3000, 10), v.P: 101325, v.N: 1, v.X('MO'): (0, 1, 0.01)}

binary = BinaryStrategy(db, comps, phases, conditions)
binary.initialize()
binary.do_map()
plot_binary(binary, ax=ax)
plt.tight_layout()
plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
plt.savefig(f"./{comps[0]}-{comps[1]}-phase_diagram.png")