from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import binplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

fig, ax = plt.subplots(figsize=(8,6))

comps = ['C', 'CR', 'VA']
binplot(db, comps, phases, {v.X('C'): (0, 1.01, 0.01), v.T: (300, 2500, 10), v.P: 101325, v.N: 1}, plot_kwargs={'ax': ax})
ax.set_xlim((0, 1))
fig.tight_layout()
plt.savefig('./C-CR-phase_diagram.png')