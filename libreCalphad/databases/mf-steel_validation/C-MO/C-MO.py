from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import binplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

fig, ax = plt.subplots(figsize=(8,6))

comps = ['C', 'MO', 'VA']
binplot(db, comps, phases, {v.X('C'): (0, 1, 0.01), v.T: (500+273, 3500+273, 10), v.P: 101325, v.N: 1}, plot_kwargs={'ax': ax})

fig.tight_layout()
plt.savefig('./C-MO-phase_diagram.png')
ax.set_xlim((0, 0.5))
ax.set_ylim((1200,3200))
plt.savefig('./C-MO-phase_diagram_Mo-rich_Andersson1988b_Fig2.png')
ax.set_xlim((0, 10e-3))
ax.set_ylim((1600,2600))
plt.savefig('./C-MO-phase_diagram_BCC_C_solubility_Andersson1988b_Fig3.png')