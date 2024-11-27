from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import binplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_B2', 'BCC_4SL', 'FCC_L10', 'FCC2_L10', 'FCC_L12', 'HCP_L12', 'GAS']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

fig, ax = plt.subplots(figsize=(8,6))
components = ['FE', 'NI', 'VA']
conditions = {v.T: (300, 2000, 10), v.P: 101325, v.N: 1, v.X('NI'): (0, 1, 0.01)}

binplot(db, components, phases, conditions, plot_kwargs={'ax': ax})
fig.tight_layout()
plt.savefig('./FE-NI-phase_diagram.png')