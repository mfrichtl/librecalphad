from libreCalphad.databases.db_utils import load_database
from pycalphad import binplot, variables as v
import matplotlib.pyplot as plt

db = load_database('mf-steel.tdb')
components = ['FE', 'O', 'VA']
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
conditions = {v.T: (300, 2500, 15), v.P: 101325, v.N: 1, v.X('O'): (0, 1, 0.01)}
fontsize = 20
params = {'legend.fontsize': fontsize,
          'axes.labelsize': fontsize*1.25,
          'axes.titlesize': fontsize*1.5,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,}
plt.rcParams.update(params)

fig, ax = plt.subplots(figsize=(20,15))
ax = binplot(db, components, phases, conditions, plot_kwargs = {'ax': ax})
fig.tight_layout()
plt.savefig('./FE-O_phase_diagram.png')

