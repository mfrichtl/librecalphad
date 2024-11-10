from libreCalphad.databases.db_utils import load_database
from pycalphad import binplot, variables as v
import matplotlib.pyplot as plt

db = load_database('mf-steel.tdb')
components = ['FE', 'N', 'VA']
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS', 'IONIC_LIQ']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
conditions = {v.T: (300, 2000, 15), v.P: 101325, v.N: 1, v.X('N'): (0, 1.01, 0.01)}
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
ax.set_xlim((0,1))
plt.savefig('./FE-N_phase_diagram.png')
