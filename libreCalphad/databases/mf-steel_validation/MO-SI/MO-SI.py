from libreCalphad.databases.db_utils import load_database
from pycalphad import binplot, variables as v
import matplotlib.pyplot as plt

db = load_database('mf-steel.tdb')
components = ['MO', 'SI', 'VA']
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS', 'IONIC_LIQ']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

fontsize = 20
params = {'legend.fontsize': fontsize,
          'axes.labelsize': fontsize*1.25,
          'axes.titlesize': fontsize*1.5,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,}
plt.rcParams.update(params)
fig, ax = plt.subplots(figsize=(20,15))

binplot(db, components, phases, {v.X('SI'): (0, 1.01, 0.01), v.T: (300, 3000, 20), v.P: 101325, v.N: 1}, plot_kwargs={'ax': ax})
ax.set_xlim((0,1))
fig.tight_layout()
plt.savefig(f"./{'-'.join([components[0], components[1]])}-phase_diagram.png")
plt.close()