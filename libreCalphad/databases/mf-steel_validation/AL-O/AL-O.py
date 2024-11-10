from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import Database, binplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
comps = ['O', 'AL', 'VA']
disabled_phases = ['BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
fontsize = 20
params = {'legend.fontsize': fontsize,
          'axes.labelsize': fontsize*1.25,
          'axes.titlesize': fontsize*1.5,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,}
plt.rcParams.update(params)
fig, ax = plt.subplots(figsize=(20,15))

binplot(db, comps, phases, {v.X('O'): (0, 1, 0.01), v.T: (300, 3000, 15), v.P: 101325, v.N: 1}, plot_kwargs={'ax': ax})
fig.tight_layout()
plt.savefig('./AL-O-phase_diagram.png')
ax.set_xlim((0,0.004))
plt.savefig('./AL-O-phase_diagram-Al-rich.png')
ax.set_xlim((0.598,0.602))
plt.savefig('./AL-O-phase_diagram-corundum_invariant.png')