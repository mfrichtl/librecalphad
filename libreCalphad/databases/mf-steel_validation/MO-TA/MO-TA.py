from libreCalphad.databases.db_utils import load_database
from pycalphad import Database, binplot, variables as v
import matplotlib.pyplot as plt

db = load_database('mf-steel.tdb')
comps = ['MO', 'TA', 'VA']
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
fontsize = 20
params = {'legend.fontsize': fontsize,
          'axes.labelsize': fontsize*1.25,
          'axes.titlesize': fontsize*1.5,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,}
plt.rcParams.update(params)
fig, ax = plt.subplots(figsize=(20,15))

binplot(db, comps, phases, {v.X('TA'): (0, 1, 0.01), v.T: (300, 3050+273, 15), v.P: 101325, v.N: 1}, plot_kwargs={'ax': ax})
plt.tight_layout()
plt.savefig('./MO-TA-phase_diagram.png')
ax.set_ylim((2873, 3050+273))
plt.savefig('./MO-TA-phase_diagram_high_temps.png')