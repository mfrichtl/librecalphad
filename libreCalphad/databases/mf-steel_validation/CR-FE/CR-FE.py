from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad.mapping import BinaryStrategy, plot_binary
import pycalphad.variables as v
from datetime import date

db = load_database('mf-steel.tdb')
comps = ['FE', 'CR', 'VA']
disabled_phases = ['BCC_B2', 'BCC_4SL', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'IONIC_LIQ']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
fontsize = 20
params = {'legend.fontsize': fontsize,
          'axes.labelsize': fontsize*1.25,
          'axes.titlesize': fontsize*1.5,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,}
plt.rcParams.update(params)
fig, ax = plt.subplots(figsize=(20,15))
conditions = {v.X('CR'): (0, 1, 0.01), v.T: (300, 2500, 20), v.P: 101325, v.N: 1}

binary = BinaryStrategy(db, comps, phases, conditions)
binary.initialize()
binary.do_map()
plot_binary(binary, ax=ax)
fig.tight_layout()
plt.savefig('./CR-FE-phase_diagram.png')
ax.set_xlim((0, 0.2))
ax.set_ylim((1000,1750))
plt.savefig('./CR-FE-gamma_loop.png')