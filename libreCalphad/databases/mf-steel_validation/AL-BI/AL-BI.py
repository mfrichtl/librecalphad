from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import binplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
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
components = ['AL', 'BI', 'VA']
conditions = {v.T: (300, 1500, 10), v.P: 101325, v.N: 1, v.X('BI'): (0, 1.01, 0.01)}

binplot(db, components, phases, conditions, plot_kwargs={'ax': ax})
fig.tight_layout()
plt.xlim((0,1))
plt.savefig('./AL-BI-phase_diagram.png')