from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import binplot
import pycalphad.variables as v
from datetime import date

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
fontsize = 20
params = {'legend.fontsize': fontsize,
          'axes.labelsize': fontsize*1.25,
          'axes.titlesize': fontsize*1.5,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,}
plt.rcParams.update(params)
fig, ax = plt.subplots(figsize=(20,15))
comps = ['CR', 'W', 'VA']
conditions = {v.T: (300, 4000, 25), v.P: 101325, v.N: 1, v.X('CR'): (0, 1, 0.01)}

binplot(db, comps, phases, conditions, plot_kwargs={'ax': ax})
fig.tight_layout()
plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
plt.savefig(f"./{comps[0]}-{comps[1]}-phase_diagram.png")