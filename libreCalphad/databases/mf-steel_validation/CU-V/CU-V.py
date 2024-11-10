# stable diagram
from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import binplot
import pycalphad.variables as v
from datetime import date

db = load_database('mf-steel.tdb')
comps = ['CU', 'V', 'VA']
disabled_phases = ['FCC_L10', 'FCC_L12', 'BCC_B2', 'BCC_D03', 'HCP_L12', 'GAS', 'IONIC_LIQ']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
fontsize = 20
params = {'legend.fontsize': fontsize,
          'axes.labelsize': fontsize*1.25,
          'axes.titlesize': fontsize*1.5,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,}
plt.rcParams.update(params)

fig, ax = plt.subplots(figsize=(20,15))

binplot(db, comps, phases, {v.X('V'): (0, 1.01, 0.01), v.T: (300, 3200, 20), v.P: 101325, v.N: 1}, plot_kwargs={'ax': ax})
fig.tight_layout()
ax.set_xlim((0,1))
plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
plt.savefig(f"./{comps[0]}-{comps[1]}-phase_diagram.png")