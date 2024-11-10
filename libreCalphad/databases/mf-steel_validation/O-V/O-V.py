# stable diagram
from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad.mapping import BinaryStrategy, plot_binary
from pycalphad import binplot
import pycalphad.variables as v
from datetime import date

db = load_database('mf-steel.tdb')
comps = ['O', 'V', 'VA']

disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'LIQUID']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
fontsize = 20
params = {'legend.fontsize': fontsize,
          'axes.labelsize': fontsize*1.25,
          'axes.titlesize': fontsize*1.5,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,}
plt.rcParams.update(params)
fig, ax = plt.subplots(figsize=(20,15))
conditions = {v.X('O'): (0, 1, 0.01), v.T: (300, 2800, 20), v.P: 101325, v.N: 1}

binary = BinaryStrategy(db, comps, phases, conditions)
binary.initialize()
binary.do_map()
plot_binary(binary, ax=ax)
# binplot(db, comps, phases, conditions, plot_kwargs={'ax': ax})
# ax.set_xlim((0,1))
fig.tight_layout()
plt.tight_layout()
plt.title(f"{comps[0]}-{comps[1]}, {date.today().strftime('%Y-%m-%d')}")
plt.savefig(f"./{comps[0]}-{comps[1]}-phase_diagram.png")