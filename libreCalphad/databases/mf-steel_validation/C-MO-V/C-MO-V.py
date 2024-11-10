# Ternary @ 1273 K, Zheng2015 Figure 7
# There are clearly issues here that may cascade into martensite fitting functions.
from libreCalphad.databases.db_utils import load_database
from pycalphad import ternplot, variables as v
import matplotlib.pyplot as plt

db = load_database('mf-steel.tdb')
comps = ['C', 'MO', 'V', 'VA']
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'IONIC_LIQ', 'GAS'] 
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
system = '-'.join(comps[0:3])
temps = [2273]
fig, ax = plt.subplots(figsize=(20,20))
for temp in temps:
    conds = {v.T: temp, v.P: 101325, v.X('C'): (0, 1.01, 0.01), v.X('MO'): (0, 1.01, 0.01)}
    ternplot(db, comps, phases, conds, x=v.X('C'), y=v.X('MO'), plot_kwargs = {'ax': ax})
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    plt.title(f"{system} system, {temp} K, Bratberg 2002")
    plt.savefig(f'./{system}-{temp}K.png', facecolor='white', transparent=False)
    print(f"Completed {temp}")