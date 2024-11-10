from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad.mapping import TernaryStrategy, plot_ternary
import pycalphad.variables as v
from datetime import date

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'IONIC_LIQ']

phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

comps = ['AL', 'FE', 'NB', 'VA']
system = '-'.join(comps[0:3])
temps = [1073, 1273, 1423, 1573, 1723]
for temp in temps:
    fig, ax = plt.subplots(figsize=(13,10), subplot_kw={'projection': 'triangular'})
    conds = {v.T: temp, v.P: 101325, v.X(f"{comps[0]}"): (0, 1, 0.01), v.X(f"{comps[1]}"): (0, 1, 0.01)}
    ternary = TernaryStrategy(db, comps, phases, conds)
    ternary.initialize()
    ternary.do_map()
    plot_ternary(strategy=ternary, x=v.X('AL'), y=v.X('NB'), label_nodes=True, ax=ax)
    plt.title(f"{system} system, {temp} K, {date.today()}, Guo 2017")
    fig.tight_layout()
    plt.savefig(f'./{system}-{temp}K.png', facecolor='white', transparent=False)
    print(f"Completed {temp}.")
