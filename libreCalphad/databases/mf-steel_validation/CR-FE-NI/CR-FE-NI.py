from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad.mapping import TernaryStrategy, plot_ternary
import pycalphad.variables as v
from datetime import date

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_B2', 'BCC_4SL', 'FCC_L10', 'FCC2_L10', 'LAVES2_C14', 'FCC_L12', 'HCP_L12', 'IONIC_LIQ', 'SIGMA_D8B']

phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

comps = ['CR', 'FE', 'NI', 'VA']
system = '-'.join(comps[0:3])
# 2D plots
temps = [1073, 1193, 1273, 1673]
for temp in temps:
    fig, ax = plt.subplots(figsize=(10,10))
    conds = {v.T: temp, v.P: 101325, v.X(f"{comps[0]}"): (0, 1, 0.01), v.X(f"{comps[1]}"): (0, 1, 0.01)}
    ternary = TernaryStrategy(db, comps, phases, conds)
    ternary.initialize()
    ternary.do_map()
    plot_ternary(strategy=ternary, x=v.W('NI'), y=v.W('CR'), label_nodes=True, ax=ax)
    plt.title(f"{system} system, {temp} K, {date.today()}, Hillert 1990")
    fig.tight_layout()
    plt.savefig(f'./{system}-{temp}K.png', facecolor='white', transparent=False)
    print(f"Completed {temp}.")
