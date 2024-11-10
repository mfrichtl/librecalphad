from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import ternplot
import pycalphad.variables as v
from datetime import date

db = load_database('mf-steel.tdb')
comps = ['B', 'CO', 'MO', 'VA']
system = '-'.join(comps[0:3])
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS', 'IONIC_LIQ']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

for temp in [1073, 1273]:
    conds = {v.T: temp, v.P: 101325, v.X('B'): (0, 1, 0.01), v.X('CO'): (0, 1, 0.01)}
    plt.figure(figsize=(20,20))
    ternplot(db, comps, phases, conds, x=v.X('MO'), y=v.X('CO')) 
    plt.title(f"{system} System, {temp} K, {date.today()}, Liu 2021")
    plt.savefig(f'./{system}_{temp}K.png', facecolor='white', transparent=False)
    print(f"Completed {temp} K")