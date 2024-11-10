from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import Database, ternplot
import pycalphad.variables as v
from datetime import date

db = load_database('mf-steel.tdb')
comps = ['CO', 'CR', 'MO', 'VA']
system = '-'.join(comps[0:3])
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

for temp in [927+273, 1373, 1473, 1573]:
    conds = {v.T: temp, v.P: 101325, v.X('CO'): (0, 1.01, 0.01), v.X('CR'): (0, 1.01, 0.01)}
    plt.figure(figsize=(20,20))
    ternplot(db, comps, phases, conds, x=v.X('CO'), y=v.X('MO')) 
    plt.title(f"{system} System, {temp} K, Ostrowska 2020, {date.today().strftime('%Y-%m-%d')}")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.savefig(f'./{system}_{temp}K.png', facecolor='white', transparent=False)
    print(f"Completed {temp}")