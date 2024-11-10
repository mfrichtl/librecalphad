from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import Database, ternplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

comps = ['AL', 'CR', 'MN', 'VA']
system = '-'.join(comps[0:3])
for temp in [800+273, 900+273, 950+273, 970+273, 1010+273, 1070+273]:
    conds = {v.T: temp, v.P: 101325, v.X('AL'): (0, 1.005, 0.005), v.X('MN'): (0, 1.005, 0.005)}
    plt.figure(figsize=(20,20))
    ternplot(db, comps, phases, conds, x=v.X('CR'), y=v.X('AL'))
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title(f"{system} System, {temp} K, Cui 2017")
    plt.savefig(f'./{system}_{temp}K.png', facecolor='white', transparent=False)