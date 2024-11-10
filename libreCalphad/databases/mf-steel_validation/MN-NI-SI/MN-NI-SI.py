from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import ternplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
comps = ['MN', 'NI', 'SI', 'VA']
system = '-'.join(comps[0:3])
disabled_phases = ['BCC_D03', 'FCC_L10', 'HCP_L12', 'GAS']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

for temp in [1073, 1273]:
    conds = {v.T: temp, v.P: 101325, v.X('NI'): (0, 1.005, 0.005), v.X('SI'): (0, 1.005, 0.005)}
    plt.figure(figsize=(20,20))
    ternplot(db, comps, phases, conds, x=v.X('MN'), y=v.X('SI')) 
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title(f"{system} System, {temp} K, Hu 2011")
    plt.savefig(f'./{system}_{temp}K.png', facecolor='white', transparent=False)
    print(f"Completed {temp} K")