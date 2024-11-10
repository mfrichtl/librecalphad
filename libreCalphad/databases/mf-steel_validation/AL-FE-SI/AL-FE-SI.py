from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import Database, ternplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

comps = ['AL', 'FE', 'SI', 'VA']
system = '-'.join(comps[0:3])
for temp in [873, 1273]:
    conds = {v.T: temp, v.P: 101325, v.X('SI'): (0, 1.01, 0.01), v.X('AL'): (0, 1.01, 0.01)}
    plt.figure(figsize=(20,20))
    ternplot(db, comps, phases, conds, x=v.X('SI'), y=v.X('AL')) 
    plt.title(f"{system} System, {temp} K, Liu 1999")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.savefig(f'./{system}_{temp}K.png', facecolor='white', transparent=False)