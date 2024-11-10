from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import Database, ternplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
comps = ['CR', 'MN', 'NI', 'VA']
system = '-'.join(comps[0:3])
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

for temp in [1273, 1373, 1473]:
    conds = {v.T: temp, v.P: 101325, v.X('MN'): (0, 1.01, 0.01), v.X('NI'): (0, 1.01, 0.01)}
    plt.figure(figsize=(20,20))
    ternplot(db, comps, phases, conds, x=v.X('NI'), y=v.X('MN')) 
    plt.title(f"{system} System, {temp} K, Do 2022")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.savefig(f'./{system}_{temp}K.png', facecolor='white', transparent=False)