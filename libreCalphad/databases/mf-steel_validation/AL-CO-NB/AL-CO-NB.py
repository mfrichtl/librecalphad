from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import Database, ternplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS', 'IONIC_LIQ']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

comps = ['AL', 'CO', 'NB', 'VA']
system = '-'.join(comps[0:3])
for temp in [1073, 1273, 1423, 1473, 1523]:
    conds = {v.T: temp, v.P: 101325, v.X('AL'): (0, 1.01, 0.01), v.X('CO'): (0, 1.01, 0.01)}
    plt.figure(figsize=(20,20))
    ternplot(db, comps, phases, conds, x=v.X('AL'), y=v.X('NB'))
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title(f"{system} System, {temp} K, He 2015")
    plt.savefig(f'./{system}_{temp}K.png', facecolor='white', transparent=False)