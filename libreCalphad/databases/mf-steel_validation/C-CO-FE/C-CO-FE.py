from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import ternplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS', 'IONIC_LIQ']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

comps = ['C', 'CO', 'FE', 'VA']
system = '-'.join(comps[0:3])
temps = [1273]
for temp in temps:
    plt.figure(figsize=(20,20))
    conds = {v.T: temp, v.P: 101325, v.X('C'): (0, 1.01, 0.01), v.X('CO'): (0, 1.01, 0.01)}
    ternplot(db, comps, phases, conds, x=v.X('C'), y=v.X('FE'))
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title(f"{system} system, {temp} K")
    plt.savefig(f'./{system}-{temp}K.png', facecolor='white', transparent=False)
    print(f"Completed {temp}")