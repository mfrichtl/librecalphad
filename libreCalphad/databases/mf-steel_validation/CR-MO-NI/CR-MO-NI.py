from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import ternplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
comps = ['CR', 'MO', 'NI', 'VA']
system = '-'.join(comps[0:3])
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'GAS', 'IONIC_LIQ']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

for temp in [573, 773, 973, 850+273, 1050+273, 1250+273]:
    conds = {v.T: temp, v.P: 101325, v.X('CR'): (0, 1.01, 0.01), v.X('MO'): (0, 1.01, 0.01)}
    plt.figure(figsize=(20,20))
    ternplot(db, comps, phases, conds, x=v.X('CR'), y=v.X('MO')) 
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title(f"{system} System, {temp} K, Turchi 2006")
    plt.savefig(f'./{system}_{temp}K.png', facecolor='white', transparent=False)
    print(f"Completed {temp} K")