from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import ternplot
import pycalphad.variables as v

db = load_database('mf-steel.tdb')
disabled_phases = ['BCC_4SL', 'BCC_B2', 'FCC_L10', 'FCC_L12', 'FCC2_L10', 'HCP_L12', 'GAS', 'IONIC_LIQ']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]

comps = ['C', 'CR', 'FE', 'VA']
system = '-'.join(comps[0:3])
temps = [500, 1273, 1348]
for temp in temps:
    print(f"Plotting {temp}")
    plt.figure(figsize=(6,4))
    conds = {v.T: temp, v.P: 101325, v.X('CR'): (0, 1, 0.01), v.X('C'): (0, 1, 0.01)}
    ternplot(db, comps, phases, conds, x=v.X('CR'), y=v.X('C'))
    plt.title(f"{system} system, {temp} K")
    plt.tight_layout()
    plt.savefig(f'./{system}-{temp}K.png', facecolor='white', transparent=False)