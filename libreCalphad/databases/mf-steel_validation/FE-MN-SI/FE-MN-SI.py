# Ternary @ 1273 K, Zheng2015 Figure 7
# There are clearly issues here that may cascade into martensite fitting functions.
from libreCalphad.databases.db_utils import load_database
from pycalphad import ternplot, variables as v
import matplotlib.pyplot as plt

db = load_database('mf-steel.tdb')
comps = ['FE', 'SI', 'MN', 'VA']
disabled_phases = ['FCC_L10', 'FCC_L12', 'BCC_D03', 'HCP_L12'] 
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
conds = {v.T: 1273, v.P: 101325, v.X('MN'): (0, 1, 0.01), v.X('SI'): (0, 1, 0.01)}

fig, ax = plt.subplots(figsize=(20,20))
ternplot(db, comps, phases, conds, x=v.X('MN'), y=v.X('SI'), plot_kwargs = {'ax': ax})
plt.title("FE-MN-SI System, 1273 K, Zheng2015 Figure 7a")
plt.savefig('./FE-MN-SI_1273K.png', facecolor='white', transparent=False)