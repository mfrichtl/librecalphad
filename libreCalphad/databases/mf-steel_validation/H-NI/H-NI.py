from libreCalphad.databases.db_utils import load_database
from pycalphad import binplot, variables as v
import matplotlib.pyplot as plt

db = load_database('mf-steel.tdb')
components = ['H', 'NI', 'VA']
disabled_phases = ['BCC_B2', 'BCC_D03', 'FCC_L10', 'FCC_L12', 'HCP_L12', 'IONIC_LIQ']
phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
conditions = {v.T: (300, 1900, 15), v.P: 101325, v.N: 1, v.X('H'): (0, 1.01, 0.01)}
fontsize = 20
params = {'legend.fontsize': fontsize,
          'axes.labelsize': fontsize*1.25,
          'axes.titlesize': fontsize*1.5,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,}
plt.rcParams.update(params)

fig, ax = plt.subplots(figsize=(20,15))
ax = binplot(db, components, phases, conditions, plot_kwargs = {'ax': ax})
ax.set_title(f"H-NI @ 101325 Pa, Bourgeois 2015 Figure 9")
fig.tight_layout()
ax.set_xlim((0, 0.012))
plt.savefig('./H-NI_phase_diagram.png')

conditions = {v.T: (300, 1900, 15), v.P: 1e7, v.N: 1, v.X('H'): (0, 1.01, 0.01)}
fig, ax = plt.subplots(figsize=(20,15))
ax = binplot(db, components, phases, conditions, plot_kwargs = {'ax': ax})
ax.set_title(f"H-NI @ 1e7 Pa, Bourgeois 2015 Figure 9")
fig.tight_layout()
ax.set_xlim((0, 0.012))
plt.savefig('./H-NI_phase_diagram_1e7Pa.png')