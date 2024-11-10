from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import Database, calculate, ReferenceState, variables as v

db = load_database('mf-steel-3g.tdb')
components = ['FE', 'VA']
phases = ['BCC_A2', 'FCC_A1', 'LIQUID', 'HCP_A3']
fig, ax = plt.subplots()
for phase in phases:
    res = calculate(db, components, phase, T=(0,3000,2), P=101325, N=1, output='heat_capacity')
    ax.plot(res.T, res.heat_capacity.squeeze(), label=phase)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Isobaric Heat Capacity (J/mol-forumula-K)")
ax.set_title("Chen 2001 Figure 5 & Bigdeli 2016 Figure 7")
fig.legend()
plt.savefig('./FE-heat_capacities_3000K.png')
ax.set_xlim((0,300))
plt.savefig('./FE-heat_capacities_300K.png')

fig, ax = plt.subplots()
fe_ref = ReferenceState('FE', 'BCC_A2')

bcc_res = calculate(db, components, ['BCC_A2'], T=(0.01,3000,5), P=101325, N=1, output='GM')

for phase in phases:
    res = calculate(db, components, phase, T=(0.01,3000,5), P=101325, N=1, output='GM')
    ax.plot(res.T, (res.GM.squeeze()-bcc_res.GM.squeeze())/1000, label=phase)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Gibbs Free Energy (kJ/mol-forumula)")
ax.set_title("Bigdelli 2016 Figure 6")
ax.set_ylim((-12, 15))
fig.legend()
plt.savefig('./gibbs_BCC_ref.png')