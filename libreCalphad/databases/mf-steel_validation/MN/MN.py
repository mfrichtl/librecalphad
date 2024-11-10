from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import calculate, ReferenceState, variables as v

db = load_database('mf-steel-3g.tdb')
components = ['MN', 'VA']
phases = ['BCC_A2', 'FCC_A1', 'LIQUID', 'HCP_A3', 'CUB_A13', 'CBCC_A12']
fig, ax = plt.subplots()
for phase in phases:
    res = calculate(db, components, phase, T=(0.01,3000,2), P=101325, N=1, output='heat_capacity')
    ax.plot(res.T, res.heat_capacity.squeeze(), label=phase)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Isobaric Heat Capacity (J/mol-forumula-K)")
ax.set_title("Bigdeli 2015 Figure 4")
fig.legend()
plt.savefig('./MN-heat_capacities_3000K.png')
ax.set_xlim((0,300))
plt.savefig('./MN-heat_capacities_300K.png')
plt.close()

fig, ax = plt.subplots()
mn_ref = ReferenceState('MN', 'CBCC_A12')

bcc_res = calculate(db, components, ['CBCC_A12'], T=(0.01,3000,5), P=101325, N=1, output='GM')

for phase in phases:
    res = calculate(db, components, phase, T=(0.01,3000,5), P=101325, N=1, output='GM')
    ax.plot(res.T, (res.GM.squeeze()-bcc_res.GM.squeeze())/1000, label=phase)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Gibbs Free Energy (kJ/mol-forumula)")
ax.set_title("Bigdelli 2016 Figure 8")
ax.set_ylim((-25,30))
fig.legend()
plt.savefig('./gibbs_CBCC_ref.png')
plt.close()

fig, ax = plt.subplots()
for phase in phases:
    res = calculate(db, components, phase, T=(0.01,3000,2), P=101325, N=1, output='SM')
    ax.plot(res.T, res.SM.squeeze(), label=phase)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Entropy (J/mol-forumula-K)")
ax.set_title("Bigdeli 2015 Figure 6")
fig.legend()
plt.savefig('./MN-entropy_3000K.png')
plt.close()

fig, ax = plt.subplots()
for phase in phases:
    res = calculate(db, components, phase, T=(0.01,3000,2), P=101325, N=1, output='HM')
    ax.plot(res.T, res.HM.squeeze()/1e4, label=phase)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Enthalpy (10 kJ/mol-forumula-K)")
ax.set_title("Bigdeli 2015 Figure 6")
ax.set_ylim((-2,14))
fig.legend()
plt.savefig('./MN-enthalpy_3000K.png')
plt.close()