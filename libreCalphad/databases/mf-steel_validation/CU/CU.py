from libreCalphad.databases.db_utils import load_database
import matplotlib.pyplot as plt
from pycalphad import Database, calculate, ReferenceState, variables as v
from datetime import date

db = load_database('mf-steel-3g.tdb')
components = ['CU', 'VA']
phases = ['FCC_A1']
fig, ax = plt.subplots()
for phase in phases:
    res = calculate(db, components, phase, T=(0,1400,2), P=101325, N=1, output='heat_capacity')
    ax.plot(res.T, res.heat_capacity.squeeze(), label=phase)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Isobaric Heat Capacity (J/mol-forumula-K)")
ax.set_title(f"Heat Capacity from Khvan 2024 Figure 6, {date.today().strftime('%Y-%m-%d')}")
fig.legend()
ax.set_ylim((0,35))
plt.savefig('./CU-heat_capacities_1400K.png')
ax.set_xlim((0,100))
ax.set_ylim((0,18))
plt.savefig('./CU-heat_capacities_100K.png')

fig, ax = plt.subplots()

phases = ['FCC_A1', 'LIQUID']
for phase in phases:
    res = calculate(db, components, phase, T=(0.01,5000,5), P=101325, N=1, output='SM')
    ax.plot(res.T, res.SM.squeeze(), label=phase)

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Entropy (J/mol-forumula/K)")
ax.set_title(f"Khvan Figure 9, {date.today().strftime('%Y-%m-%d')}")
ax.set_ylim((0, 140))
fig.legend()
plt.savefig('./CU-entropy_5000K.png')
ax.set_xlim((0,200))
ax.set_ylim((0,25))
plt.savefig('./CU-entropy_100K.png')

fig, ax = plt.subplots()
for phase in phases:
    res = calculate(db, components, phase, T=(0,2000,2), P=101325, N=1, output='heat_capacity')
    ax.plot(res.T, res.heat_capacity.squeeze(), label=phase)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Isobaric Heat Capacity (J/mol-forumula-K)")
ax.set_title(f"Heat Capacity from Khvan 2024 Figure 10, {date.today().strftime('%Y-%m-%d')}")
fig.legend()
ax.set_ylim((0,40))
plt.savefig('./CU-LIQUID_heat_capacities_2000K.png')


fig, ax = plt.subplots()
phases = ['LIQUID', 'BCC_A2', 'HCP_A3', 'FCC_A1']

ref_phase = 'FCC_A1'
ref_state = ReferenceState('CU', ref_phase)
ref_calc = calculate(db, components, ref_phase, T=(0,3000,2), P=101325, N=1, output='GM')

for phase in phases:
    res = calculate(db, components, phase, T=(0,3000,2), P=101325, N=1)
    ax.plot(res.T, res.GM.squeeze()-ref_calc.GM.squeeze(), label=phase)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Gibbs Energy (J/mol-forumula)")
ax.set_title(f"Gibbs Energy from Khvan 2024 Figure 11, {date.today().strftime('%Y-%m-%d')}")
fig.legend()
ax.set_ylim((-10000,10000))
plt.savefig('./CU-Gibbs_energies_3000K.png')