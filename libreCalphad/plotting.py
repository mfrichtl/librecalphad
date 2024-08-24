"""
Functions to help make useful plots.
"""


import matplotlib.pyplot as plt
import numpy as np
from pycalphad import equilibrium, variables as v
from pycalphad.plot.utils import phase_legend

def step_plot(db, components, conditions, disabled_phases=[], yscale='log', ylims=(1e-5,1.1), fontsize=20):
    """
    Function to make a step plot.
    """
    
    fig, ax = plt.subplots(figsize=(25,15))
    params = {'legend.fontsize': fontsize,
              'axes.labelsize': fontsize*1.25,
              'axes.titlesize': fontsize*1.5,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,}
    plt.rcParams.update(params)

    phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
    eq = equilibrium(db, components, phases, conditions)
    eq_phases = []
    for entry in eq.Phase.squeeze():
        for phase in entry.values:
            if phase != '' and phase not in eq_phases:
                eq_phases.append(phase)
    phase_handles, phasemap = phase_legend(eq_phases)

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Phase Fraction')
    ax.set_ylim(ylims)
    ax.set_xlim((conditions[v.T][0], conditions[v.T][1]))
    ax.set_yscale(yscale)

    for name in eq_phases:
        phase_indices = np.nonzero(eq.Phase.values == name)
        plt.scatter(np.take(eq['T'].values, phase_indices[2]), eq.NP.values[phase_indices], color=phasemap[name])
        ax.legend(phase_handles, eq_phases, loc='lower right')

    return fig, ax