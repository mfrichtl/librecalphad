"""
Functions to help make useful plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from pycalphad import equilibrium
from pycalphad.plot.utils import phase_legend


def step_plot(db, components, conditions, disabled_phases=[], **fig_kw):
    """
    Function to make a step plot.
    """

    fig, ax = plt.subplots(**fig_kw)

    phases = [phase for phase in list(db.phases.keys()) if phase not in disabled_phases]
    eq = equilibrium(db, components, phases, conditions)
    eq_phases = []
    for entry in eq.Phase.squeeze():
        for phase in entry.values:
            if phase != "" and phase not in eq_phases:
                eq_phases.append(phase)
    phase_handles, phasemap = phase_legend(eq_phases)

    for name in eq_phases:
        phase_indices = np.nonzero(eq.Phase.values == name)
        plt.scatter(
            np.take(eq["T"].values, phase_indices[2]),
            eq.NP.values[phase_indices],
            color=phasemap[name],
        )
        ax.legend(phase_handles, eq_phases, loc="lower right")

    return fig, ax

