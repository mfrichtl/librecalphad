"""
This is the primary code for interacting with the calphad open DB. Users will be able to select a database
they want to use and optionally provide lists of elements and phases they want to include in their calculations.
Functions in this file should import the necessary assessments to dynamically build the thermodynamic database
that provides the best results possible.
"""

from calphadOpenDB import databases
from importlib_resources import files
import numpy as np
import pandas as pd
from pycalphad import Database, equilibrium, variables as v

def importDB(system='LAS', elements=None, phases=None):
    """
    This is the primary function the user should interact with to construct the database for their system.
    """

    db = None
    if system == 'LAS':
        implemented_systems = ['C-FE', 'FE-NI']
        if elements == None:
            elements = list(set([x for x in '-'.join(implemented_systems).split('-')]))
        db1 = Database(str(files(pycalphadOpenDB.LAS).joinpath('./cfe_gus.tdb')))
    return db1


def generate_test_energies(dbf, dependent_component=None, desired_components=None, phases=None, num_points=5, seed=12345, overwrite=False):
    """
    This function is used to generate data for tests. It will generate data used for testing the provided database.
    
    If no data currently exists for the provided database, it will create a pickle file containing the energies
    of each implemented phase at different compositions and temperatures throughout the implemented ranges.
    
    If data does exist, it will compare the provided database to the existing data and generate data for any newly
    implemented phases.

    Inputs:
    dbf: String selecting which database to use. Must match a file in the 'databases' folder.
    dependent_component: The component to use as the dependent component in the energy calculations.
    num_points: An integer for the number of random points to generate for testing.
    seed: The seed for random number generations for the component fractions and temperatures.
    overwrite: Boolean to indicate whether to update or create an entirely new energy database file. 

    Returns:
    Nothing. The energy file will be created or updated.
    """

    assert dependent_component != None, "Define the dependent component for the Database."

    energy_file = None
    energies = None
    for f in files(databases).iterdir():
        if dbf in f.name:
            if '.pkl' in f.name:
                energy_file = str(f)
                print("Found existing energy file: " + f.name)
            else:
                print("Loading database: " + f.name)
                db_file = str(f)
                db = Database(db_file)

    if not energy_file:
        energy_file = db_file + '.pkl'
        print("No energy file found, creating a new one at " + energy_file)
    else:
        if overwrite:
            if phases and not desired_components:
                # remove entries matching the desired phases
                print(f"Overwriting all components in phases {phases} in file at " + energy_file)
                energies = pd.read_pickle(energy_file).query("phase not in @phases")
            if phases and desired_components:
                # remove entries matching the desired phases only containing all components
                print(f"Overwriting {desired_components} for {phases} in file " + energy_file)
                energies = pd.read_pickle(energy_file)
                energy_sub = pd.DataFrame([], columns=energies.columns)
                for comp_set in energies.loc[:, 'components']: 
                    if all([comp in comp_set for comp in desired_components]):
                        comp_set = [comp_set]
                        energy_sub = pd.concat([energy_sub, energies.query("phase not in @phases and components != @comp_set")])
                energies = energy_sub.copy()
            if not phases and desired_components:
                print(f"Overwriting for {desired_components} in file " + energy_file)
                energies = pd.read_pickle(energy_file)
            else:
                print("Overwriting energy file at " + energy_file)
                energies = pd.DataFrame([], columns=['phase', 'components', 'conditions', 'GM'])
        else:
            print("Updating existing energy file at " + energy_file)
            energies = pd.read_pickle(energy_file)
    
    assert type(energies) == pd.DataFrame, "Energy file not properly initialized."

    rng = np.random.default_rng(seed)
    temps = rng.integers(low=300, high=5000, size=num_points)
    fracs = rng.random(size=num_points) * 0.5  # Don't exceed a solute fraction of 0.5 for low-alloy steel database
    fracs = [round(x,4) for x in fracs]

    if phases == None:
        phases = db.phases
    else:
        phase_dict = {}
        for phase in phases:
            phase_dict[phase] = db.phases[phase]
        phases = phase_dict


    for phase in phases:
        num_sublattices = len(phases[phase].constituents)
        sites = phases[phase].sublattices
        num_constituents = 0
        constituents = []
        line_compound = False
        for i in np.arange(num_sublattices):
            num_constituents += len(phases[phase].constituents[i])
            for con in phases[phase].constituents[i]:
                if str(con) not in constituents and str(con) != 'VA':
                    if desired_components == None:
                        constituents.append(str(con))
                    else:
                        if str(con) in desired_components or str(con) == dependent_component:
                            constituents.append(str(con))
        if num_constituents == num_sublattices:  # line compound check, a specific stoichiometry must be maintained for calculations
            line_compound = True
        for i in np.arange(len(constituents)):  # unary and binary systems
            for j in np.arange(len(constituents)):   # ternary systems
                solutes = []
                if constituents[i] == constituents[j]:  # unary systems
                    components = [constituents[i], 'VA']
                else:
                    if constituents[i] == dependent_component:  # binary systems
                        components = [dependent_component, 'VA', constituents[j]]
                        solutes.append(constituents[j])
                    elif constituents[j] == dependent_component:  # binary systems
                        components = [dependent_component, 'VA', constituents[i]]
                        solutes.append(constituents[i])
                    else:  # ternary systems
                        components = [dependent_component, 'VA', constituents[i], constituents[j]]
                        solutes.append(constituents[i])
                        solutes.append(constituents[j])
                energy_sub = energies.query("phase == @phase")
                if set(components) not in [set(comps) for comps in np.unique(energy_sub.loc[:, 'components'].values)]:
                    print(f"Found new system: {phase}: {components}")
                    for T in temps:
                        conditions = {}
                        if len(components) == 2:  # unary compound, no need to set a fraction
                            conditions = {v.P: 101325, v.N: 1, v.T: T}
                        elif line_compound:
                            # assume dependent component on first sublattice, this may need to change
                            print("Line compound found")
                            x = np.sum(phases[phase].sublattices[1:])/np.sum(phases[phase].sublattices)
                            conditions = {v.P: 101325, v.N: 1, v.T: T, v.X(solutes[0]): x}
                        for x in fracs:
                            energy = np.nan
                            if len(components) == 3 and not line_compound:  # binary
                                conditions = {v.P: 101325, v.N: 1, v.T: T, v.X(solutes[0]): x}
                            if len(components) == 4 and not line_compound:  # ternary
                                conditions = {v.P: 101325, v.N: 1, v.T: T, v.X(solutes[0]): x, v.X(solutes[1]): x}
                            try:
                                energy = equilibrium(db, components, [phase], conditions).GM.squeeze().values
                                if ~np.isnan(energy):
                                    local_dict = {'phase': phase, 'components': [components], 'conditions': [conditions], 'GM': energy}
                                    energies = pd.concat([energies, pd.DataFrame((local_dict))])
                            except:
                                print(f"Error with {phase}, {components}. Components may be restricted to only one sublattice.")


    energies = energies.reset_index(drop=True)
    energies.to_pickle(energy_file)