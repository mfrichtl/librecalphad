import numpy as np
import pandas as pd
from pycalphad import equilibrium, variables as v
from pymatgen.core import Composition


def DG(db, components, phases, conditions, calc_opts={}):
    """
    Function to calculate the Gibbs free energy difference between two phases.

    Parameters: db, pycalphad Database
                    The Database to use for calculations
                components, list
                    List of components included in the calculation.
                phases, list
                    List of the phases used to calculate the difference.
                conditions, dictionary
                    Dictionary containing the conditions to use for the calculations.
                calc_opts, dictionary
                    Dictionary containing keyword arguments for passing to pycalphad's calculate function.

    Returns: delta_g, float, J/mol
                Float of the molar Gibbs free energy difference, phases[0] - phases[1].
                Returns np.nan if there is a calculation error.
    """
    
    eq_GM = []
    try:
        for phase in phases:
            eq_GM.append(equilibrium(db, components, [phase], conditions, output='GM', calc_opts=calc_opts).GM.squeeze().values)
        return eq_GM[0] - eq_GM[1]
    except Exception as e:
        print(e)
        return np.nan
    

def parse_composition(row, dependent_element):
    """
    Function to parse a composition definition from a value in a Pandas Series object.
    Assumes the composition is defined with a label of 'material_at%' or 'material_wt%'.
    
    The input row is modified to include a String descriptor of the alloy system, a pymatgen.core.Composition
    object, and a pymatgen.core.Composition.chemical_system description.

    Parameters: row : Pandas Series
                    A row from a Pandas DataFrame of materials data.
                dependent_element : String
                    String indicating the dependent element.
    
    Returns: row : Pandas Series
                The modified row with the composition objects described above.
    """
    
    comp_dict = {}
    weight_percent = False
    interstitials = ['B', 'C', 'H', 'N', 'O']
    dependent_element = dependent_element.lower()
    material_col = 'material_at%'
    if pd.isnull(row['material_at%']):  # no atomic percent entered
        weight_percent = True
        material_col = 'material_wt%'

    # Parse hyphenated values
    split_mat = row[material_col].split('-')
    split_mat = [elem.lower() for elem in split_mat]
    assert dependent_element in split_mat, f"Did not identify the dependent component ({dependent_element}) in {split_mat}."
    frac = 0
    components = []
    for value in split_mat:
        if value == dependent_element:
            components.append(dependent_element.upper())
        elif value == '':
            continue
        else:
            element = ''
            for char in value:
                if char.isalpha():
                    element += char
            try:
                frac = float(value.split(element)[0]) / 100
                components.append(element.upper())
            except Exception as e:
                print(str(e))
                print(f"Identified element: {element}")
                print(row)
            comp_dict[element.capitalize()] = frac

    if 'VA' not in components:
        components.append('VA')
    row['components'] = components
    solute_fraction = np.sum(list(comp_dict.values()))
    comp_dict[dependent_element.capitalize()] = 1 - solute_fraction
    comp_set = None
    alloy = []
    conditions = {}
    try:
        if weight_percent:
            comp_set = Composition.from_weight_dict(comp_dict)
        else:
            comp_set = Composition(comp_dict)
        row['composition'] = comp_set
        row['system'] = comp_set.chemical_system
        for element, frac in comp_set.to_reduced_dict.items():
            row[str(element)] = frac
            # semi-arbitrary cutoff points for describing the alloy system, higher for substitutional elements
            if element.lower() == dependent_element:
                continue
            elif element in interstitials and frac > 0.0005:
                alloy.append(element.capitalize())
            elif frac > 0.002:
                alloy.append(element.capitalize())
        conditions.update({v.X(element.upper()): frac})
    except Exception as e:
        print(str(e))
        print(row)

    alloy.sort()
    row['alloy_system'] = dependent_element.capitalize() + '-' + '-'.join(alloy)
    row['conditions'] = conditions

    return row


def trim_conditions(components, conditions, max_num_conditions=1000, solute_threshold=1e-12, always_remove_list=[], always_keep_list=[]):
    """
    Function to trim conditions for a pycalphad calculation based on a variety of criteria. This is necessary because on certain occasions
    the equilibrium calculations do not function very well or cause problems if too many conditions are passed or if concentrations of
    certain elements are very low. This generally takes some trial and error to determine, but this function helps to systematically
    trim the offending conditions to enable bulk calculations.

    If the number of conditions exceeds to maximum allowed after removing those components in the always_remove_list or with concentrations less than the
    solute_threshold, the components with the lowest concentration will be removed first, unless they are in the always_keep_list.

    Parameters: components : array_like
                    The pycalphad component list.
                conditions, dictionary
                    The pycalphad conditions dictionary.
                max_num_conditions, int, defaults to 1000
                    The maximum number of conditions to be returned.
                solute_threshold, float, defaults to 1e-12
                    The minimum concentration threshold for an element to be returned.
                always_remove_list, array_like
                    A list of components to always be removed.
                always_keep_list, array_like
                    A list of components to always keep.

    Returns :   components, list
                    The modified components list.
                conditions, dictionary
                    The modificed conditions dictionary.
    """

    n = 0
    original_conditions = conditions.copy()
    if type(components) != np.ndarray:  # otherwise np.delete does not seem to work as expected.
        components = np.array(components)
    for key, val in original_conditions.items():
        if str(key).startswith('X_'): 
            component = str(key).split('_')[1]
            if component in always_remove_list or val < solute_threshold:  # trim very small quantities because they seem to cause problems.
                components = np.delete(components, np.where(components == component))
                conditions.pop(key)

    while len(conditions) > max_num_conditions:  # pycalphad seems to crash with too many conditions, kept getting stack smashing errors
        min_element = list(conditions.keys())[list(conditions.values()).index(sorted(conditions.values())[n])]
        component = str(min_element).split('_')[1]
        if component not in always_keep_list:
            components = np.delete(components, np.where(components == component))
            conditions.pop(min_element)
        n += 1
    
    return components, conditions


def convert_conditions(conditions):
    # Convert a conditions dictionary from strings to pycalphad.variables objects to prepare for calculations.

    converted_conditions = {}
    for key, value in conditions.items():
        if key == 'N':
            converted_conditions[v.N] = value
        elif key == 'P':
            converted_conditions[v.P] = value
        elif key == 'T':
            converted_conditions[v.T] = value
        elif key.startswith('X'):
            converted_conditions[v.X(key.split('_')[1])] = value
        else:
            raise NotImplementedError(f"Cannot identify pycalphad variable type for, {key} : {value}.")
    return converted_conditions


def get_components_from_conditions(conditions, dependent_component):
    # Return a list of components from conditions. Still needs a dependent component. Always adds VA.

    components = [dependent_component, 'VA']

    for key, value in conditions.items():
        if str(key).startswith('X'):
            components.append(str(key).split('_')[1])
    
    return components