import numpy as np
import pandas as pd
from pycalphad import equilibrium, variables as v
from pymatgen.core import Composition

def DG(db, components, phases, conditions):
    """Calculate Gibbs free energy difference between two phases under provided conditions."""
    eq_GM = []
    try:
        for phase in phases:
            eq_GM.append(equilibrium(db, components, [phase], conditions, output='GM', calc_opts={'pdens': 1}).GM.squeeze().values)
        return eq_GM[0] - eq_GM[1]
    except Exception as e:
        print(e)
        return np.nan
    

def parse_composition(row):
    """
    Function to parse a composition definition from a value in a Pandas Series object.
    Assumes the composition is defined with a label of 'material_at%' or 'material_wt%'.
    
    The input row is modified to include a String descriptor of the alloy system, a pymatgen.core.Composition
    object, and a pymatgen.core.Composition.chemical_system description.

    Parameters: row : Pandas Series
                  A row from a Pandas DataFrame of materials data.
    
    Returns :   row : Pandas Series
                  The modified row with the composition objects described above.
    """
    
    comp_dict = {}
    weight_percent = False
    material_col = 'material_at%'
    if pd.isnull(row['material_at%']):  # no atomic percent entered
        weight_percent = True
        material_col = 'material_wt%'

    # Parse hyphenated values
    split_mat = row[material_col].split('-')
    alloy = []
    frac = 0
    conditions = {}
    components = []
    for value in split_mat:
        if value.lower() == 'fe':
            matrix = 'Fe'
            components.append('FE')
        elif value == '':
            continue
        else:
            element = ''
            for char in value:
                if char.isalpha():
                    element += char
            element = element.capitalize()
            try:
                frac = float(value.split(element)[0]) / 100
                # if (element != 'C' and element != 'N') and frac > 0.001:
                #     use_element = True
                # if (element == 'C' or element == 'N' or element == 'O') and frac > 0.0007:
                if frac > 0.0005:  # really small phase fractions seem to cause problems. This threshold is semi-arbitrarily chosen
                    alloy.append(element)
                    conditions.update({v.X(element.upper()): frac})
                    components.append(element.upper())
            except:
                print("ERROR WITH:")
                print(row)
            comp_dict[element] = frac
    
    assert matrix == 'Fe', "Non-iron or no matrix found"

    alloy.sort()
    row['components'] = components
    row['conditions'] = conditions
    row['alloy_system'] = 'Fe-' + '-'.join(alloy)
    solute_fraction = np.sum(list(comp_dict.values()))
    comp_dict[matrix] = 1 - solute_fraction
    comp_set = None
    try:
        if weight_percent:
            comp_set = Composition.from_weight_dict(comp_dict)
        else:
            comp_set = Composition(comp_dict)
        row['composition'] = comp_set
        row['system'] = comp_set.chemical_system
        
        for element, frac in comp_set.items():
            row[str(element)] = frac
    except:
        print("ISSUE WITH:")
        print(row)
    
    return row