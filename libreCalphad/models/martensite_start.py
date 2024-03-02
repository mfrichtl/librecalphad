from collections import defaultdict
import numpy as np
import pycalphad.variables as v
from scipy.optimize import curve_fit
import utilities

def get_plate_model_storm(components, conditions):
    """
    Get the plate martensite energy barrier from Stormvinter2012.

    Parameters: components : list
                    List of composition components.

                conditions : dictionary
                    Dictionary of conditions of pycalphad variables used in the model.

    Returns:    plate_barrier : float
                    The energy barrier for plate martensite formation provided by the Stormvinter2012 model.
    """     

    conditions = conditions.copy()
    comp_dict = defaultdict(lambda: 0)
    for component in components:
        if v.X(component.upper()) in conditions.keys():
            comp_dict.update({component.upper(): conditions[v.X(component.upper())]})

    plate_barrier = 2100 + 75000*comp_dict['C']**2/(1-comp_dict['C']) - 11500*comp_dict['C'] - 2970*comp_dict['CR'] \
        + 3574*comp_dict['MN'] - 5104*comp_dict['NI'] + 441700*comp_dict['C']*comp_dict['CR']/(1-comp_dict['C'])
    
    return plate_barrier


def get_lath_model_storm(T, components, conditions):
    """
    Get the lath martensite energy barrier from Stormvinter2012.

    Parameters: T : float
                    The martensite start temperature.

                components : list
                    List of composition components.

                conditions : dictionary
                    Dictionary of conditions of pycalphad variables used in the model.

    Returns:    lath_barrier : float
                    The energy barrier for lath martensite formation provided by the Stormvinter2012 model.
    """

    comp_dict = defaultdict(lambda: 0)
    for component in components:
        if v.X(component) in conditions.keys():
            comp_dict.update({component: conditions[v.X(component)]})

    lath_barrier = 3640 - 2.92*T +346400*comp_dict['C']**2/(1-comp_dict['C']) -16430*comp_dict['C'] - 785.5*comp_dict['CR'] \
        + 7119*comp_dict['MN'] - 4306*comp_dict['NI'] + 350600*comp_dict['C']*comp_dict['CR']/(1-comp_dict['C'])

    return lath_barrier


def fit_mf_model(orders, data):
    """
    Fit a new martensite model. Redlich-Kister summations are used for the fitting.

    Parameters: orders, dictionary
                    Dictionary of the fitting orders for each system included in the model.
                data, pandas.DataFrame
                    Fitting data used.

    Returns:    fits, dictionary
                    Dictionary of the fitting coefficients and covariances from scipy.optimize.curve_fit.
                
    """
    terms = list(orders.keys())
    assert 'martensite_start' in terms, "Need to pass the Ms temperatures for fitting the non-chemistry-dependent terms...try again."
    fits = {}
    
    baseline_DG = data.drop(data[data['alloy_system'] != 'Fe-'].index)['DG']
    fits['martensite_start'] = [[np.mean(baseline_DG)]]
    data['excess_DG'] = data['DG'] - (fits['martensite_start'][0][0])
    ternary_systems = []

    for term in terms:
        if len(term.split('-')) == 2: # ternary interaction
            ternary_systems.append(term)
            continue
        if term == 'martensite_start':
            continue
        order = orders[term]
        system = f'Fe-{term}'
        sub_data = data.query("alloy_system == @system")
        if len(term.split('-')) == 1:  # binary interaction
            if order == 0:
                fits[term] = curve_fit(lambda x, C_0: C_0*x, sub_data[term], sub_data['excess_DG'])
                data['excess_DG'] = data['excess_DG'] - data[term]*fits[term][0][0]
            if order == 1:
                fits[term] = curve_fit(lambda x, C_0, C_1: x*(C_0 + C_1*(1-x)), sub_data[term], sub_data['excess_DG'])
                data['excess_DG'] = data['excess_DG'] - (data[term]*(fits[term][0][0] + fits[term][0][1]*(1-data[term])))
            if order == 2:
                fits[term] = curve_fit(lambda x, C_0, C_1, C_2: x*(C_0 + C_1*(1-x) + C_2*(1-x)**2), sub_data[term], sub_data['excess_DG'])
                data['excess_DG'] = data['excess_DG'] - (data[term]*(fits[term][0][0] + fits[term][0][1]*(1-data[term]) + fits[term][0][2]*(1-data[term])**2))

    for term in ternary_systems:
        order = orders[term]
        system = f'Fe-{term}'
        print(system)
        sub_data = data.query("alloy_system == @system")
        splits = term.split('-')
        if order == 0:
            fits[term] = curve_fit(lambda x, C_0: C_0*x[0]*x[1], np.vstack([sub_data[splits[0]], sub_data[splits[1]]]), sub_data['excess_DG'])
            data['excess_DG'] = data['excess_DG'] - (data[splits[0]]*data[splits[1]]*fits[term][0][0])
    return fits


def get_dG(row, db, cond):
    """
    Get the Gibbs free energy difference between appropriate phases for the martensite start model.

    Parameters: row, numpy.Series
                    Row from a pandas.DataFrame.
                db, pycalphad.Database
                    Database to use for the Gibbs free energy calculations.
                cond, dictionary
                    Dictionary of pycalphad.variable variablesq for equilibrium conditions.
    Returns:    energy, float
                    The calculated Gibbs free energy difference, in J/mol.
    """
    local_comp = ['FE', 'VA']
    undesired_comps = ['Fe', 'S', 'O']
    other_comps = [c.upper() for c in row['composition'].as_dict().keys() if c not in undesired_comps]
    local_comp = np.concatenate([local_comp, other_comps])
    for comp in local_comp:
        if comp == 'FE' or comp == 'VA':
            continue
        if row.loc['composition'][comp.capitalize()] == 0.00:  # These values result in NaNs for the GM during equilibrium calcs for some reason
            local_comp.remove(comp)
        else:
            cond.update({v.X(comp): row.loc['composition'][comp.capitalize()]})
    if 'martensite_start' in row.index:
        cond.update({v.T: row.loc['martensite_start']})
    elif 'austenite_start' in row.index:
        cond.update({v.T: row.loc['austenite_start']})
    else:
        return False
    i = 0
    while len(cond) > 14:  # pycalphad seems to crash with too many conditions
        min_element = list(cond.keys())[list(cond.values()).index(sorted(cond.values())[i])]
        if str(min_element).split('_')[1] not in ['C', 'N', 'NB', 'V']:  # don't want to remove these elements
            cond.pop(min_element)
            print(f"Removing {min_element}")
        i += 1

    comps = ['FE', 'VA']
    for i in list(cond.keys()):
        if str(i).startswith('X_'):
            comps.append(str(i).split('_')[1])

    if row['type'] == 'epsilon':  # select applicable phases.
        phases = ['FCC_A1', 'HCP_A3']
    else:
        phases = ['FCC_A1', 'BCC_A2']

    print(cond, phases)
    energy = utilities.DG(db, comps, phases, cond)
    print(energy)
    return energy