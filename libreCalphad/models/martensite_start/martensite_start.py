"""
Functions implementing the martensite-start temperature model developed by Matt Frichtl. [TODO: Update with reference after publication]
Examples of its use are contained in martensite_start.ipynb.
"""

from collections import defaultdict
import json
from libreCalphad.models.utilities import get_components_from_conditions, DG
from libreCalphad.models.martensite_start.calc_martensite_start import get_model
import numpy as np
import pandas as pd
import pycalphad.variables as v
from scipy.optimize import Bounds, minimize


def get_plate_model_storm(conditions):
    """
    Get the plate martensite energy barrier from Stormvinter2012. See [1] for more information.

    [1]: A. Stormvinter, A. Borgenstam, and J. Ågren, “Thermodynamically Based Prediction of the Martensite Start Temperature for Commercial Steels,” 
            Metallurgical and Materials Transactions A, vol. 43, no. 10, pp. 3870–3879, Jun. 2012, doi: 10.1007/s11661-012-1171-z.

    Parameters: conditions : dictionary
                    Dictionary of conditions of pycalphad variables used in the model.

    Returns:    plate_barrier : float
                    The predicted Gibbs energy barrier for plate martensite formation provided by the Stormvinter2012 model.
    """     

    components = get_components_from_conditions(conditions, dependent_component='FE')
    comp_dict = defaultdict(lambda: 0)
    for component in components:
        if v.X(component.upper()) in conditions.keys():
            comp_dict.update({component.upper(): conditions[v.X(component.upper())]})

    plate_barrier = 2100 + 75000*comp_dict['C']**2/(1-comp_dict['C']) - 11500*comp_dict['C'] - 2970*comp_dict['CR'] \
        + 3574*comp_dict['MN'] - 5104*comp_dict['NI'] + 441700*comp_dict['C']*comp_dict['CR']/(1-comp_dict['C'])
    
    return plate_barrier


def get_lath_model_storm(conditions):
    """
    Get the lath martensite energy barrier from Stormvinter2012. See [1] for more information.

    [1]: A. Stormvinter, A. Borgenstam, and J. Ågren, “Thermodynamically Based Prediction of the Martensite Start Temperature for Commercial Steels,” 
            Metallurgical and Materials Transactions A, vol. 43, no. 10, pp. 3870–3879, Jun. 2012, doi: 10.1007/s11661-012-1171-z.

    Parameters: conditions : dictionary
                    Dictionary of conditions of pycalphad variables used in the model.

    Returns:    lath_barrier : float
                    The predicted Gibbs energy barrier for lath martensite formation provided by the Stormvinter2012 model.
    """

    components = get_components_from_conditions(conditions, dependent_component='FE')
    comp_dict = defaultdict(lambda: 0)
    for component in components:
        if v.X(component) in conditions.keys():
            comp_dict.update({component: conditions[v.X(component)]})
    T = conditions[v.T]

    lath_barrier = 3640 - 2.92*T +346400*comp_dict['C']**2/(1-comp_dict['C']) -16430*comp_dict['C'] - 785.5*comp_dict['CR'] \
        + 7119*comp_dict['MN'] - 4306*comp_dict['NI'] + 350600*comp_dict['C']*comp_dict['CR']/(1-comp_dict['C'])

    return lath_barrier


def get_lath_model_mf(conditions, pags=np.nan):
    """
    Get the current lath martensite energy barrier from the mf-lath model. Reference pending.

    Parameters: conditions : dictionary
                    Dictionary of conditions of pycalphad variables used in the model.

                pags : float
                    The prior-austenite grain size in microns.

    Returns: energy_barrier : float
                The predicted Gibbs energy barrier for lath martensite formation provided by the mf-lath model in J/mol.
                
             contrib_dict : dictionary
                Dictionary of the predicted contributions of each modeled subsystem of the alloy, in J/mol.
    """

    fits = pd.read_json('./model_params/mf_lath_parameters.json')['mf_lath_fits'][0]
    components = get_components_from_conditions(conditions, dependent_component='FE')
    energy_barrier, contrib_dict = get_model(components, conditions, fits, 'lath-mf', pags)
    return energy_barrier, contrib_dict


def get_plate_model_mf(conditions, pags=np.nan):
    """
    Get the current plate martensite energy barrier from the mf-plate model. Reference pending.

    Parameters: conditions : dictionary
                    Dictionary of conditions of pycalphad variables used in the model.

                pags : float
                    The prior-austenite grain size.

    Returns: energy_barrier : float, J/mol
                The predicted Gibbs energy barrier for lath martensite formation provided by the mf-lath model.

             contrib_dict : dictionary
                Dictionary of the predicted contributions of each modeled subsystem of the alloy, in J/mol.
    """

    fits = pd.read_json('./model_params/mf_plate_parameters.json')['mf_plate_fits'][0]
    components = get_components_from_conditions(conditions, dependent_component='FE')
    energy_barrier, contrib_dict = get_model(components, conditions, fits, 'plate-mf', pags)
    return energy_barrier, contrib_dict


def get_epsilon_model_mf(conditions, pags=np.nan):
    """
    Get the current epsilon martensite energy barrier from the mf-epsilon model. Reference pending.

    Parameters: conditions : dictionary
                    Dictionary of conditions of pycalphad variables used in the model.

                pags : float, defaults to NaN.
                    The prior-austenite grain size.

    Returns: energy_barrier : float, J/mol
                The predicted Gibbs energy barrier for lath martensite formation provided by the mf-lath model.

             contrib_dict : dictionary
                Dictionary of the predicted contributions of each modeled subsystem of the alloy, in J/mol.
    """

    fits = pd.read_json('./model_params/mf_epsilon_parameters.json')['mf_epsilon_fits'][0]
    components = get_components_from_conditions(conditions, dependent_component='FE')
    energy_barrier, contrib_dict = get_model(components, conditions, fits, 'epsilon-mf', pags)
    return energy_barrier, contrib_dict


def predict_martensite_type(conditions):
    """
    Uses the trained machine-learning model to predict the type of martensite that will form. Requires a temperature to be
    included in the conditions dictionary.

    Parameters: conditions : dictionary
                    Dictionary of conditions of pycalphad variables used in the model.
    
    Returns: types : numpy array
                An array of strings of the predicted martensite types.
    """

    gpc = pd.read_pickle('./gp_classifier_model.pkl').iloc[0]['gp_classifier']  # load the clasifier model
    feature_dict = {}
    df = pd.DataFrame()
    if type(conditions[v.T]) == int or type(conditions[v.T]) == float:  # only a single temperature
        for feature in gpc.feature_names_in_:
            if feature == 'martensite_start':
                feature_dict[feature] = conditions[v.T] / 1000  # need to transform the temperature to be a similar scale as the atomic fractions
            elif v.X(feature.capitalize()) in conditions:
                feature_dict[feature] = conditions[v.X(feature)]
            else:
                feature_dict[feature] = [0]
        df = pd.DataFrame.from_dict(feature_dict)
    elif type(conditions[v.T]) == tuple:  # assume pycalphad standard 3-term tuple
        temp_range = conditions[v.T]
        for temp in np.arange(start=temp_range[0], stop=temp_range[1], step=temp_range[2]):
            feature_dict = {}
            for feature in gpc.feature_names_in_:
                if feature == 'martensite_start':
                    feature_dict[feature] = temp / 1000  # need to transform the temperature to be a similar scale as the atomic fractions
                elif v.X(feature.capitalize()) in conditions:
                    feature_dict[feature] = conditions[v.X(feature)]
                else:
                    feature_dict[feature] = [0]
            if df.empty:
                df = pd.DataFrame.from_dict(feature_dict)
            else:
                df = pd.concat([df, pd.DataFrame.from_dict(feature_dict)])


    types = gpc.predict(df)
    labels = gpc.classes_
    probs = gpc.predict_proba(df)
    return types, labels, probs


def get_martensite_start(barrier, db, conditions, martensite_type=None, phases=None):
    """
    Function to predict the martensite start temperature of an alloy.
    """

    def delta_driving_force(T, db, components, phases, conditions, barrier):
        conditions.update({v.T: T})
        driving_force = -DG(db, components, phases, conditions)

        return np.abs(barrier - driving_force)
    
    assert ~all([martensite_type == None, phases == None]), "Must provide either a martensite type or the two phases for calculating driving force."

    cond = conditions.copy()
    for key, value in conditions.items():
        if cond[key] == 0:
            cond.pop(key)
    components = get_components_from_conditions(cond, dependent_component='FE')
    temp_bounds = Bounds(298.15, 1300)

    if martensite_type == 'lath' or martensite_type == 'plate':
        phases = ['BCC_A2', 'FCC_A1']
    elif martensite_type == 'epsilon':
        phases = ['HCP_A3', 'FCC_A1']
    else:
        raise NotImplementedError(f"Unknown martensite type passed, {martensite_type}. Must be 'lath', 'plate', or 'epsilon'.")

    min_res = minimize(delta_driving_force, 500, args=(db, components, phases, cond, barrier), method='Nelder-Mead', bounds=temp_bounds)

    return min_res.x[0]