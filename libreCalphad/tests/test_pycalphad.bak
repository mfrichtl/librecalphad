from libreCalphad.databases.db_utils import load_database
from pycalphad import equilibrium, variables as v


def test_complex_equilibrium():
    # This equilibrium calc failes with pycalphad 0.11.0.
    # This may take a little time to calculate.
    # TODO: Open an issue with pycalphad team.
    dbf = load_database('mf-steel.tdb')
    disabled_phases = ['BCC_B2', 'BCC_4SL', 'FCC_L10', 'FCC2_L10', 'FCC_L12', 'GAS', 'HCP_L12', 'KAPPA_A1', 'KAPPA_E21', 'IONIC_LIQ', 'TAU2_ALFEMO_A2', 'LAVES2_C14']  # disable some phases for calculations
    phases = [phase for phase in list(dbf.phases) if phase not in disabled_phases]
    components = ['FE', 'VA', 'C', 'CR', 'MN', 'MO', 'N', 'NI', 'P', 'SI', 'V']
    conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X("C"): 0.0036663193, v.X("CR"): 0.11433036740000001, 
                  v.X("MN"): 0.0081155962, v.X("MO"): 0.0075158862, v.X("N"): 0.0086456066, v.X("NI"): 0.0001875639,
                  v.X("P"): 0.0003376512, v.X("SI"): 0.0043117059, v.X("V"): 0.0033496434}
    equilibrium(dbf, components, phases, conditions)