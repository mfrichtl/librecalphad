import numpy as np
from pycalphad import Database, Model, variables as v
import pycalphad.io.tdb_keywords
from symengine import S
from tinydb import where


class ShearModulusModel(Model):
    """
    This property model is an implementation of the Fe-based alloy shear modulus model by Ghosh and Olson [1]
    using pycalphad.

    References:
    [1] :   Ghosh, B. and Olson, G. B., "The isotropic shear modulus of multicomponent Fe-base solid solutions,"
            Acta Materiala, Vol. 50, No. 10, p. 2655-2675, 2002-06, doi: 10.1016/s1359-6454(02)00096-4.
    """
  
    def build_phase(self, dbe):
        super(ShearModulusModel, self).build_phase(dbe)
        self.modulus = self.build_modulus(dbe)

    def build_modulus(self, dbe):
        dmu_dx_dict = {'BCC_A2': {'AL': -10.085, 'BE': -12.890, 'C': -14.194, 'CO': 2.389, 'CR': 3.406, 'CU': -0.145,
                                  'GE': -15.622, 'IR': 1.202, 'MN': -2.263, 'MO': -1.671, 'N': -15.439, 'NB': -8.707,
                                  'NI': -9.065, 'PD': -7.315, 'PT': -7.315, 'RE': 7.345, 'RH': -5.234, 'RU': -0.388,
                                  'SI': -10.914, 'TI': -7.459, 'V': 1.014, 'W': 7.267,},
                       'FCC_A1': {'AL': -27.741, 'BE': -9.544, 'C': -10.889, 'CO': -2.024, 'CR': -0.738, 'CU': 10.423,
                                  'GE': -17.787, 'IR': 5.249, 'MN': -2.675, 'MO': 2.940, 'N': -12.313, 'NB': -9.241,
                                  'NI': -6.049, 'PD': -8.708, 'PT': -8.706, 'RE': 6.193, 'RH': 59.922, 'RU': 2.232,
                                  'SI': -14.914, 'TI': -11.169, 'V': 2.350, 'W': 8.196,},}
        dmu_dx_dict['BCT'] = dmu_dx_dict['BCC_A2']  # elemental coefficients for lath martensite same as ferrite

        phase = dbe.phases[self.phase_name]
        shear_modulus_dmu_dx = 0
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self.components)
            ratio = self.site_ratios[subl_index]
            for comp in active_comps:
                sitefrac = v.SiteFraction(phase.name, subl_index, comp)
                if str(comp) in dmu_dx_dict[phase.name].keys():
                    shear_modulus_dmu_dx += sitefrac*ratio*dmu_dx_dict[phase.name][str(comp)]
                    # shear_modulus_dmu_dx += dmu_dx_dict[phase.name][str(comp)]*sitefrac*comp.number_of_atoms
        # shear_modulus_dmu_dx = shear_modulus_dmu_dx / self._site_ratio_normalizationx
        # print(shear_modulus_dmu_dx)
        modulus = None
        if phase.name == 'BCC_A2':  # ferrite
            modulus = (8.407 + shear_modulus_dmu_dx) * (1 - 0.48797*(v.T/self.TC)**2 + 0.12651*(v.T/self.TC)**3)
        elif phase.name == 'BCT':  # lath martensite
            modulus = (8.068 + shear_modulus_dmu_dx) * (1 - 0.48797*(v.T/self.TC)**2 + 0.12651*(v.T/self.TC)**3)
        elif phase.name == 'FCC_A1':
            modulus = (9.2648 + shear_modulus_dmu_dx) * (1 - 7.9921E-7*v.T**2 + 3.317E-10*v.T**3)

        assert modulus != None, "Shear modululs model does not include the selected phase."
        modulus = self.shear_modulus = modulus * 1E10  # convert to Pa

        return self.shear_modulus
