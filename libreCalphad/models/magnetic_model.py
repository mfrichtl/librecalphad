# Custom model to return magnetic energy contribution for use in calculations

from pycalphad import Model, variables as v


class MagneticModel(Model):
    def build_phase(self, dbe):
        super(MagneticModel, self).build_phase(dbe)
        self.magnetic_energy = self.get_magnetic_energy_contribution(dbe)

    def get_magnetic_energy_contribution(self, dbe):
        return self.magnetic_energy(dbe)
