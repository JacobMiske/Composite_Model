# ME 5661 Composite Materials
# Jacob Miske
# For composite modeling

import numpy as np

class Composite():

    def __init__(self):
        self.PR = 0.1   # no dimension
        self.Ef = 10    # MPa
        self.Em = 1    # MPa
        self.Gf = 10   # MPa
        self.Gm = 10   # MPa
        self.Vf = 0.5   # Volume fraction of fibers
        self.Vm = 0.5   # Volume fraction of fibers
        self.theta = [0, -1, 1]  # Angle of fibers in each layer in radians
        self.layers = len(self.theta)
        self.E1 = 1
        self.E2 = 1
        self.G12 = 1
        self.Ex = [1,1,1]
        self.Ey = [1,1,1]
        self.G12 = [1,1,1]

    def set_composite_longitudinal_modulus(self):
        """
        Given a composite, return the longitudinal modulus
        :return:
        """
        self.E1 = self.Vf * self.Ef + self.Vm * self.Em
        return 0


    def set_composite_transverse_modulus(self):
        """
        Given a composite, return the transverse modulus
        :return:
        """
        self.E2 = (self.Em * self.Ef)/(self.Vf * self.Em + self.Vf * self.Ef)


    def set_composite_shear_modulus(self):
        """
        Given a composite, return the shear modulus
        :return:
        """
        self.G12 = (self.Gm * self.Gf)/(self.Vf * self.Gm + self.Vf * self.Gf)


    def set_composite_modulus_by_layer(self):
        """
        Given a composite, return the modulus at angle
        :return:
        """
        comp_mod_by_layer = []
        for i, count in enumerate(self.theta, 0):
            comp_mod_by_layer.append( (self.E1 * self.E2)/
                                    (self.E1 * np.sin(self.theta[i])**4 + 2*self.G12 * np.sin(self.theta[i])**2 * np.cos(self.theta[i])**2 + self.E2 * np.cos(self.theta[i])**4) )
        self.Ex = comp_mod_by_layer


def get_data_from_model():
    """
    Given a model, get stress strain response plot from 0 to 50% strain
    """
    return 0

def get_data_from_test():
    """
    Given a experiment, get stress strain response plot from 0 to 50% strain
    """
    return 0

def plot_model_against_test():
    """
    Plot data from model and test
    """
    return 0


if __name__ == '__main__':
    # Input material properties
    # First VTP cube
    VTP1 = Composite()
    VTP1.Em = 1.1 # MPa
    VTP1.Ef = 10  # MPa
    VTP1.Gm = 0.8 # MPa
    VTP1.Gf = 1.5 # MPa
    VTP1.Vf = 0.5 
    VTP1.Vm - 0.5
    VTP1.set_composite_longitudinal_modulus()
    VTP1.set_composite_transverse_modulus()
    VTP1.set_composite_shear_modulus()
    VTP1.set_composite_modulus_by_layer()
    print(VTP1.Ex)

    # Second VTP cube
    VTP2 = Composite()
    VTP2.Em = 1.1 # MPa
    VTP2.Ef = 10  # MPa
    VTP2.Gm = 0.8 # MPa
    VTP2.Gf = 1.5 # MPa
    VTP2.Vf = 0.8 # nd
    VTP2.Vm = 0.2 # nd

    VTP2.set_composite_longitudinal_modulus()
    VTP2.set_composite_transverse_modulus()
    VTP2.set_composite_shear_modulus()
    VTP2.set_composite_modulus_by_layer()
