# ME 5661 Composite Materials
# Jacob Miske
# For composite modeling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def get_data_from_test(file_path):
    """
    Given a experiment, get stress strain response plot from 0 to 50% strain
    """
    # Read the CSV and skip the first two rows
    df = pd.read_csv(file_path, skiprows=2)
    # Check the data
    print(df.head())
    # Convert columns to plottable data types if not already
    col1 = df.iloc[:, 0].astype(float)  # First column as floats
    col2 = df.iloc[:, 1].astype(float)  # Second column as floats
    col3 = df.iloc[:, 2].astype(float)  # Third column as floats
    # Take every 515th row and save to a list
    sampled_rows = df.iloc[::515].astype(float)
    print("Sampled rows:", sampled_rows)
    return col1, col2, col3, sampled_rows


def get_stress_strain_from_data(displacement, force, area, start_length):
    """
    Given force, area, displacement, and L0
    return stress and strain data
    """
    epsilon = [i/start_length for i in displacement]
    sigma = [j/area for j in force]
    return epsilon, sigma


def plot_model_data(col2, col3):
    """
    Plot model data
    """
    plt.figure(figsize=(8, 6))
    print(type(col2))
    plt.plot(col2, col3, label="At 100 Kilocycles Sawzall")
    plt.legend()
    plt.grid()
    plt.xlabel('Deformation [mm]')
    plt.ylabel('Force [kN]')
    plt.show()
    return 0


def plot_experimental_data(col1, col2, plot_title, file_name):
    """
    Plot experimental data
    """
    # run linear fit
    linear_coeffs = np.polyfit(col1[-1800:], col2[-1800:], 1)
    y_line = [i * linear_coeffs[0] + linear_coeffs[1] for i in col1]
    # Generate plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(col1, col2, label="Experimental Data")
    plt.plot(col1, y_line, label="linear fit", linestyle='dashed')
    plt.xlim([0, 0.1])
    plt.ylim([0, 800])
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, "Young's Mod: " + str(round(linear_coeffs[0], 2)) + " kPa", transform=ax.transAxes, fontsize=14,
        verticalalignment='top')
    plt.legend()
    plt.grid()
    plt.xlabel('Strain')
    plt.ylabel('Stress [kPa]')
    plt.title(plot_title)
    plt.savefig(file_name)
    plt.show()
    return 0


def plot_model_against_test():
    """
    Plot data from model and test
    """
    return 0


if __name__ == '__main__':
    # Input material properties
    # Solid VTP Cube
    VTP0 = Composite()
    VTP0.Em = 1.1 # MPa
    VTP0.Ef = 10  # MPa
    VTP0.Gm = 0.8 # MPa
    VTP0.Gf = 1.5 # MPa
    VTP0.Vf = 1
    VTP0.Vm = 0

    # First VTP cube
    VTP1 = Composite()
    VTP1.Em = 1.1 # MPa
    VTP1.Ef = 10  # MPa
    VTP1.Gm = 0.8 # MPa
    VTP1.Gf = 1.5 # MPa
    VTP1.Vf = 0.5 
    VTP1.Vm = 0.5
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

    # Get data from all three tests
    area0 = 0.0009 # square meters
    area1 = 0.0009 # square meters
    area2 = 0.0009 # square meters
    L0_0 = 0.03*1000 # millimeters
    L0_1 = 0.0290*1000 # millimeters
    L0_2 = 0.0295*1000 # millimeters
    
    time0, disp0, force0, rows0 = get_data_from_test(file_path="./solid_TPU_cube_1.csv")
    time1, disp1, force1, rows1 = get_data_from_test(file_path="./Vf5_cube_1.csv")
    time2, disp2, force2, rows2 = get_data_from_test(file_path="./Vf8_cube_1.csv")
    strain0, stress0 = get_stress_strain_from_data(displacement=list(disp0), force=list(force0), area=area0, start_length=L0_0)
    strain1, stress1 = get_stress_strain_from_data(displacement=list(disp1), force=list(force1), area=area1, start_length=L0_1)
    strain2, stress2 = get_stress_strain_from_data(displacement=list(disp2), force=list(force2), area=area2, start_length=L0_2)

    # Plot each test case on it's own
    plot_experimental_data(col1=strain0, col2=stress0, plot_title="Solid TPU Cube", file_name="./TRL_solid_VTP_cube_stress_strain.png")

    # Plot all tests together

    # Get model data for all three tests

    # Plot each model on it's own

    # Plot all models together

    # Plot data from experiments against model
