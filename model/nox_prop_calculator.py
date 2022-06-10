import numpy as np
import pandas as pd



class NOXProp():

    def __init__(self):
        self.nox_properties = pd.read_excel("../model/N2O_Properties.xlsx", sheet_name="Courbe de saturation N2O")
        ### os.listdir() to check where is the current folder location
        self.gamma = 1.303
        self.MM = 30.01
        self.R = 8.314

    def find_from_t(self, temp: float):
        if temp <= self.nox_properties["t\n[K]"][-1] and temp >= self.nox_properties["t\n[K]"][0]:
            return {
                "psat": np.interp([temp], xp=self.nox_properties["t\n[K]"], fp=self.nox_properties["p\n[Pa]"]),
                "rhol": np.interp([temp], xp=self.nox_properties["t\n[K]"], fp=self.nox_properties["rho l\n[kg/m3]"]),
                "rhog": np.interp([temp], xp=self.nox_properties["t\n[K]"], fp=self.nox_properties["rho g\n[kg/m3]"])
            }
        else:
            return {"psat": np.nan, "rhol": np.nan, "rhog": np.nan}

    def find_from_p(self, p: float):
        if p <= self.nox_properties["p\n[Pa]"][-1] and p >= self.nox_properties["p\n[Pa]"][0]:
            return {
                "psat": np.interp([p], xp=self.nox_properties["p\n[Pa]"], fp=self.nox_properties["t\n[K]"]),
                "rhol": np.interp([p], xp=self.nox_properties["p\n[Pa]"], fp=self.nox_properties["rho l\n[kg/m3]"]),
                "rhog": np.interp([p], xp=self.nox_properties["p\n[Pa]"], fp=self.nox_properties["rho g\n[kg/m3]"])
            }
        else:
            return {"psat": np.nan, "rhol": np.nan, "rhog": np.nan}

    def find_Vl(self, m: float, psat: float, Vb: float, rhol: float, Tb: float):
        return (m - Vb*rhol)/(psat*self.MM/self.R/Tb - rhol)