import numpy as np
import pandas as pd



class NOXProp():

    def __init__(self):
        self.nox_properties = pd.read_excel("../model/N2O_Properties.xlsx", sheet_name="Courbe de saturation N2O")
        self.temp_min = list(self.nox_properties["t\n[K]"])[0]
        self.temp_max = list(self.nox_properties["t\n[K]"])[-1]
        self.p_min = list(self.nox_properties["p\n[Pa]"])[0]
        self.p_max = list(self.nox_properties["p\n[Pa]"])[-1]
        ### os.listdir() to check where is the current folder location
        self.gamma = 1.303
        self.MM = 30.01*10**-3
        self.R = 8.314


    def find_from_t(self, temp: float):
        if temp >= self.temp_min and temp <= self.temp_max:
            return {
                "psat":  float(np.interp([temp], xp=self.nox_properties["t\n[K]"], fp=self.nox_properties["p\n[Pa]"])),
                "rhol":  float(np.interp([temp], xp=self.nox_properties["t\n[K]"],
                               fp=self.nox_properties["rho l\n[kg/m3]"])),
                "rhog": float(np.interp([temp], xp=self.nox_properties["t\n[K]"],
                                        fp=self.nox_properties["rho g\n[kg/m3]"]))
            }
        else:
            return {"psat": float(np.nan), "rhol": float(np.nan), "rhog": float(np.nan)}

    def find_from_p(self, p: float):
        if p >= self.p_min and p <= self.p_max:
            return {
                "Tsat": float(np.interp([p], xp=self.nox_properties["p\n[Pa]"], fp=self.nox_properties["t\n[K]"])),
                "rhol": float(np.interp([p], xp=self.nox_properties["p\n[Pa]"],
                                        fp=self.nox_properties["rho l\n[kg/m3]"])),
                "rhog": float(np.interp([p], xp=self.nox_properties["p\n[Pa]"],
                                        fp=self.nox_properties["rho g\n[kg/m3]"]))
            }
        else:
            return {"Tsat": float(np.nan), "rhol": float(np.nan), "rhog": float(np.nan)}

    def find_Vl(self, m: float, Vb: float, rhol: float, rhog: float):
        # return (m - Vb*rhol)/(psat*self.MM/self.R/Tb - rhol)
        Vl = (m - Vb * rhog) / (rhol - rhog)
        if Vl > Vb:
            raise Exception("The bottle volume is too little to contain the NOX mass!")
        else:
            return Vl