import numpy as np
import pandas as pd



class LiquidNOX():
    def __init__(self, temp: float, **kwargs):
        """NOX temperature in K"""
        super().__init__(**kwargs)
        self.nox_properties = pd.read_excel("model/N2O_Properties.xlsx", sheet_name="Courbe de saturation N2O")
        self.temp = temp
        th_props = self.find_boiling_prop()
        self.pressure = th_props["pressure"]
        self.deltaH = th_props["deltaH"]
        self.rho = th_props["rho"]

    def find_boiling_prop(self):
        if self.temp <= self.nox_properties["t\n[K]"][-1] and self.temp >= self.nox_properties["t\n[K]"][0]:
            return {
                "pressure": np.interp([self.temp], xp=self.nox_properties["t\n[K]"],
                                  fp=self.nox_properties["p\n[Pa]"]),
                "deltaH": np.interp([self.pressure * 10 ** 5], xp=self.nox_properties["p\n[Pa]"],
                                  fp=self.nox_properties["vaph\n[J/kg]"]),
                "rho": np.interp([self.pressure * 10 ** 5], xp=self.nox_properties["p\n[Pa]"],
                                  fp=self.nox_properties["rho l\n[kg/m3]"])}
        else:
            return {"pressure": np.nan, "deltaH": np.nan, "rho": np.nan}

    # def find_liquid_volume(self):