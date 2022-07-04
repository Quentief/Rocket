import openmdao.api as om
import numpy as np


class NOXBottle(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('p', shape=(nn,), desc='Pressure inside the nox bottle', units='Pa')
        self.add_input('pout', shape=(nn,), desc='Pressure outside the nox_bottle', units='Pa')
        self.add_input('deltap', shape=(nn,), desc='Nox bottle pressure losses', units='Pa')

        self.add_input('Vb', shape=(nn,), desc='Bottle volume', units='m**3')
        self.add_input('Vl', shape=(nn,), desc='Liquid volume', units='m**3')

        self.add_input('gamma', shape=(nn,), desc='Heat capacity ratio')

        self.add_input('rhol', shape=(nn,), desc='Liquid density', units='kg/m**3')
        self.add_input('Aout', shape=(nn,), desc='Output nox_bottle area', units='m**2')

        # Outputs
        self.add_output('Vl_dot', shape=(nn,), desc='Volume flow rate', units='m**3/s')
        self.add_output('p_dot', shape=(nn,), desc='Pressure change rate', units='Pa/s')

        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt=['*'], method='cs')

    def compute(self, inputs, outputs):
        p = inputs['p']
        pout = inputs['pout']
        deltap = inputs['deltap']
        rhol = inputs['rhol']
        Aout = inputs['Aout']

        Vb = inputs['Vb']
        Vl = inputs['Vl']
        gamma = inputs['gamma']

        outputs['Vl_dot'] = Vl_dot = -Aout * np.sqrt(2 / rhol * (p - pout - deltap))
        outputs['p_dot'] = gamma * p / (Vb - Vl) * Vl_dot

    # def compute_partials(self, inputs, partials):
    #     p = inputs['p']
    #     pout = inputs['pout']
    #     deltap = inputs['deltap']
    #     rhol = inputs['rhol']
    #     Aout = inputs['Aout']
    #
    #     pressure_diff = p - pout - deltap
    #     dVldot_on_dp = Aout/np.sqrt(2*rhol*pressure_diff)
    #
    #     partials['Vl_dot', 'p'] = dVldot_on_dp
    #     partials['Vl_dot', 'pout'] = -dVldot_on_dp
    #     partials['Vl_dot', 'deltap'] = -dVldot_on_dp
    #     partials['Vl_dot', 'rhol'] = -Aout/rhol**(3/2)*np.sqrt(pressure_diff/2)