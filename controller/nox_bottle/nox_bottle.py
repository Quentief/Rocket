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

        # self.declare_partials(of='*', wrt='*', method='cs')
        # self.declare_coloring(wrt=['*'], method='cs')

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

    def compute_partials(self, inputs, partials):
        p = inputs['p']
        pout = inputs['pout']
        deltap = inputs['deltap']
        rhol = inputs['rhol']
        Aout = inputs['Aout']
        Vb = inputs['Vb']
        Vl = inputs['Vl']
        gamma = inputs['gamma']

        pressure_diff = p - pout - deltap
        Vl_dot = -Aout * np.sqrt(2/rhol * (p - pout - deltap))

        partials['Vl_dot', 'p'] = -Aout/(2*rhol * pressure_diff)
        partials['Vl_dot', 'pout'] = Aout/(2*rhol * pressure_diff)
        partials['Vl_dot', 'deltap'] = Aout/(2*rhol * pressure_diff)
        partials['Vl_dot', 'rhol'] = Aout*np.sqrt(pressure_diff/2)/rhol**(3/2)
        partials['Vl_dot', 'Aout'] = -np.sqrt(2*pressure_diff/rhol)
        partials['Vl_dot', 'Vb'] = 0
        partials['Vl_dot', 'Vl'] = 0
        partials['Vl_dot', 'gamma'] = 0

        partials['p_dot', 'p'] = gamma/(Vb - Vl) * (Vl_dot - p * Aout/np.sqrt(2*rhol * pressure_diff))
        partials['p_dot', 'pout'] = gamma/(Vb - Vl) * p * Aout/np.sqrt(2*rhol * pressure_diff)
        partials['p_dot', 'deltap'] = gamma/(Vb - Vl) * p * Aout/np.sqrt(2*rhol * pressure_diff)
        partials['p_dot', 'rhol'] = gamma/(Vb - Vl) * p * Aout * np.sqrt(pressure_diff/2)/rhol**(3/2)
        partials['p_dot', 'Aout'] = -gamma/(Vb - Vl) * p * np.sqrt(2*pressure_diff/rhol)
        partials['p_dot', 'Vb'] = -gamma * p * Vl_dot/(Vb - Vl)**2
        partials['p_dot', 'Vl'] = -gamma * p * Vl_dot/(Vb - Vl)**2
        partials['p_dot', 'gamma'] = p * Vl_dot/(Vb - Vl)**2