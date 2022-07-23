import openmdao.api as om
import numpy as np
 

class GasBottle(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('p', shape=(nn,), desc='Pressure inside the nox bottle', units='Pa')
        self.add_input('pout', shape=(nn,), desc='Pressure outside the gas_bottle', units='Pa')
        self.add_input('deltap', shape=(nn,), desc='Nox bottle pressure losses', units='Pa')

        self.add_input('Vb', shape=(nn,), desc='Bottle volume', units='m**3')
        self.add_input('Vl', shape=(nn,), desc='Liquid volume', units='m**3')

        self.add_input('gamma', shape=(nn,), desc='Heat capacity ratio')

        self.add_input('rhol', shape=(nn,), desc='Liquid density', units='kg/m**3')
        self.add_input('Aout', shape=(nn,), desc='Output gas_bottle area', units='m**2')

        # Outputs
        self.add_output('Vl_dot', shape=(nn,), desc='Volume flow rate', units='m**3/s')
        self.add_output('p_dot', shape=(nn,), desc='Pressure change rate', units='Pa/s')

        ar = np.arange(nn)
        self.declare_partials(of='*', wrt='*', rows=ar, cols=ar)

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

        with np.errstate(divide='ignore'):
            dVldp = -Aout * np.sqrt(1 / (2 * rhol * pressure_diff))
            dVldpout = Aout * np.sqrt(1 / (2 * rhol * pressure_diff))
            dVlddeltap = Aout * np.sqrt(1 / (2 * rhol * pressure_diff))
            dVldrhol = -Aout * np.sqrt(pressure_diff / (2 * rhol))
            dVldAout = -np.sqrt(2 * pressure_diff / rhol)

        for vector in [dVldp, dVldpout, dVlddeltap, dVldrhol, dVldAout]:
            vector[np.isinf(vector)] = 0

        partials['Vl_dot', 'p'] = dVldp
        partials['Vl_dot', 'pout'] = dVldpout
        partials['Vl_dot', 'deltap'] = dVlddeltap
        partials['Vl_dot', 'rhol'] = dVldrhol
        partials['Vl_dot', 'Aout'] = dVldAout
        partials['Vl_dot', 'Vb'] = 0
        partials['Vl_dot', 'Vl'] = 0
        partials['Vl_dot', 'gamma'] = 0

        Vl_dot = -Aout * np.sqrt(2 / rhol * (p - pout - deltap))

        partials['p_dot', 'p'] = gamma * Vl_dot/(Vb - Vl) + gamma * p/(Vb - Vl) * dVldp
        partials['p_dot', 'pout'] = gamma * p/(Vb - Vl) * dVldpout
        partials['p_dot', 'deltap'] = gamma * p/(Vb - Vl) * dVlddeltap
        partials['p_dot', 'rhol'] = gamma * p/(Vb - Vl) * dVldrhol
        partials['p_dot', 'Aout'] = gamma * p/(Vb - Vl) * dVldAout
        partials['p_dot', 'Vb'] = -gamma * p * Vl_dot/(Vb - Vl)**2
        partials['p_dot', 'Vl'] = gamma * p * Vl_dot/(Vb - Vl)**2
        partials['p_dot', 'gamma'] = p * Vl_dot/(Vb - Vl)