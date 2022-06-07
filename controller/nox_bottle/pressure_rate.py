import numpy as np
import openmdao.api as om


class PressureRateODE(om.ExplicitComponent):
    """
    A Dymos ODE for a damped harmonic oscillator.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('p', shape=(nn,), desc='Pressure inside the nox_bottle', units='Pa')
        self.add_input('Vb', shape=(nn,), desc='Bottle volume', units='m**3')
        self.add_input('Vl', shape=(nn,), desc='Liquid volume', units='m**3')
        self.add_input('Vl_dot', shape=(nn,), desc='Liquid volumic flow rate', units='m**3/s')
        self.add_input('gamma', shape=(nn,), desc='Heat capacity ratio')

        # Outputs
        self.add_output('p_dot', val=np.zeros(nn), desc='Pressure change rate', units='Pa/s')

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        p = inputs['p']
        Vb = inputs['Vb']
        Vl = inputs['Vl']
        Vl_dot = inputs['Vl_dot']
        gamma = inputs['gamma']

        outputs['p_dot'] = -2/(1 + 1/gamma) * p/(Vb - Vl) * Vl_dot