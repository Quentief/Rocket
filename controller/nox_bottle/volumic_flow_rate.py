import numpy as np
import openmdao.api as om


class VolumicFlowRate(om.ExplicitComponent):
    """
    A Dymos ODE for a damped harmonic oscillator.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('p', shape=(nn,), desc='Pressure inside the nox_bottle', units='Pa')
        self.add_input('pout', shape=(nn,), desc='Pressure outside the nox_bottle', units='Pa')
        self.add_input('rhol', shape=(nn,), desc='Liquid volumic mass', units='kg/m**3')
        self.add_input('Aout', shape=(nn,), desc='Output nox_bottle area', units='m²')

        # Outputs
        self.add_output('Vl_dot', val=np.zeros(nn), desc='Volumic mass flow rate', units='m**3/s')

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        p = inputs['p']
        pout = inputs['pout']
        rhol = inputs['rhol']
        Aout = inputs['Aout']

        outputs['p_dot'] = Aout*np.sqrt(2/rhol*(p - pout))