import openmdao.api as om


class PressureRate(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('p', shape=(nn,), desc='Pressure inside the nox bottle', units='Pa')
        self.add_input('Vb', shape=(nn,), desc='Bottle volume', units='m**3')
        self.add_input('Vl', shape=(nn,), desc='Liquid volume', units='m**3')
        self.add_input('Vl_dot', shape=(nn,), desc='Liquid volume flow rate', units='m**3/s')
        self.add_input('gamma', shape=(nn,), desc='Heat capacity ratio')

        # Outputs
        self.add_output('p_dot', shape=(nn,), desc='Pressure change rate', units='Pa/s')

        # Derivative
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        p = inputs['p']
        Vb = inputs['Vb']
        Vl = inputs['Vl']
        Vl_dot = inputs['Vl_dot']
        gamma = inputs['gamma']

        outputs['p_dot'] = gamma * p/(Vb - Vl) * Vl_dot