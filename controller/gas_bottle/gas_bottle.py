import openmdao.api as om
import numpy as np


class GasBottle(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
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

        # Derivatives
        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt=['*'], method='cs')
        # ar = np.arange(nn)
        # self.declare_partials(of='*', wrt='*', rows=ar, cols=ar)

        # self.iter_nb = 0

    def compute(self, inputs, outputs):

        # self.iter_nb += 1
        # if self.iter_nb == 10000:
        #     print(self.iter_nb)
        # print("iteration n°: " + str(self.iter_nb))

        p = inputs['p']
        pout = inputs['pout']
        deltap = inputs['deltap']
        rhol = inputs['rhol']
        Aout = inputs['Aout']

        Vb = inputs['Vb']
        Vl = inputs['Vl']
        gamma = inputs['gamma']

        pressure_diff = p - pout - deltap
        pressure_diff[pressure_diff < 0] = 0

        outputs['Vl_dot'] = Vl_dot = -Aout * np.sqrt(2 / rhol * pressure_diff)
        outputs['p_dot'] = gamma * p * Vl_dot / (Vb - Vl)

    # def compute_partials(self, inputs, partials):
    #     p = inputs['p']
    #     pout = inputs['pout']
    #     deltap = inputs['deltap']
    #     rhol = inputs['rhol']
    #     Aout = inputs['Aout']
    #     Vb = inputs['Vb']
    #     Vl = inputs['Vl']
    #     gamma = inputs['gamma']
    #
    #     pressure_diff = p - pout - deltap
    #     pressure_diff[pressure_diff < 0] = 0
    #
    #     one_to_deltaV = 1 / (Vb - Vl)
    #     one_to_deltaV[np.isinf(one_to_deltaV)] = 0
    #
    #     one_to_deltaV_to_2 = 1 / (Vb - Vl)**2
    #     one_to_deltaV_to_2[np.isinf(one_to_deltaV)] = 0
    #
    #     partials['Vl_dot', 'p'] = dVldp = -Aout * np.sqrt(1 / (2 * rhol * pressure_diff))
    #     partials['Vl_dot', 'pout'] = dVldpout = Aout * np.sqrt(1 / (2 * rhol * pressure_diff))
    #     partials['Vl_dot', 'deltap'] = dVlddeltap = Aout * np.sqrt(1 / (2 * rhol * pressure_diff))
    #     partials['Vl_dot', 'rhol'] = dVldrhol = -Aout * np.sqrt(pressure_diff / (2 * rhol))
    #     partials['Vl_dot', 'Aout'] = dVldAout = -np.sqrt(2 * pressure_diff / rhol)
    #     partials['Vl_dot', 'Vb'] = 0
    #     partials['Vl_dot', 'Vl'] = 0
    #     partials['Vl_dot', 'gamma'] = 0
    #
    #     Vl_dot = -Aout * np.sqrt(2 / rhol * (p - pout - deltap))
    #
    #     partials['p_dot', 'p'] = gamma * Vl_dot * one_to_deltaV + gamma * p * one_to_deltaV * dVldp
    #     partials['p_dot', 'pout'] = gamma * p * one_to_deltaV * dVldpout
    #     partials['p_dot', 'deltap'] = gamma * p * one_to_deltaV * dVlddeltap
    #     partials['p_dot', 'rhol'] = gamma * p * one_to_deltaV * dVldrhol
    #     partials['p_dot', 'Aout'] = gamma * p * one_to_deltaV * dVldAout
    #     partials['p_dot', 'Vb'] = -gamma * p * Vl_dot / (Vb - Vl)**2
    #     partials['p_dot', 'Vl'] = gamma * p * Vl_dot / (Vb - Vl)**2
    #     partials['p_dot', 'gamma'] = p * Vl_dot * one_to_deltaV