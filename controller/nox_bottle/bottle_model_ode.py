import openmdao.api as om

from controller.nox_bottle.pressure_rate import PressureRate
from controller.nox_bottle.volume_flow_rate import VolumeFlowRate


class BottleModelODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('volume_flow_rate', subsys=VolumeFlowRate(num_nodes=nn),
                           promotes_inputs=['p', "pout", 'deltap', 'rhol', "Aout"], promotes_outputs=['Vl_dot'])
        self.add_subsystem('pressure_rate', subsys=PressureRate(num_nodes=nn),
                           promotes_inputs=['p', "Vb", "Vl", "Vl_dot", "gamma"], promotes_outputs=['p_dot'])