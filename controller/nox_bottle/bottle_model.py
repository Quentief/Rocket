import openmdao.api as om

from controller.nox_bottle.pressure_rate_ode import PressureRateODE
from controller.nox_bottle.volume_flow_rate import VolumeFlowRate


class BottleModel(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='pressure_rate_bottle',
                           subsys=PressureRateODE(num_nodes=nn),
                           promotes=['p', 'Vb', "Vl", "Vl_dot", 'gamma'])

        self.add_subsystem(name='volume_flow_rate_output_bottle',
                           subsys=VolumeFlowRate(num_nodes=nn),
                           promotes=['p', 'pout', "deltap", "rhol", 'Aout'])

        self.connect('pressure_rate_bottle.p', 'volume_flow_rate_output_bottle.p')
        self.connect('pressure_rate_bottle.Vl_dot', 'volume_flow_rate_output_bottle.Vl_dot')

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=20)
        self.linear_solver = om.DirectSolver()