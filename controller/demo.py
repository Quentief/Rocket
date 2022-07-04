import openmdao
import openmdao.api as om
import dymos as dm
import numpy as np



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
        ar = np.arange(nn)
        self.declare_partials(of='*', wrt='*', rows=ar, cols=ar)
        # self.declare_partials(of='*', wrt='*', method='fd')
        # self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        p = inputs['p']
        Vb = inputs['Vb']
        Vl = inputs['Vl']
        Vl_dot = inputs['Vl_dot']
        gamma = inputs['gamma']

        outputs['p_dot'] = gamma * p/(Vb - Vl) * Vl_dot

    def compute_partials(self, inputs, partials):
        p = inputs['p']
        Vb = inputs['Vb']
        Vl = inputs['Vl']
        Vl_dot = inputs['Vl_dot']
        gamma = inputs['gamma']

        partials['p_dot', 'p'] = gamma/(Vb - Vl) * Vl_dot
        partials['p_dot', 'Vb'] = gamma * -p/(Vb - Vl)**2 * Vl_dot
        partials['p_dot', 'Vl'] = gamma * p/(Vb - Vl)**2 * Vl_dot
        partials['p_dot', 'Vl_dot'] = gamma * p/(Vb - Vl)
        partials['p_dot', 'gamma'] = p/(Vb - Vl) * Vl_dot



class VolumeFlowRate(om.ExplicitComponent):
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
        self.add_input('deltap', shape=(nn,), desc='Nox bottle pressure losses', units='Pa')
        self.add_input('rhol', shape=(nn,), desc='Liquid density', units='kg/m**3')
        self.add_input('Aout', shape=(nn,), desc='Output nox_bottle area', units='m**2')

        # Outputs
        self.add_output('Vl_dot', shape=(nn,), desc='Volume flow rate', units='m**3/s')

        # Derivative
        ar = np.arange(nn)
        self.declare_partials(of='*', wrt='*', rows=ar, cols=ar)
        # self.declare_partials(of='*', wrt='*', method='fd')
        # self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        p = inputs['p']
        pout = inputs['pout']
        deltap = inputs['deltap']
        rhol = inputs['rhol']
        Aout = inputs['Aout']

        outputs['Vl_dot'] = Aout*np.sqrt(2/rhol*(p - pout - deltap))

    def compute_partials(self, inputs, partials):
        p = inputs['p']
        pout = inputs['pout']
        deltap = inputs['deltap']
        rhol = inputs['rhol']
        Aout = inputs['Aout']

        pressure_diff = p - pout - deltap
        dVldot_on_dp = Aout/np.sqrt(2*rhol*pressure_diff)

        partials['Vl_dot', 'p'] = dVldot_on_dp
        partials['Vl_dot', 'pout'] = -dVldot_on_dp
        partials['Vl_dot', 'deltap'] = -dVldot_on_dp
        partials['Vl_dot', 'rhol'] = -Aout/rhol**(3/2)*np.sqrt(pressure_diff/2)
        partials['Vl_dot', 'Aout'] = np.sqrt(2*pressure_diff/rhol)


class BottleModelODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('volume_flow_rate', subsys=VolumeFlowRate(num_nodes=nn),
                           promotes_inputs=['p', "pout", 'deltap', 'rhol', "Aout"], promotes_outputs=['Vl_dot'])
        self.add_subsystem('pressure_rate', subsys=PressureRate(num_nodes=nn),
                           promotes_inputs=['p', "Vb", "Vl", "Vl_dot", "gamma"], promotes_outputs=['p_dot'])


def expulsion_phase_fn(transcription: dm.transcriptions.pseudospectral.radau_pseudospectral.Radau, pout: float):

    phase = dm.Phase(ode_class=BottleModelODE, transcription=transcription)

    phase.set_time_options(fix_initial=True, fix_duration=True)

    # Define the states variables
    phase.add_state('p', units='bar', rate_source='p_dot', targets=['p'], fix_initial=True, fix_final=False,
                    lower=pout)
    phase.add_state('Vl', units='m**3', rate_source='Vl_dot', targets=['Vl'], fix_initial=True, fix_final=False,
                    lower=0)

    # Define the parameters variables
    phase.add_parameter('Vb', targets=['Vb'], units='m**3')
    phase.add_parameter('gamma', targets=['gamma'])
    phase.add_parameter('rhol', targets=['rhol'], units='kg/m**3')
    phase.add_parameter('Aout', targets=['Aout'], units='m**2')
    phase.add_parameter('pout', targets=['pout'], units="Pa")
    phase.add_parameter('deltap', targets=['deltap'], units="Pa")

    return phase


def expulsion_set_params(prob: openmdao.core.problem.Problem, phases_dict: dict, params: dict):

    # Assign values to the times and states
    prob.set_val('traj.expulsion.t_initial', 0.0)
    prob.set_val('traj.expulsion.t_duration', 200.0)

    # Assign initial states values
    prob.set_val('traj.expulsion.states:p', phases_dict["expulsion"].interp("p", [params["psat"], params["pout"]]),
                 units="Pa")
    prob.set_val('traj.expulsion.states:Vl', phases_dict["expulsion"].interp('Vl', [params["Vl"], 0]), units="m**3")

    # Assign parameters values
    prob.set_val('traj.expulsion.parameters:Vb', params["Vb"])
    prob.set_val('traj.expulsion.parameters:gamma', params["gamma"])
    prob.set_val('traj.expulsion.parameters:rhol', params["rhol"])
    prob.set_val('traj.expulsion.parameters:Aout', params["Aout"])
    prob.set_val('traj.expulsion.parameters:pout', params["pout"])
    prob.set_val('traj.expulsion.parameters:deltap', params["deltap"])

    return prob


def set_trajectories(prob: openmdao.core.problem.Problem, bottle_params: dict):


    # Instantiate a Dymos Trajectory and add it to the Problem model.
    traj = dm.Trajectory()
    prob.model.add_subsystem("traj", traj)

    # Define the rocket expulsion phase and add it to the trajectory.
    expulsion_phase = traj.add_phase('expulsion', expulsion_phase_fn(transcription=dm.Radau(num_segments=50, order=3),
                                                                     pout=bottle_params["pout"]))
    expulsion_phase.add_objective("time", loc="final")

    # Add trajectory to OpenMDAO problem and set the variables values up
    prob.model.linear_solver = openmdao.api.DirectSolver()
    prob.setup()
    expulsion_set_params(prob=prob, phases_dict={"expulsion": expulsion_phase}, params=bottle_params)

    return prob, traj


def launch_compt():

    # Set constants and initial parameters
    Tamb = 20 + 273.15
    pamb = 100*10**3
    deltap = 0
    Vb = 15*10**-3
    Aout = 10*10**-4
    mNOX = 20/2.205

    # Set NOX bottle properties up
    # nox_prop = NOXProp()
    # prop_t = nox_prop.find_from_t(temp=Tamb)
    # Vl = nox_prop.find_Vl(m=mNOX, Vb=Vb, rhol=prop_t["rhol"], rhog=prop_t["rhog"])
    bottle_init = {'Vb': 0.015, 'gamma': 1.303, 'Aout': 0.001, 'rhol': 786.6, 'pout': 100000, 'deltap': 0,
                   'psat': 5060000.0, 'Vl': 0.01065838470100318}

    # Instantiate an OpenMDAO Problem instance
    prob = om.Problem(model=om.Group())
    prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', maxiter=2000)

    # Instantiate a Dymos trajectory and add it to the Problem model
    prob, traj = set_trajectories(bottle_params=bottle_init, prob=prob)

    # Finish Problem Setup
    prob.run_driver()


if __name__ == '__main__':
    launch_compt()