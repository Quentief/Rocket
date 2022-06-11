import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from dymos.utils.lgl import lgl


from controller.nox_bottle import bottle, pressure_rate_ode
from controller.nox_bottle.pressure_rate_ode import PressureRateODE
from controller.nox_bottle.volume_flow_rate import VolumeFlowRate
from model.nox_prop_calculator import NOXProp

if __name__ == '__main__':

    # Set NOX properties up
    nox_prop = NOXProp()
    temp = 273.15 + 20
    Vb = 0.005
    prop_t = nox_prop.find_from_t(temp=temp)
    Vl = nox_prop.find_Vl(m=20/2.205, psat=prop_t["psat"], Vb=Vb, rhol=prop_t["rhol"], Tb=temp)

    # Instantiate an OpenMDAO Problem instance.
    prob = om.Problem()

    # We need an optimization driver.  To solve this simple problem ScipyOptimizerDriver will work.
    prob.driver = om.ScipyOptimizeDriver()

    # Instantiate a Dymos Trajectory and add it to the Problem model.
    traj = dm.Trajectory()
    prob.model.add_subsystem('traj', traj)

    # Instantiate a Phase and add it to the Trajectory.
    phase = dm.Phase(ode_class=pressure_rate_ode, transcription=dm.Radau(num_segments=10))
    phase.add_subsystem(name='volume_flow_rate_output_bottle',
                       subsys=VolumeFlowRate(num_nodes=16),
                       promotes=['p', 'pout', "deltap", "rhol", 'Aout'])
    traj.add_phase('phase0', phase)

    # Tell Dymos that the duration of the phase is bounded.
    phase.set_time_options(fix_initial=True, fix_duration=True)

    # Tell Dymos the states to be propagated using the given ODE.
    phase.add_state('p', rate_source='p_dot', targets=['p'], units='Pa')
    phase.add_state('Vl', rate_source='Vl_dot', targets=['Vl'], units='m**3')

    # The spring constant, damping coefficient, and mass are inputs to the system
    # that are constant throughout the phase.
    phase.add_parameter('gamma', targets=['gamma'])
    phase.add_parameter('Vb', units='m**3', targets=['Vb'])
    phase.add_parameter('Aout', units='m**2', targets=['Aout'])
    phase.add_parameter('rhol', units='kg/m**3', targets=['rhol'])
    phase.add_parameter('pout', units='Pa', targets=['pout'])
    phase.add_parameter('deltap', units='Pa', targets=['deltap'])

    # Since we're using an optimization driver, an objective is required.  We'll minimize
    # the final time in this case.
    phase.add_objective('time', loc='final')

    # Setup the OpenMDAO problem
    prob.setup()

    # Assign values to the times and states
    prob.set_val('traj.phase0.t_initial', 0.0)
    prob.set_val('traj.phase0.t_duration', 200.0)

    prob.set_val('traj.phase0.states:p', prop_t["psat"])
    prob.set_val('traj.phase0.states:Vl', Vl)

    prob.set_val('traj.phase0.parameters:gamma', nox_prop.gamma)
    prob.set_val('traj.phase0.parameters:Vb', Vb)
    prob.set_val('traj.phase0.parameters:Aout', 2 * 10 ** -4)
    prob.set_val('traj.phase0.parameters:rhol', prop_t["rhol"])
    prob.set_val('traj.phase0.parameters:pout', 100*10**3)
    prob.set_val('traj.phase0.parameters:deltap', 0)

    # Perform a single execution of the model (executing the model is required before simulation).
    prob.run_driver()

    # Perform an explicit simulation of our ODE from the initial conditions.
    sim_out = traj.simulate(times_per_seg=50)

    # Plot the state values obtained from the phase timeseries objects in the simulation output.
    t_sol = prob.get_val('traj.phase0.timeseries.time')
    t_sim = sim_out.get_val('traj.phase0.timeseries.time')

    states = ["p"]
    units = {"p": "(PSI)", "Vl_dot": "m³/s"}
    fig, axes = plt.subplots(2, 1)
    for i, state in enumerate(states):
        sol = axes[i].plot(t_sol, prob.get_val(f'traj.phase0.timeseries.states:{state}')/6895, 'o')
        sim = axes[i].plot(t_sim, sim_out.get_val(f'traj.phase0.timeseries.states:{state}')/6895, '-')
        axes[i].set_ylabel(state + " " + units[state])
    axes[-1].set_xlabel('time (s)')
    fig.legend((sol[0], sim[0]), ('solution', 'simulation'), 'lower right', ncol=2)
    plt.tight_layout()
    plt.show()

    # prob = om.Problem()
    #
    # opt = prob.driver = om.ScipyOptimizeDriver()
    # opt.declare_coloring()
    # opt.options['optimizer'] = 'SLSQP'
    #
    # num_seg = 5
    # seg_ends, _ = lgl(num_seg + 1)
    #
    # traj = prob.model.add_subsystem('traj', dm.Trajectory())
    #
    # # First phase: normal operation.
    # transcription = dm.Radau(num_segments=num_seg, order=5, segment_ends=seg_ends, compressed=False)
    # phase0 = dm.Phase(ode_class=bottle_model, transcription=transcription)
    # traj_p0 = traj.add_phase('phase0', phase0)