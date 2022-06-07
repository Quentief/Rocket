import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt


from controller.nox_bottle import bottle_model



if __name__ == '__main__':

    # Instantiate an OpenMDAO Problem instance.
    prob = om.Problem()

    # We need an optimization driver.  To solve this simple problem ScipyOptimizerDriver will work.
    prob.driver = om.ScipyOptimizeDriver()

    # Instantiate a Dymos Trajectory and add it to the Problem model.
    traj = dm.Trajectory()
    prob.model.add_subsystem('traj', traj)

    # Instantiate a Phase and add it to the Trajectory.
    phase = dm.Phase(ode_class=bottle_model, transcription=dm.Radau(num_segments=10))
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

    prob.set_val('traj.phase0.states:p', 1820220.000154)
    prob.set_val('traj.phase0.states:Vl', 50*10**-6)

    prob.set_val('traj.phase0.parameters:gamma', 1.303)
    prob.set_val('traj.phase0.parameters:Vb', 10**-3)
    prob.set_val('traj.phase0.parameters:Aout', 2 * 10 ** -4)
    prob.set_val('traj.phase0.parameters:rhol', 1133.6)
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