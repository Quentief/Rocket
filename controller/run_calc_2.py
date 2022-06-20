import openmdao.api as om
import dymos as dm

from controller.rocket_phases.expulsion.expulsion_phase import expulsion_phase_fn
from model.nox_prop_calculator import NOXProp



def rocket_trajectory(pamb: float):

    transcript = dm.Radau(num_segments=3)
    traj = dm.Trajectory()

    # Add phases to trajectory
    expulsion_phase = traj.add_phase('expulsion', expulsion_phase_fn(transcription=transcript, pamb=pamb))

    return traj, expulsion_phase


def launch_compt():

    # Set ambiant conditions
    Tamb = 20 + 273.15
    pamb = 100*10**3
    deltap = 0
    Vb = 5*10**-3
    Aout = 10*10**-4

    # Set NOX bottle properties up
    nox_prop = NOXProp()
    prop_t = nox_prop.find_from_t(temp=Tamb)
    Vl = nox_prop.find_Vl(m=20/2.205, psat=prop_t["psat"], Vb=Vb, rhol=prop_t["rhol"], Tb=Tamb)
    bottle_params = {"Vb": Vb, "gamma": nox_prop.gamma, "Aout": Aout, "rhol": prop_t["rhol"], "pout": pamb,
                     "deltap": deltap, "psat": prop_t["psat"], "Vl": Vl}

    # Instantiate an OpenMDAO Problem instance
    prob = om.Problem(model=om.Group())
    prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')

    # Instantiate a Dymos trjectory and add it to the Problem model
    traj, phase = rocket_trajectory(pamb= 100*10*3)
    phase.add_objective("time", loc="final")

    # Setup the OpenMDAO problem
    prob.model.add_subsystem("traj", traj)
    prob.setup()

    # Assign values to the times and states
    prob.set_val('traj.expulsion.t_initial', 0.0)
    prob.set_val('traj.expulsion.t_duration', 200.0)

    prob.set_val('traj.expulsion.states:p', bottle_params["psat"])
    prob.set_val('traj.expulsion.states:Vl', bottle_params["Vl"])

    prob.set_val('traj.expulsion.parameters:Vb', bottle_params["Vb"])
    prob.set_val('traj.expulsion.parameters:gamma', bottle_params["gamma"])
    prob.set_val('traj.expulsion.parameters:rhol', bottle_params["rhol"])
    prob.set_val('traj.expulsion.parameters:Aout', bottle_params["Aout"])
    prob.set_val('traj.expulsion.parameters:pout', bottle_params["pout"])
    prob.set_val('traj.expulsion.parameters:deltap', bottle_params["deltap"])

    # Perform a single execution of the model (executing the model is required before simulation).
    prob.run_driver()

    # # Plot the state values obtained from the phase timeseries objects in the simulation output.
    # t_sol = prob.get_val('traj.expulsion.timeseries.time')
    #
    # states = ["p", "Vl"]
    # units = {"p": "(PSI)", "Vl_dot": "m³/s"}
    # fig, axes = plt.subplots(2, 1)
    # for i , state in enumerate(states):
    #     axes[i].plot(t_sol, prob.get_val(f'traj.expulsion.timeseries.states:{state}')/6895, 'o')
    #     axes[i].set_ylabel(state + " " + units[state])
    # axes[-1].set_xlabel('time (s)')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    launch_compt()