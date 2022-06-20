import openmdao.api as om
import dymos as dm

from matplotlib import pyplot as plt

from controller.rocket_phases.trajectories import set_trajectories
from model.nox_prop_calculator import NOXProp




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
    bottle_init = {"Vb": Vb, "gamma": nox_prop.gamma, "Aout": Aout, "rhol": prop_t["rhol"], "pout": pamb,
                   "deltap": deltap, "psat": prop_t["psat"], "Vl": Vl}

    # Instantiate an OpenMDAO Problem instance
    prob = om.Problem(model=om.Group())
    prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', maxiter=1000)

    # Instantiate a Dymos trajectory and add it to the Problem model
    prob, traj = set_trajectories(bottle_params=bottle_init, prob=prob)

    # Finish Problem Setup
    prob.run_driver()
    # dm.run_problem(prob, run_driver=True, simulate=True)

    # Perform an explicit simulation of our ODE from the initial conditions.
    # sim_out = traj.simulate(times_per_seg=50)
    # Plot the state values obtained from the phase timeseries objects in the simulation output.
    # t_sol = prob.get_val('traj.expulsion.timeseries.time')
    # t_sim = sim_out.get_val('traj.expulsion.timeseries.time')

    # states = ["p"]
    # units = {"p": "(PSI)", "Vl_dot": "m³/s"}
    # fig, axes = plt.subplots(2, 1)
    # for i , state in enumerate(states):
    #     sol = axes[i].plot(t_sol, prob.get_val(f'traj.expulsion.timeseries.states:{state}')/6895, 'o')
    #     sim = axes[i].plot(t_sim, sim_out.get_val(f'traj.expulsion.timeseries.states:{state}')/6895, '-')
    #     axes[i].set_ylabel(state + " " + units[state])
    # axes[-1].set_xlabel('time (s)')
    # fig.legend((sol[0], sim[0]), ('solution', 'simulation'), 'lower right', ncol=2)
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    launch_compt()