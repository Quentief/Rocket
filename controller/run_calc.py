import openmdao.api as om
import dymos as dm



from controller.rocket_phases.trajectories import set_trajectories
from model.NOX_data.nox_prop_finder import NOXProp




def launch_compt():

    # Set constants and initial parameters
    Tamb = 20 + 273.15
    pamb = 100*10**3
    deltap = 0
    Vb = 15*10**-3
    Aout = 10*10**-4
    mNOX = 20/2.205

    # Set NOX bottle properties up
    nox_prop = NOXProp()
    prop_t = nox_prop.find_from_t(temp=Tamb)
    Vl = nox_prop.find_Vl(m=mNOX, Vb=Vb, rhol=prop_t["rhol"], rhog=prop_t["rhog"])
    bottle_init = {"Vb": Vb, "gamma": nox_prop.gamma, "Aout": Aout, "rhol": prop_t["rhol"], "pout": pamb,
                   "deltap": deltap, "psat": prop_t["psat"], "Vl": Vl}

    # Instantiate an OpenMDAO Problem instance
    prob = om.Problem(model=om.Group())
    prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', maxiter=2000)
    prob.driver.declare_coloring(tol=1e-20)

    # Instantiate a Dymos trajectory and add it to the Problem model
    prob, traj = set_trajectories(bottle_params=bottle_init, prob=prob)

    # Run the simulation
    dm.run_problem(prob, run_driver=True, simulate=True, solution_record_file='model/dymos_data/dymos_solution.db',
                   simulation_record_file='model/dymos_data/dymos_simulation.db',
                   make_plots=True, plot_dir="view/output_plots")