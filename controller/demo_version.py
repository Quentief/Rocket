import openmdao.api as om
import dymos as dm
from dymos.examples.water_rocket.phases import new_water_rocket_trajectory, set_sane_initial_guesses

def set_sane_initial_guesses(problem, phases):
    p = problem
    # Set Initial Guesses
    p.set_val('traj.propelled_ascent.t_initial', 0.0)
    p.set_val('traj.propelled_ascent.t_duration', 0.3)

    p.set_val('traj.propelled_ascent.states:r',
              phases['propelled_ascent'].interp('r', [0, 3]))
    p.set_val('traj.propelled_ascent.states:h',
              phases['propelled_ascent'].interp('h', [0, 10]))
    # Set initial value for velocity as non-zero to avoid undefined EOM
    p.set_val('traj.propelled_ascent.states:v',
              phases['propelled_ascent'].interp('v', [0.1, 100]))
    p.set_val('traj.propelled_ascent.states:gam',
              phases['propelled_ascent'].interp('gam', [80, 80]),
              units='deg')
    p.set_val('traj.propelled_ascent.states:V_w',
              phases['propelled_ascent'].interp('V_w', [9, 0]),
              units='L')
    p.set_val('traj.propelled_ascent.states:p',
              phases['propelled_ascent'].interp('p', [6.5, 3.5]),
              units='bar')

    p.set_val('traj.ballistic_ascent.t_initial', 0.3)
    p.set_val('traj.ballistic_ascent.t_duration', 5)

    p.set_val('traj.ballistic_ascent.states:r',
              phases['ballistic_ascent'].interp('r', [0, 10]))
    p.set_val('traj.ballistic_ascent.states:h',
              phases['ballistic_ascent'].interp('h', [10, 100]))
    p.set_val('traj.ballistic_ascent.states:v',
              phases['ballistic_ascent'].interp('v', [60, 20]))
    p.set_val('traj.ballistic_ascent.states:gam',
              phases['ballistic_ascent'].interp('gam', [80, 0]),
              units='deg')

    p.set_val('traj.descent.t_initial', 10.0)
    p.set_val('traj.descent.t_duration', 10.0)

    p.set_val('traj.descent.states:r',
              phases['descent'].interp('r', [10, 20]))
    p.set_val('traj.descent.states:h',
              phases['descent'].interp('h', [10, 0]))
    p.set_val('traj.descent.states:v',
              phases['descent'].interp('v', [20, 60]))
    p.set_val('traj.descent.states:gam',
              phases['descent'].interp('gam', [0, -45]),
              units='deg')

p = om.Problem(model=om.Group())

traj, phases = new_water_rocket_trajectory(objective='range')
traj = p.model.add_subsystem('traj', traj)

p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
# p.driver.opt_settings['print_level'] = 4
p.driver.opt_settings['maxiter'] = 100
# p.driver.opt_settings['mu_strategy'] = 'monotone'
# p.driver.declare_coloring(tol=1.0E-12)

# Finish Problem Setup
p.model.linear_solver = om.DirectSolver()

p.setup()
set_sane_initial_guesses(p, phases)

dm.run_problem(p, run_driver=True, simulate=True)