import dymos as dm
import openmdao

from controller.rocket_phases.expulsion.expulsion_params import expulsion_set_params
from controller.rocket_phases.expulsion.expulsion_phase import expulsion_phase_fn


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