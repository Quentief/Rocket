import dymos as dm
import openmdao

from controller.rocket_phases.expulsion.expulsion_params import expulsion_set_params
from controller.rocket_phases.expulsion.expulsion_phase import expulsion_phase_fn


def set_trajectories(prob: openmdao.core.problem.Problem, bottle_params: dict):

    transcript = dm.Radau(num_segments=3)

    # Add phases to trajectory
    traj = dm.Trajectory()
    expulsion_phase = traj.add_phase('expulsion', expulsion_phase_fn(transcription=transcript,
                                                                     pout=bottle_params["pout"]))
    expulsion_phase.add_objective("time", loc="final")

    # Add trajectory to OpenMDAO problem and set the variables values up
    prob.model.add_subsystem("traj", traj)
    prob.setup()
    expulsion_set_params(prob, bottle_params)

    return prob, traj