import openmdao
import dymos


def expulsion_set_params(prob: openmdao.core.problem.Problem, expulsion_phase: dymos.phase.phase.Phase, params: dict):

    # Assign values to the times and states
    prob.set_val('traj.expulsion.t_initial', 0.0)
    prob.set_val('traj.expulsion.t_duration', params["delta_t"])

    # Assign initial states values
    prob.set_val('traj.expulsion.states:p', expulsion_phase.interp("p", [params["psat"], params["pout"]]), units="Pa")
    prob.set_val('traj.expulsion.states:Vl', expulsion_phase.interp('Vl', [params["Vl"], 0]), units="m**3")

    # Assign parameters values
    prob.set_val('traj.expulsion.parameters:Vb', params["Vb"])
    prob.set_val('traj.expulsion.parameters:gamma', params["gamma"])
    prob.set_val('traj.expulsion.parameters:rhol', params["rhol"])
    prob.set_val('traj.expulsion.parameters:Aout', params["Aout"])
    prob.set_val('traj.expulsion.parameters:pout', params["pout"])
    prob.set_val('traj.expulsion.parameters:deltap', params["deltap"])

    return prob