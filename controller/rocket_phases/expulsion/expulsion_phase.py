import dymos as dm

from controller.nox_bottle.bottle_model_ode import BottleModelODE


def expulsion_phase_fn(transcription: dm.transcriptions.pseudospectral.radau_pseudospectral.Radau, pout: float):

    phase = dm.Phase(ode_class=BottleModelODE, transcription=transcription)

    phase.set_time_options(fix_initial=True, fix_duration=True)

    # Define the states variables
    phase.add_state('p', units='bar', rate_source='p_dot', targets=['p'], fix_initial=True, fix_final=False,
                    lower=pout)
    phase.add_state('Vl', units='m**3', rate_source='Vl_dot', targets=['Vl'], fix_initial=False, fix_final=True,
                    lower=0)

    # Define the parameters variables
    phase.add_parameter('Vb', targets=['Vb'], units='m**3')
    phase.add_parameter('gamma', targets=['gamma'])
    phase.add_parameter('rhol', targets=['rhol'], units='kg/m**3')
    phase.add_parameter('Aout', targets=['Aout'], units='m**2')
    phase.add_parameter('pout', targets=['pout'], units="Pa")
    phase.add_parameter('deltap', targets=['deltap'], units="Pa")

    return phase