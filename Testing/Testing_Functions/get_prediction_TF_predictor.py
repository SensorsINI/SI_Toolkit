import numpy as np

from tqdm import trange

try:
    from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, CONTROL_INPUTS, CONTROL_INDICES
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

from SI_Toolkit.Predictors.predictor_autoregressive_tf_Jerome import predictor_autoregressive_tf
# from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf


def get_data_for_gui_TF(a, dataset, net_name, dt, intermediate_steps):
    states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]

    Q = dataset[CONTROL_INPUTS].to_numpy()
    Q_array = [Q[..., i:-a.test_max_horizon+i, :] for i in range(a.test_max_horizon)]
    Q_array = np.stack(Q_array, axis=1)

    output_array = np.zeros([a.test_len, a.test_max_horizon + 1, len(a.features) + len(CONTROL_INPUTS)],
                            dtype=np.float32)
    output_array[:, :-1, -len(CONTROL_INPUTS):] = Q_array

    if net_name[:5] == 'Dense':
        mode = 'batch'
    else:
        mode = 'sequential'
    mode = 'batch'
    # mode = 'sequential'
    if mode == 'batch':
        # All at once
        # TODO: Euler TF here in batch mode produces strange results
        if net_name == 'EulerTF':
            predictor = predictor_ODE_tf(horizon=a.test_max_horizon, dt=dt, intermediate_steps=intermediate_steps)
        else:
            predictor = predictor_autoregressive_tf(horizon=a.test_max_horizon, batch_size=a.test_len,
                                                    net_name=net_name)

        output = predictor.predict(states_0, Q_array)
        output_array[:, :, :-len(CONTROL_INPUTS)] = output[..., [STATE_INDICES.get(key) for key in a.features]]

    elif mode == 'sequential':
        # predictor = predictor_autoregressive_tf(a=a, batch_size=1)
        if net_name == 'EulerTF':
            predictor = predictor_ODE_tf(horizon=a.test_max_horizon, dt=dt, intermediate_steps=intermediate_steps)
        else:
            predictor = predictor_autoregressive_tf(horizon=a.test_max_horizon, batch_size=1, net_name=net_name)
        # Iteratively (to test internal state update)
        output = np.zeros([a.test_len, a.test_max_horizon + 1, len(a.features)],
                            dtype=np.float32)
        for timestep in trange(a.test_len):
            Q_current_timestep = Q_array[np.newaxis, timestep, :]
            s_current_timestep = states_0[timestep, np.newaxis]
            output[timestep, :, :] = predictor.predict(s_current_timestep, Q_current_timestep)
            predictor.update_internal_state(s_current_timestep, Q_current_timestep[:, :1, :])

        output_array[:, :, :-len(CONTROL_INPUTS)] = output[..., [STATE_INDICES.get(key) for key in a.features]]

    return output_array
