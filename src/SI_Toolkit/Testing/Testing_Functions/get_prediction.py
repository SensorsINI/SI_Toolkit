import numpy as np
from tqdm import trange

try:
    from SI_Toolkit_ASF_global.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, CONTROL_INDICES
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')

from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
# from SI_Toolkit.Predictors.predictor_autoregressive_tf_Jerome import predictor_autoregressive_tf
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf


def get_prediction(a, dataset, predictor_name, dt, intermediate_steps):
    states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]

    Q = dataset[CONTROL_INPUTS].to_numpy()
    Q_array = [Q[..., i:-a.test_max_horizon + i, :] for i in range(a.test_max_horizon)]
    Q_array = np.stack(Q_array, axis=1)

    output_array = np.zeros([a.test_len, a.test_max_horizon + 1, len(a.features)],
                            dtype=np.float32)

    stateful_components = ['RNN', 'GRU', 'LSTM']
    if any(stateful_component in predictor_name for stateful_component in stateful_components):
        mode = 'sequential'
    else:
        mode = 'batch'

    # mode = 'sequential'
    # mode = 'batch'

    if 'EulerTF' in predictor_name:
        predictor = predictor_ODE_tf(horizon=a.test_max_horizon, dt=dt, intermediate_steps=intermediate_steps)
    elif 'Euler' in predictor_name:
        predictor = predictor_ODE(horizon=a.test_max_horizon, dt=dt, intermediate_steps=intermediate_steps)
    else:
        if mode == 'batch':
            predictor = predictor_autoregressive_tf(horizon=a.test_max_horizon, batch_size=a.test_len,
                                                    net_name=predictor_name)
        else:
            predictor = predictor_autoregressive_tf(horizon=a.test_max_horizon, batch_size=1, net_name=predictor_name)

    if mode == 'batch':
        output = predictor.predict(states_0, Q_array)
        output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in a.features]]

    else:

        output = np.zeros([a.test_len, a.test_max_horizon + 1, len(a.features)],
                          dtype=np.float32)
        for timestep in trange(a.test_len):
            Q_current_timestep = Q_array[np.newaxis, timestep, :, :]
            s0 = states_0[np.newaxis, timestep, :]
            output[timestep, :, :] = predictor.predict(s0, Q_current_timestep)
            predictor.update_internal_state(Q_current_timestep[:, np.newaxis, 1, :], s0)

        output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in a.features]]

    return output_array
