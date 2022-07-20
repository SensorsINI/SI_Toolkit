import numpy as np
from tqdm import trange

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, CONTROL_INDICES
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')


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
        from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
        predictor = predictor_ODE_tf(horizon=a.test_max_horizon, dt=dt, intermediate_steps=intermediate_steps, batch_size=a.test_len)
    elif 'Euler' in predictor_name:
        from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
        predictor = predictor_ODE(horizon=a.test_max_horizon, dt=dt, intermediate_steps=intermediate_steps, batch_size=a.test_len)
    elif 'GP' in predictor_name:
        from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
        if mode == 'batch':
            predictor = predictor_autoregressive_GP(model_name=predictor_name, horizon=a.test_max_horizon, num_rollouts=a.test_len)
        else:
            predictor = predictor_autoregressive_GP(model_name=predictor_name, horizon=a.test_max_horizon)
    else:
        from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
        if mode == 'batch':
            predictor = predictor_autoregressive_tf(horizon=a.test_max_horizon, batch_size=a.test_len,
                                                    net_name=predictor_name)
        else:
            predictor = predictor_autoregressive_tf(horizon=a.test_max_horizon, batch_size=1, net_name=predictor_name, dt=dt)

    if mode == 'batch':
        output = predictor.predict(states_0, Q_array)
        output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in a.features]]

    else:

        output = None
        for timestep in trange(a.test_len):
            Q_current_timestep = Q_array[np.newaxis, timestep, :, :]
            s0 = states_0[np.newaxis, timestep, :]
            if output is None:
                output = predictor.predict(s0, Q_current_timestep)
            else:
                output = np.concatenate((output, predictor.predict(s0, Q_current_timestep)), axis=0)
            predictor.update_internal_state(Q_current_timestep[:, np.newaxis, 1, :], s0)

        output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in a.features]]

    return output_array
