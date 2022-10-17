import numpy as np
from tqdm import trange

from SI_Toolkit_ASF.predictors_customization_tf import STATE_VARIABLES, CONTROL_INPUTS

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_INDICES
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')


def get_prediction(
        dataset,
        predictor,
        features_to_plot: list,
        test_max_horizon: int,
        **kwargs,
):

    test_len = dataset.shape[0]-test_max_horizon # Overwrites the config which might be string ('max') with a value computed at preprocessing

    states_0 = dataset[STATE_VARIABLES].to_numpy()[:-test_max_horizon, :]

    Q = dataset[CONTROL_INPUTS].to_numpy()
    Q_array = [Q[..., i:-test_max_horizon + i, :] for i in range(test_max_horizon)]
    Q_array = np.stack(Q_array, axis=1)

    output_array = np.zeros([test_len, test_max_horizon + 1, len(features_to_plot)],
                            dtype=np.float32)


    stateful_components = ['RNN', 'GRU', 'LSTM']
    if predictor.predictor_type == 'neural' and any(stateful_component in predictor.model_name for stateful_component in stateful_components):
        mode = 'sequential'
    else:
        mode = 'batch'

    # mode = 'sequential'
    # mode = 'batch'

    if mode == 'batch':
        predictor.configure_with_compilation(batch_size=test_len, horizon=test_max_horizon)
    else:
        predictor.configure_with_compilation(batch_size=1, horizon=test_max_horizon)

    if mode == 'batch':
        output = predictor.predict(states_0, Q_array)
        output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in features_to_plot]]

    else:

        output = None
        for timestep in trange(test_len):
            Q_current_timestep = Q_array[np.newaxis, timestep, :, :]
            s0 = states_0[np.newaxis, timestep, :]
            if output is None:
                output = predictor.predict(s0, Q_current_timestep)
            else:
                output = np.concatenate((output, predictor.predict(s0, Q_current_timestep)), axis=0)
            predictor.update(Q_current_timestep[:, np.newaxis, 1, :], s0)

        output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in features_to_plot]]

    return output_array
