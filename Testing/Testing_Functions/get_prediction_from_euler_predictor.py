import numpy as np

try:
    from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, CONTROL_INPUTS, CONTROL_INDICES
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE


def get_prediction_from_euler_predictor(a, dataset, dt_sampling, intermediate_steps=1):

    states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]

    Q = dataset['Q'].to_numpy()
    Q_array = [Q[i:-a.test_max_horizon+i] for i in range(a.test_max_horizon)]
    Q_array = np.vstack(Q_array).transpose()
    Q_array = Q_array[..., np.newaxis]  # Add axis marking that there is only one control input

    output_array = np.zeros([a.test_len, a.test_max_horizon + 1, len(a.features) + len(CONTROL_INPUTS)],
                            dtype=np.float32)

    output_array[:, :-1, -len(CONTROL_INPUTS):] = Q_array


    predictor = predictor_ODE(horizon=a.test_max_horizon, dt=dt_sampling, intermediate_steps=intermediate_steps)

    output = predictor.predict(states_0, Q_array)
    output_array[:, :, :-len(CONTROL_INPUTS)] = output[..., [STATE_INDICES.get(key) for key in a.features]]

    return output_array
