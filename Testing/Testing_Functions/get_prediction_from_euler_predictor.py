import numpy as np

from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES

from SI_Toolkit_ApplicationSpecificFiles.predictor_ODE import predictor_ODE

DEFAULT_SAMPLING_INTERVAL = 0.02  # s, Corresponds to our lab cartpole
def get_prediction_from_euler_predictor(a, dataset, dt_sampling, intermediate_steps=1):

    states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]
    Q = dataset['Q'].to_numpy()
    Q_array = [Q[i:-a.test_max_horizon+i] for i in range(a.test_max_horizon)]
    Q_array = np.vstack(Q_array).transpose()

    predictor = predictor_ODE(horizon=a.test_max_horizon, dt=dt_sampling, intermediate_steps=intermediate_steps)

    predictor.setup(initial_state=states_0, prediction_denorm=True)
    output_array = predictor.predict(Q_array)  # Should be shape=(a.test_max_horizon, a.test_len, len(outputs))
    output_array = output_array[..., [STATE_INDICES.get(key) for key in a.features]+[-1]]

    return output_array


