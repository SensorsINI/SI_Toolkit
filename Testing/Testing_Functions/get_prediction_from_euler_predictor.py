import numpy as np

from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES

from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf

DEFAULT_SAMPLING_INTERVAL = 0.02  # s, Corresponds to our lab cartpole
def get_prediction_from_euler_predictor(a, dataset, dt_sampling, intermediate_steps=1, name=None):

    states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]
    Q = dataset['Q'].to_numpy()
    Q_array = [Q[i:-a.test_max_horizon+i] for i in range(a.test_max_horizon)]
    Q_array = np.vstack(Q_array).transpose()

    if name == 'Euler':
        predictor = predictor_ODE(horizon=a.test_max_horizon, dt=dt_sampling, intermediate_steps=intermediate_steps)
    elif name == 'EulerTF':
        predictor = predictor_ODE_tf(horizon=a.test_max_horizon, dt=dt_sampling, intermediate_steps=intermediate_steps)
    elif name is None:
        raise ValueError('Name of predictor missing')
    else:
        raise NotImplementedError('Predictor with this name is not defined')

    predictor.setup(initial_state=states_0)
    output_array = predictor.predict(Q_array)  # Should be shape=(a.test_max_horizon, a.test_len, len(outputs))
    output_array = output_array[..., [STATE_INDICES.get(key) for key in a.features]+[-1]]

    return output_array


