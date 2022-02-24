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

    predictor = predictor_ODE(horizon=a.test_max_horizon, dt=dt_sampling, intermediate_steps=intermediate_steps)

    predictor.setup(initial_state=states_0)
    output_array = predictor.predict(Q_array)
    output_array = output_array[..., [STATE_INDICES.get(key) for key in a.features] + [CONTROL_INDICES.get(key) for key in CONTROL_INPUTS]]

    return output_array
