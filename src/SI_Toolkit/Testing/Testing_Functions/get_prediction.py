from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
import numpy as np
from tqdm import trange

from SI_Toolkit_ASF.predictors_customization_tf import STATE_VARIABLES, CONTROL_INPUTS

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_INDICES
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')


def get_prediction(
        dataset,
        predictor: PredictorWrapper,
        features_to_plot: list,
        test_max_horizon: int,
        **kwargs,
):

    test_len = dataset.shape[0]-test_max_horizon # Overwrites the config which might be string ('max') with a value computed at preprocessing

    states_0 = dataset[STATE_VARIABLES].to_numpy()[:-test_max_horizon, :]

    Q = dataset[CONTROL_INPUTS].to_numpy()
    Q_array = [Q[..., i:-test_max_horizon + i, :] for i in range(test_max_horizon)]
    Q_array = np.stack(Q_array, axis=1)

    #parameters for multiple/single run modes
    #multiple = True
    iterations = 10

    #if multiple == True:
    #    output_array = np.zeros([iterations, test_len, test_max_horizon + 1, len(features_to_plot)],
    #                       dtype=np.float32)
    #else:
    #    output_array = np.zeros([test_len, test_max_horizon + 1, len(features_to_plot)],
    #                            dtype=np.float32)

    output_array = np.zeros([iterations, test_len, test_max_horizon + 1, len(features_to_plot)], dtype=np.float32)



    stateful_components = ['RNN', 'GRU', 'LSTM']
    if predictor.predictor_type == 'neural' and any(stateful_component in predictor.model_name for stateful_component in stateful_components):
        mode = 'sequential'
    else:
        mode = 'batch'

    #mode = 'sequential'
    # mode = 'batch'


    if mode == 'batch':
        predictor.configure_with_compilation(batch_size=test_len, horizon=test_max_horizon, dt=0.02)
    else:
        predictor.configure_with_compilation(batch_size=1, horizon=test_max_horizon, dt=0.02)

    if mode == 'batch':
        #if multiple == True:
        #    for i in range(iterations):
        #        #output = predictor.predict(states_0, Q_array)
        #        #output_array = np.zeros([iterations, test_len, test_max_horizon + 1, len(features_to_plot)],
        #        #                        dtype=np.float32)
        #        #output_array[:, :, :, :] = output[..., [STATE_INDICES.get(key) for key in features_to_plot]]
        #        output1 = np.copy(predictor.predict(states_0, Q_array))
        #        output_array1 = np.zeros([test_len, test_max_horizon + 1, len(features_to_plot)],
        #                                dtype=np.float32)
        #        output_array1[:, :, :] = np.copy(output1[..., [STATE_INDICES.get(key) for key in features_to_plot]]) #numpy concatenate or stack
        #        output_array[i, :, :, :] = np.copy(output_array1)
        #        #output_array_tot = np.concatenate((output_array, output_array), axis = 0)
        #else:
        #    output = predictor.predict(states_0, Q_array)
        #    output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in features_to_plot]]

        for i in range(iterations):
            output1 = np.copy(predictor.predict(states_0, Q_array))
            output_array1 = np.zeros([test_len, test_max_horizon + 1, len(features_to_plot)],dtype=np.float32)
            output_array1[:, :, :] = np.copy(output1[..., [STATE_INDICES.get(key) for key in features_to_plot]]) #numpy concatenate or stack
            output_array[i, :, :, :] = np.copy(output_array1)

    else:

        output = None
        print("hi")
        for timestep in trange(test_len):
            Q_current_timestep = Q_array[np.newaxis, timestep, :, :]
            s0 = states_0[np.newaxis, timestep, :] #clone and flatten
            s0_aug = []
            if output is None:
                output = predictor.predict(s0, Q_current_timestep) #flatten for predictor.predict
            else:
                output = np.concatenate((output, predictor.predict(s0, Q_current_timestep)), axis=0)
            predictor.update(Q_current_timestep[:, np.newaxis, 1, :], s0)
#unflatten here?
        output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in features_to_plot]]

    return output_array
