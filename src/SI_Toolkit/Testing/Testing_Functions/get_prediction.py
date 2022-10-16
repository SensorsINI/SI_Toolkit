import copy
import numpy as np
from tqdm import trange

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, CONTROL_INDICES
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')

def get_predictor(predictor_specification, config_predictors):

    config_predictors_local = copy.deepcopy(config_predictors)

    # Sort out predictor name from component specification
    predictor_name_components = predictor_specification.split(":")

    networks_names = ['Dense', 'RNN', 'GRU', 'DeltaGRU', 'LSTM']
    if any(network_name in predictor_specification for network_name in networks_names):
        predictor_name = 'neural'
        model_name = predictor_specification
    elif 'SGP' in predictor_specification:
        predictor_name = 'GP'
        model_name = predictor_specification
    else:
        predictor_name = predictor_name_components[0]
        model_name = None

    config_predictors_local['predictor_name_main'] = predictor_name

    predictor = PredictorWrapper(config_predictors_local)

    if model_name is not None:
        predictor.model_name = model_name

    if isinstance(predictor_specification, list) and len(predictor_specification) > 1:
        predictor.model_name = predictor_specification[1]

    return predictor


def get_prediction(config_testing, dataset, predictor):

    test_max_horizon = config_testing['MAX_HORIZON']
    test_len = config_testing['TEST_LEN']
    features_to_plot = config_testing['FEATURES_TO_PLOT']

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
        predictor.initialize_with_compilation(batch_size=test_len, horizon=test_max_horizon)
    else:
        predictor.initialize_with_compilation(batch_size=1, horizon=test_max_horizon)

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
