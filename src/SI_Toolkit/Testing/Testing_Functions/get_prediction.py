from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
import numpy as np
from tqdm import trange

from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import STATE_VARIABLES, CONTROL_INPUTS

def get_prediction(
        dataset,
        predictor: PredictorWrapper,
        dt: float,
        routine: str,
        test_max_horizon: int,
        hls: bool = False,
        **kwargs,
):
    if routine == "simple evaluation":
        predictor_horizon = 1
        if predictor.predictor_type != 'neural':
            raise NotImplementedError('Simple evaluation routine is currently only implemented for neural predictor')
    else:
        predictor_horizon = test_max_horizon

    test_len = dataset.shape[0]-test_max_horizon  # Overwrites the config which might be string ('max') with a value computed at preprocessing

    stateful_components = ['RNN', 'GRU', 'LSTM']
    if predictor.predictor_type == 'neural' and any(stateful_component in predictor.model_name for stateful_component in stateful_components):
        mode = 'sequential'
    else:
        mode = 'batch'

    # mode = 'sequential'
    # mode = 'batch'

    if mode == 'batch':
        predictor.configure_with_compilation(batch_size=test_len, horizon=predictor_horizon, dt=dt, mode=routine, hls=hls)
    else:
        predictor.configure_with_compilation(batch_size=1, horizon=predictor_horizon, dt=dt, mode=routine, hls=hls)

    if hasattr(predictor.predictor, 'net_info') and hasattr(predictor.predictor.net_info, 'dt') and predictor.predictor.net_info.dt == 0.0:
        dt_predictions = 0.0
    elif hasattr(predictor.predictor, 'dt'):
        dt_predictions = predictor.predictor.dt
    else:
        dt_predictions = dt


    predictor_initial_input = dataset[predictor.predictor.predictor_initial_input_features].to_numpy()
    if predictor_initial_input is not None:
        predictor_initial_input = predictor_initial_input[:-test_max_horizon, :]
    try:
        predictor_external_input = dataset[predictor.predictor.predictor_external_input_features].to_numpy()
    except KeyError as e:
        similar_features = []
        for element in dataset.columns:
            # Check if any element in predictor.predictor.predictor_external_input_features is a substring of the current element in dataset.columns
            if any(fx in element for fx in predictor.predictor.predictor_external_input_features):
                # If yes, add the element to the similar_features
                similar_features.append(element)
        print(f"Predictor external input features {predictor.predictor.predictor_external_input_features} not found in the dataset.\n"
                       f"Similar features to requested found in the dataset: {similar_features}\n"
                       f"It might be that there exist versions of the feature with and without added noise.\n"
                       f"For cartpole e.g. this is Q (old but still used, general one), Q_applied (with noise), Q_calculated (without noise)\n"
                       "Select the proper one, for cartpole in state_utilities.py\n"
                       )
        raise KeyError(f'{e}' + f" See terminal output for more information.")



    predictor_external_input_array = [predictor_external_input[..., i:-predictor_horizon + i, :] for i in range(predictor_horizon)]
    predictor_external_input_array = np.stack(predictor_external_input_array, axis=1)

    if routine == "simple evaluation":
        predictor_external_input_array = predictor_external_input_array[:len(predictor_external_input) - test_max_horizon, ...]
    else:
        predictor_external_input_array = predictor_external_input_array[:len(predictor_external_input), ...]

    if mode == 'batch':
        output = predictor.predict(predictor_initial_input, predictor_external_input_array)

    else:

        output = None
        for timestep in trange(test_len):
            predictor_external_input_current_timestep = predictor_external_input_array[np.newaxis, timestep, :, :]
            predictor_initial_input_formatted = predictor_initial_input[np.newaxis, timestep, :]
            if output is None:
                output = predictor.predict(predictor_initial_input_formatted, predictor_external_input_current_timestep)
            else:
                output = np.concatenate((output, predictor.predict(predictor_initial_input_formatted, predictor_external_input_current_timestep)), axis=0)
            predictor.update(predictor_external_input_current_timestep[:, np.newaxis, 0, :], predictor_initial_input_formatted)

    if routine == 'autoregressive':
        output = output[:, 1:, :]  # Remove initial state

    prediction = [output, predictor.predictor.predictor_output_features, dt_predictions]

    return prediction
