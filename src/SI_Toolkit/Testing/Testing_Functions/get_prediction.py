from typing import Optional

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
import numpy as np
from tqdm import trange


def get_prediction(
        dataset,
        predictor: PredictorWrapper,
        dt: float,
        routine: str,
        test_max_horizon: int,
        hls: bool = False,
        forward_predictor: Optional[PredictorWrapper] = None,
        **kwargs,
):
    if routine == "simple evaluation":
        predictor_horizon = 1
        if predictor.predictor_type != 'neural':
            raise NotImplementedError('Simple evaluation routine is currently only implemented for neural predictor')
    else:
        predictor_horizon = test_max_horizon

    test_len = dataset.shape[0]-test_max_horizon  # Overwrites the config which might be string ('max') with a value computed at preprocessing

    # For backward trajectory networks, use negative dt
    backward_mode = 'backward' in routine
    dt = -abs(dt) if backward_mode else abs(dt)

    if backward_mode:
        test_len = test_len - 1  # Accounts for dataset shift as control input is in an earlier row

    stateful_components = ['RNN', 'GRU', 'LSTM']
    if predictor.predictor_type == 'neural' and any(stateful_component in predictor.model_name for stateful_component in stateful_components):
        mode = 'sequential'
    else:
        mode = 'batch'

    # mode = 'sequential'
    # mode = 'batch'

    if mode == 'batch':
        predictor.configure_with_compilation(batch_size=test_len, dt=dt, mode=routine, hls=hls)
    else:
        predictor.configure_with_compilation(batch_size=1, dt=dt, mode=routine, hls=hls)

    if hasattr(predictor.predictor, 'net_info') and hasattr(predictor.predictor.net_info, 'dt') and predictor.predictor.net_info.dt != 0.0:
        dt_predictions = predictor.predictor.net_info.dt
    elif hasattr(predictor.predictor, 'dt') and predictor.predictor.dt != 0.0:
        dt_predictions = predictor.predictor.dt
    else:
        dt_predictions = dt


    predictor_initial_input = dataset[predictor.predictor.predictor_initial_input_features].to_numpy()
    predictor_initial_input_len = len(predictor_initial_input)
    if predictor_initial_input is not None:
        if backward_mode:
            predictor_initial_input = predictor_initial_input[predictor_horizon + 1:, :]  # +1 accounts for shift of control inputs which needs to be done and makes the effective length of data shorter.
        else:
            predictor_initial_input = predictor_initial_input[:predictor_initial_input_len-predictor_horizon, :]
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

    if backward_mode:
        predictor_external_input = predictor_external_input[:-1]  # Shift by one for backword mode

    predictor_external_input_len = len(predictor_external_input)
    if backward_mode:
        # Backward: for each test point k at dataset index (predictor_horizon+k), we need [u(t-1), u(t-2), ..., u(t-pred_hor)]
        predictor_external_input_array = [predictor_external_input[..., predictor_horizon-i:predictor_external_input_len-i, :] for i in range(predictor_horizon+1)]
    else:
        predictor_external_input_array = [predictor_external_input[..., i: predictor_external_input_len-predictor_horizon + i, :] for i in range(predictor_horizon+1)]
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

            predictor.update(predictor_external_input_current_timestep[:, np.newaxis, 0, :], predictor_initial_input_formatted)   # FIXME: Does it still deal with RNN update correctly? Only if predict contains an reset to memory states, which I don't know now

    # if 'autoregressive' in routine:
    #     output = output[:, 1:, :]  # Remove initial state

    prediction = [output, predictor.predictor.predictor_output_features, dt_predictions]

    ## Calculate forward prediction
    if backward_mode and forward_predictor:
        if not isinstance(forward_predictor, PredictorWrapper):
            raise TypeError("forward_predictor must be an instance of PredictorWrapper, True, or None.")

        # Forward pass starts from the max-horizon backward prediction (last but one state)
        if output.shape[1] < 3:  # Should never happen
            raise ValueError("Backward predictor output is too short to compute forward predictions.")

        forward_predictor.configure_with_compilation(batch_size=test_len, dt=abs(dt), mode=routine.replace('_backward', ''), hls=hls)

        forward_initial_state = output[:, -2, :]  # State at max_horizon backward prediction


        forward_external_input_array = predictor_external_input_array[:, :-1, :]
        forward_external_input_array = np.flip(forward_external_input_array, axis=1)

        forward_output = forward_predictor.predict(forward_initial_state, forward_external_input_array)

        forward_prediction = [forward_output, forward_predictor.predictor.predictor_output_features,  abs(dt)]
        prediction.append(forward_prediction)
    else:
        prediction.append(None)

    return prediction
