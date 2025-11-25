from typing import Optional, Dict

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
import numpy as np
from tqdm import trange


def prepare_predictor_inputs(
        dataset,
        predictor: PredictorWrapper,
        predictor_horizon: int,
        backward_mode: bool,
        routine: str = None,
        test_max_horizon: int = None,
):
    """
    Prepare initial and external inputs for predictor from dataset.
    
    Args:
        dataset: Input dataset (DataFrame)
        predictor: PredictorWrapper instance
        predictor_horizon: Prediction horizon
        backward_mode: Whether in backward prediction mode
        routine: Prediction routine (for simple evaluation check)
        test_max_horizon: Maximum test horizon (for simple evaluation)
        
    Returns:
        Tuple of (predictor_initial_input, predictor_external_input_array)
    """
    # Get initial inputs
    predictor_initial_input = dataset[predictor.predictor.predictor_initial_input_features].to_numpy()
    predictor_initial_input_len = len(predictor_initial_input)
    
    if predictor_initial_input is not None:
        if backward_mode:
            predictor_initial_input = predictor_initial_input[predictor_horizon + 1:, :]
        else:
            predictor_initial_input = predictor_initial_input[:predictor_initial_input_len-predictor_horizon, :]
    
    # Get external inputs (control signals)
    try:
        predictor_external_input = dataset[predictor.predictor.predictor_external_input_features].to_numpy()
    except KeyError as e:
        similar_features = []
        for element in dataset.columns:
            if any(fx in element for fx in predictor.predictor.predictor_external_input_features):
                similar_features.append(element)
        print(f"Predictor external input features {predictor.predictor.predictor_external_input_features} not found in the dataset.\n"
                       f"Similar features to requested found in the dataset: {similar_features}\n"
                       f"It might be that there exist versions of the feature with and without added noise.\n"
                       f"For cartpole e.g. this is Q (old but still used, general one), Q_applied (with noise), Q_calculated (without noise)\n"
                       "Select the proper one, for cartpole in state_utilities.py\n"
                       )
        raise KeyError(f'{e}' + " See terminal output for more information.")
    
    if backward_mode:
        predictor_external_input = predictor_external_input[:-1]  # Shift by one for backward mode
    
    predictor_external_input_len = len(predictor_external_input)
    
    # Build external input array for all horizons
    if backward_mode:
        # Backward: for each test point k at dataset index (predictor_horizon+k), we need [u(t-1), u(t-2), ..., u(t-pred_hor)]
        predictor_external_input_array = [
            predictor_external_input[..., predictor_horizon-i:predictor_external_input_len-i, :] 
            for i in range(predictor_horizon+1)
        ]
    else:
        predictor_external_input_array = [
            predictor_external_input[..., i: predictor_external_input_len-predictor_horizon + i, :] 
            for i in range(predictor_horizon+1)
        ]
    predictor_external_input_array = np.stack(predictor_external_input_array, axis=1)
    
    # Handle simple evaluation mode
    if routine == "simple evaluation":
        predictor_external_input_array = predictor_external_input_array[:len(predictor_external_input) - test_max_horizon, ...]
    else:
        predictor_external_input_array = predictor_external_input_array[:len(predictor_external_input), ...]
    
    return predictor_initial_input, predictor_external_input_array


def calculate_back2front_predictions(
        backward_output: np.ndarray,
        predictor_external_input_array: np.ndarray,
        forward_predictor: PredictorWrapper,
        predictor_horizon: int,
        dt: float,
        routine: str,
        hls: bool = False,
        forward_from_all_horizons: bool = False,
        test_len: int = None,
) -> Dict[int, np.ndarray]:
    """
    Calculate forward predictions from backward trajectory outputs.
    
    Args:
        backward_output: The output from backward prediction [test_len, predictor_horizon+1, features]
        predictor_external_input_array: External inputs array [test_len, predictor_horizon+1, features]
        forward_predictor: PredictorWrapper instance configured for forward prediction
        predictor_horizon: Maximum horizon used in backward prediction
        dt: Time step (will be made positive for forward prediction)
        routine: Prediction routine (e.g., "autoregressive_backward")
        hls: Whether to use HLS mode
        forward_from_all_horizons: If True, calculate forward from all horizons; if False, only from max horizon
        test_len: Number of test samples
        
    Returns:
        Dictionary mapping horizon_idx to forward prediction arrays
    """
    if backward_output.shape[1] < 3:
        raise ValueError("Backward predictor output is too short to compute forward predictions.")
    
    if test_len is None:
        test_len = backward_output.shape[0]
    
    # Configure forward predictor
    forward_predictor.configure_with_compilation(
        batch_size=test_len, 
        dt=abs(dt), 
        mode=routine.replace('_backward', ''), 
        hls=hls
    )
    
    # Determine loop range
    start_horizon = predictor_horizon
    if forward_from_all_horizons:
        stop_horizon = 0
        desc_text = f"Forward from all horizons ({predictor_horizon} to 1)"
    else:
        stop_horizon = predictor_horizon - 1
        desc_text = "Forward from max horizon"
    
    # Calculate forward trajectories
    forward_outputs_dict = {}
    for horizon_idx in trange(start_horizon, stop_horizon, -1, desc=desc_text):
        # State at this horizon
        forward_initial_state = backward_output[:, horizon_idx, :]
        
        # External inputs for forward prediction from this horizon
        # For backward prediction at horizon_idx, we went horizon_idx steps back in time
        # To reconstruct forward, we need to go those same horizon_idx steps forward
        # Take the first horizon_idx control inputs and flip them to go forward
        forward_external_input_slice = predictor_external_input_array[:, :horizon_idx, :]
        forward_external_input_slice = np.flip(forward_external_input_slice, axis=1)
        
        forward_output = forward_predictor.predict(forward_initial_state, forward_external_input_slice)
        forward_outputs_dict[horizon_idx] = forward_output
    
    return forward_outputs_dict


def get_prediction(
        dataset,
        predictor: PredictorWrapper,
        dt: float,
        routine: str,
        test_max_horizon: int,
        hls: bool = False,
        forward_predictor: Optional[PredictorWrapper] = None,
        forward_from_all_horizons: bool = False,
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

    # Prepare inputs using refactored function
    predictor_initial_input, predictor_external_input_array = prepare_predictor_inputs(
        dataset=dataset,
        predictor=predictor,
        predictor_horizon=predictor_horizon,
        backward_mode=backward_mode,
        routine=routine,
        test_max_horizon=test_max_horizon,
    )

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
            raise TypeError("forward_predictor must be an instance of PredictorWrapper.")

        # Use refactored function to calculate forward predictions
        forward_outputs_dict = calculate_back2front_predictions(
            backward_output=output,
            predictor_external_input_array=predictor_external_input_array,
            forward_predictor=forward_predictor,
            predictor_horizon=predictor_horizon,
            dt=dt,
            routine=routine,
            hls=hls,
            forward_from_all_horizons=forward_from_all_horizons,
            test_len=test_len,
        )
        
        # Package results
        if forward_from_all_horizons:
            forward_prediction = [forward_outputs_dict, forward_predictor.predictor.predictor_output_features, abs(dt), True]
        else:
            # Backward compatibility: return single array instead of dict
            forward_prediction = [forward_outputs_dict[predictor_horizon], forward_predictor.predictor.predictor_output_features, abs(dt), False]
        
        prediction.append(forward_prediction)
    else:
        prediction.append(None)

    return prediction
