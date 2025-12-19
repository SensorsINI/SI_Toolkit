from typing import Optional, Dict

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
import numpy as np
from tqdm import trange

# State augmentation for backward-to-forward predictions
try:
    from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import augment_states_numpy
    HAS_AUGMENTATION = True
except ImportError:
    HAS_AUGMENTATION = False


def predict_with_fixed_batch_size(
        predictor: PredictorWrapper,
        initial_input: np.ndarray,
        external_input: np.ndarray,
        fixed_batch_size: int,
        actual_test_len: int,
) -> np.ndarray:
    """
    Predict using a fixed batch size with padding or chunking as needed.
    
    This allows reusing a predictor configured with a fixed batch size across
    datasets of different sizes.
    
    Args:
        predictor: Configured predictor wrapper
        initial_input: Initial state input [actual_test_len, state_features]
        external_input: External/control input [actual_test_len, horizon_steps, control_features]
        fixed_batch_size: The batch size the predictor was configured with
        actual_test_len: Actual number of test samples
        
    Returns:
        Predictions array [actual_test_len, ...]
    """
    if actual_test_len <= fixed_batch_size:
        # Pad to fixed batch size if needed
        if actual_test_len < fixed_batch_size:
            pad_size = fixed_batch_size - actual_test_len
            initial_input_padded = np.pad(initial_input, ((0, pad_size), (0, 0)), mode='edge')
            external_input_padded = np.pad(external_input, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
            output_padded = predictor.predict(initial_input_padded, external_input_padded)
            return output_padded[:actual_test_len]  # Remove padding
        else:
            # Exact match
            return predictor.predict(initial_input, external_input)
    else:
        # Process in chunks if data exceeds fixed batch size
        num_chunks = int(np.ceil(actual_test_len / fixed_batch_size))
        output_chunks = []
        for i in range(num_chunks):
            start_idx = i * fixed_batch_size
            end_idx = min((i + 1) * fixed_batch_size, actual_test_len)
            chunk_size = end_idx - start_idx
            
            # Pad last chunk if needed
            if chunk_size < fixed_batch_size:
                pad_size = fixed_batch_size - chunk_size
                init_chunk = np.pad(
                    initial_input[start_idx:end_idx],
                    ((0, pad_size), (0, 0)),
                    mode='edge'
                )
                ext_chunk = np.pad(
                    external_input[start_idx:end_idx],
                    ((0, pad_size), (0, 0), (0, 0)),
                    mode='edge'
                )
                chunk_output = predictor.predict(init_chunk, ext_chunk)[:chunk_size]
            else:
                chunk_output = predictor.predict(
                    initial_input[start_idx:end_idx],
                    external_input[start_idx:end_idx]
                )
            output_chunks.append(chunk_output)
        return np.concatenate(output_chunks, axis=0)


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
    
    # Check if external inputs are already time-shifted (e.g., angular_control_-1)
    # If so, don't apply additional [:-1] shift to avoid double-shifting
    external_features = predictor.predictor.predictor_external_input_features
    features_already_shifted = any(str(f).endswith('_-1') for f in external_features)
    
    if backward_mode and not features_already_shifted:
        predictor_external_input = predictor_external_input[:-1]  # Shift by one for backward mode
    
    predictor_external_input_len = len(predictor_external_input)
    
    # Build external input array for all horizons
    if backward_mode:
        # Backward: for each test point k at dataset index (predictor_horizon+1+k), we need [u(t-1), u(t-2), ..., u(t-pred_hor)]
        # If features are already shifted (e.g., angular_control_-1[t] = angular_control[t-1]),
        # we need to offset by +1 since we didn't apply [:-1] shift
        offset = 1 if features_already_shifted else 0
        predictor_external_input_array = [
            predictor_external_input[..., predictor_horizon+offset-i:predictor_external_input_len-i, :] 
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


def _build_external_input_array_from_dataset(
        dataset,
        external_input_features,
        predictor_horizon: int,
        backward_mode: bool,
        routine: str = None,
        test_max_horizon: int = None,
) -> np.ndarray:
    """
    Build external input tensor [test_len, predictor_horizon+1, control_dim] directly from dataset columns.

    This mirrors the external-input part of prepare_predictor_inputs(), but does NOT touch initial inputs.
    It is used when backward and forward predictors have different external-input feature sets
    (e.g., backward DI uses D_* derivatives, forward ODE uses angular_control/translational_control).
    """
    try:
        predictor_external_input = dataset[external_input_features].to_numpy()
    except KeyError as e:
        # Provide a more actionable error than bare KeyError
        requested = list(external_input_features)
        available = list(getattr(dataset, "columns", []))
        raise KeyError(
            f"External input features {requested} not found in dataset columns. "
            f"Missing: {[f for f in requested if f not in available]}"
        ) from e

    # Check if external inputs are already time-shifted (e.g., angular_control_-1)
    # If so, don't apply additional [:-1] shift to avoid double-shifting
    features_already_shifted = any(str(f).endswith('_-1') for f in external_input_features)

    if backward_mode and not features_already_shifted:
        predictor_external_input = predictor_external_input[:-1]  # Shift by one for backward mode

    predictor_external_input_len = len(predictor_external_input)

    if backward_mode:
        # If features are already shifted (e.g., angular_control_-1), offset indexing by +1
        offset = 1 if features_already_shifted else 0
        predictor_external_input_array = [
            predictor_external_input[..., predictor_horizon + offset - i: predictor_external_input_len - i, :]
            for i in range(predictor_horizon + 1)
        ]
    else:
        predictor_external_input_array = [
            predictor_external_input[..., i: predictor_external_input_len - predictor_horizon + i, :]
            for i in range(predictor_horizon + 1)
        ]

    predictor_external_input_array = np.stack(predictor_external_input_array, axis=1)

    # Handle simple evaluation mode
    if routine == "simple evaluation":
        predictor_external_input_array = predictor_external_input_array[:len(predictor_external_input) - test_max_horizon, ...]
    else:
        predictor_external_input_array = predictor_external_input_array[:len(predictor_external_input), ...]

    return predictor_external_input_array


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
        max_batch_size: int = None,
        forward_predictor_created: bool = False,
        param_name: Optional[str] = None,
        param_value: Optional[float] = None,
        backward_predictor_features: Optional[np.ndarray] = None,
        backward_output_features: Optional[np.ndarray] = None,
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
        max_batch_size: Maximum batch size for predictor (for reuse with padding/chunking)
        forward_predictor_created: Whether forward predictor was just created (needs configuration)
        param_name: Name of parameter being swept (e.g., 'mu')
        param_value: Value of parameter for this sweep iteration
        backward_predictor_features: External input features of backward predictor (for reordering if needed)
        backward_output_features: Output features of backward predictor (for state augmentation)
        
    Returns:
        Dictionary mapping horizon_idx to forward prediction arrays
    """
    if backward_output.shape[1] < 3:
        raise ValueError("Backward predictor output is too short to compute forward predictions.")
    
    if test_len is None:
        test_len = backward_output.shape[0]
    
    # Determine batch size for forward predictor
    if max_batch_size is None:
        # Legacy behavior: configure with actual test_len
        configured_batch_size = test_len
    else:
        configured_batch_size = max_batch_size
    
    # Configure forward predictor (only if newly created or not yet configured)
    if forward_predictor_created or not hasattr(forward_predictor, '_configured_batch_size'):
        forward_predictor.configure_with_compilation(
            batch_size=configured_batch_size, 
            dt=abs(dt), 
            mode=routine.replace('_backward', ''), 
            hls=hls
        )
        forward_predictor._configured_batch_size = configured_batch_size
    
    # Handle parameter mapping between backward and forward predictors
    # The predictor_external_input_array was built with backward predictor's feature ordering
    # We need to ensure the forward predictor receives the parameter at the correct index
    forward_features = forward_predictor.predictor.predictor_external_input_features
    need_reorder = False
    feature_mapping = None
    
    if backward_predictor_features is not None and not np.array_equal(backward_predictor_features, forward_features):
        # Feature orderings differ - need to reorder
        need_reorder = True
        # Build mapping: forward_idx -> backward_idx
        feature_mapping = []
        
        # Create normalized backward feature names (strip _-1 suffix if present)
        backward_features_normalized = []
        for feat in backward_predictor_features:
            feat_str = str(feat)
            if feat_str.endswith('_-1'):
                backward_features_normalized.append(feat_str[:-3])  # Remove '_-1'
            else:
                backward_features_normalized.append(feat_str)
        backward_features_normalized = np.array(backward_features_normalized)
        
        for fwd_feat in forward_features:
            fwd_feat_str = str(fwd_feat)
            try:
                # First try exact match
                matches = np.where(backward_predictor_features == fwd_feat)[0]
                if len(matches) > 0:
                    bwd_idx = matches[0]
                else:
                    # Try matching with normalized names (without _-1 suffix)
                    bwd_idx = np.where(backward_features_normalized == fwd_feat_str)[0][0]
                feature_mapping.append(bwd_idx)
            except IndexError:
                raise ValueError(f"Forward predictor feature '{fwd_feat}' not found in backward predictor features: {list(backward_predictor_features)}")
        feature_mapping = np.array(feature_mapping)
    
    # Augment backward output states if needed (e.g., 8-state NN output -> 10-state ODE input)
    # Check if forward predictor expects more state features than backward output provides
    backward_state_features = backward_output.shape[-1]
    forward_initial_features = forward_predictor.predictor.predictor_initial_input_features
    forward_state_count = len(forward_initial_features) if forward_initial_features is not None else backward_state_features
    
    backward_output_augmented = backward_output
    if HAS_AUGMENTATION and forward_state_count > backward_state_features and backward_output_features is not None:
        # Need to augment backward output to match forward predictor's expected state dimension
        try:
            # Normalize backward output feature names (strip D_ prefix and _-1 suffix for differential networks)
            normalized_input_features = []
            for feat in backward_output_features:
                feat_str = str(feat)
                # Strip D_ prefix (differential output)
                if feat_str.startswith('D_'):
                    feat_str = feat_str[2:]
                # Strip _-1 suffix (backward time shift)
                if feat_str.endswith('_-1'):
                    feat_str = feat_str[:-3]
                normalized_input_features.append(feat_str)
            
            backward_output_augmented, _ = augment_states_numpy(
                states=backward_output,
                input_features=normalized_input_features,
                target_features=list(forward_initial_features),
                verbose=False
            )
        except Exception as e:
            # If augmentation fails, continue with original output
            print(f"Warning: State augmentation failed ({e}), using backward output as-is")
            backward_output_augmented = backward_output

    # Determine loop range
    start_horizon = predictor_horizon
    if forward_from_all_horizons:
        stop_horizon = 0
        horizons_to_compute = list(range(start_horizon, stop_horizon, -1))
    else:
        stop_horizon = predictor_horizon - 1
        horizons_to_compute = [predictor_horizon]
    
    num_horizons = len(horizons_to_compute)
    
    # BATCH OPTIMIZATION: Process all horizons in a single forward pass
    # Stack all initial states and pad control sequences to max horizon
    if num_horizons > 1 and max_batch_size is None:
        print(f"[Brunton] Batching forward predictions for {num_horizons} horizons...")
        
        # Prepare batched inputs
        # Stack initial states: [test_len * num_horizons, state_dim]
        all_initial_states = []
        all_controls = []
        
        max_h = max(horizons_to_compute)
        control_dim = predictor_external_input_array.shape[-1]
        if need_reorder and feature_mapping is not None:
            control_dim = len(feature_mapping)
        
        for horizon_idx in horizons_to_compute:
            # Initial state at this horizon (already augmented with linear_vel_y computed from slip_angle * v_x)
            initial_state = backward_output_augmented[:, horizon_idx, :].copy()
            all_initial_states.append(initial_state)
            
            # Control sequence (padded to max horizon)
            controls = predictor_external_input_array[:, :horizon_idx, :]
            controls = np.flip(controls, axis=1)
            if need_reorder and feature_mapping is not None:
                controls = controls[:, :, feature_mapping]
            
            # Pad to max horizon length
            if horizon_idx < max_h:
                pad_len = max_h - horizon_idx
                padding = np.zeros((test_len, pad_len, control_dim), dtype=controls.dtype)
                controls = np.concatenate([controls, padding], axis=1)
            
            all_controls.append(controls)
        
        # Stack into mega-batch: [test_len * num_horizons, ...]
        batched_initial = np.concatenate(all_initial_states, axis=0)  # [test_len*num_horizons, state_dim]
        batched_controls = np.concatenate(all_controls, axis=0)  # [test_len*num_horizons, max_h, control_dim]
        
        # Configure predictor for mega-batch (reconfigure if needed)
        mega_batch_size = test_len * num_horizons
        forward_predictor.configure_with_compilation(
            batch_size=mega_batch_size, 
            dt=abs(dt), 
            mode=routine.replace('_backward', ''), 
            hls=hls
        )
        
        # Single forward pass for all horizons
        batched_output = forward_predictor.predict(batched_initial, batched_controls)
        # batched_output: [test_len*num_horizons, max_h+1, state_dim]
        
        # Unpack results for each horizon
        forward_outputs_dict = {}
        for i, horizon_idx in enumerate(horizons_to_compute):
            start_idx = i * test_len
            end_idx = (i + 1) * test_len
            # Truncate output to actual horizon length (+1 for initial state)
            forward_outputs_dict[horizon_idx] = batched_output[start_idx:end_idx, :horizon_idx+1, :]
        
    else:
        # Sequential processing (original behavior for single horizon or with max_batch_size)
        forward_outputs_dict = {}
        desc_text = f"Forward from all horizons ({predictor_horizon} to 1)" if num_horizons > 1 else "Forward from max horizon"
        iterator = trange(len(horizons_to_compute), desc=desc_text) if num_horizons > 1 else range(len(horizons_to_compute))
        
        for i in iterator:
            horizon_idx = horizons_to_compute[i]
            # State at this horizon (already augmented with linear_vel_y computed from slip_angle * v_x)
            forward_initial_state = backward_output_augmented[:, horizon_idx, :].copy()
            
            # External inputs for forward prediction from this horizon
            forward_external_input_slice = predictor_external_input_array[:, :horizon_idx, :]
            forward_external_input_slice = np.flip(forward_external_input_slice, axis=1)
            
            # Reorder features if backward and forward predictors have different feature orderings
            if need_reorder and feature_mapping is not None:
                forward_external_input_slice = forward_external_input_slice[:, :, feature_mapping]
            
            # Use fixed batch size with padding/chunking if max_batch_size is provided
            if max_batch_size is not None and configured_batch_size != test_len:
                forward_output = predict_with_fixed_batch_size(
                    predictor=forward_predictor,
                    initial_input=forward_initial_state,
                    external_input=forward_external_input_slice,
                    fixed_batch_size=configured_batch_size,
                    actual_test_len=test_len,
                )
            else:
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
        param_name: Optional[str] = None,
        param_value: Optional[float] = None,
        test_stride: int = 1,
        **kwargs,
):
    if routine == "simple evaluation":
        predictor_horizon = 1
        if predictor.predictor_type != 'neural':
            raise NotImplementedError('Simple evaluation routine is currently only implemented for neural predictor')
    else:
        predictor_horizon = test_max_horizon

    test_len_full = dataset.shape[0]-test_max_horizon  # Full test length before stride

    # For backward trajectory networks, use negative dt
    backward_mode = 'backward' in routine
    dt = -abs(dt) if backward_mode else abs(dt)

    if backward_mode:
        test_len_full = test_len_full - 1  # Accounts for dataset shift as control input is in an earlier row

    # Apply stride to reduce number of trajectories
    test_stride = max(1, int(test_stride))
    test_len = (test_len_full + test_stride - 1) // test_stride  # Ceiling division
    
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

    # Apply stride to reduce computation (only use every n-th trajectory)
    stride_indices = None
    if test_stride > 1:
        stride_indices = np.arange(0, test_len_full, test_stride)
        predictor_initial_input = predictor_initial_input[::test_stride]
        predictor_external_input_array = predictor_external_input_array[::test_stride]
        # Update test_len to match actual data after stride
        test_len = predictor_initial_input.shape[0]

    # Override parameter value in external inputs if specified (for parameter sweep)
    if param_name is not None and param_value is not None:
        external_features = predictor.predictor.predictor_external_input_features
        param_idx = None
        for idx, feat in enumerate(external_features):
            if feat == param_name:
                param_idx = idx
                break
        if param_idx is not None:
            # Override the parameter across all samples and horizons
            predictor_external_input_array[:, :, param_idx] = param_value
        else:
            print(f"Warning: param_name '{param_name}' not found in external features: {external_features}")

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

    ## Calculate forward prediction (before NaN expansion)
    forward_prediction = None
    if backward_mode and forward_predictor:
        if not isinstance(forward_predictor, PredictorWrapper):
            raise TypeError("forward_predictor must be an instance of PredictorWrapper.")

        # Ensure forward_predictor.predictor exists before we query its feature lists.
        # (PredictorWrapper builds .predictor only during configure/configure_with_compilation.)
        if getattr(forward_predictor, "predictor", None) is None:
            forward_predictor.configure_with_compilation(
                batch_size=test_len,
                dt=abs(dt),
                mode=routine.replace('_backward', ''),
                hls=hls,
            )
            forward_predictor._configured_batch_size = test_len

        # If backward and forward predictors use different external-input feature sets (e.g., DI uses D_* derivatives
        # but ODE expects angular_control/translational_control), the default reuse+reorder logic will fail.
        # In that case, build the forward external input tensor directly from dataset columns.
        backward_ext_features = np.array(predictor.predictor.predictor_external_input_features)
        forward_ext_features = np.array(forward_predictor.predictor.predictor_external_input_features)

        external_input_for_forward = predictor_external_input_array
        backward_features_for_mapping = backward_ext_features

        if not np.array_equal(backward_ext_features, forward_ext_features):
            # Determine if forward features can be mapped from backward features; if not, fall back to dataset controls.
            backward_features_normalized = []
            for feat in backward_ext_features:
                feat_str = str(feat)
                backward_features_normalized.append(feat_str[:-3] if feat_str.endswith('_-1') else feat_str)
            backward_features_normalized = np.array(backward_features_normalized)

            missing_forward = []
            for fwd_feat in forward_ext_features:
                fwd_feat_str = str(fwd_feat)
                if (len(np.where(backward_ext_features == fwd_feat)[0]) == 0 and
                        len(np.where(backward_features_normalized == fwd_feat_str)[0]) == 0):
                    missing_forward.append(fwd_feat_str)

            if len(missing_forward) > 0:
                # Build external inputs for forward predictor directly from dataset.
                # Important: keep backward_mode=True so the control alignment matches calculate_back2front_predictions().
                external_input_for_forward = _build_external_input_array_from_dataset(
                    dataset=dataset,
                    external_input_features=list(forward_ext_features),
                    predictor_horizon=predictor_horizon,
                    backward_mode=True,
                    routine=routine,
                    test_max_horizon=test_max_horizon,
                )

                # Apply stride if Brunton is configured to evaluate every n-th trajectory
                if test_stride > 1:
                    external_input_for_forward = external_input_for_forward[::test_stride]

                # Apply parameter override for sweep, if the parameter exists in forward external inputs.
                if param_name is not None and param_value is not None:
                    try:
                        param_idx_fwd = list(forward_ext_features).index(param_name)
                        external_input_for_forward[:, :, param_idx_fwd] = param_value
                    except ValueError:
                        # param_name not part of forward external features – ignore
                        pass

                # Tell calculate_back2front_predictions() that external_input_for_forward is already in forward feature order
                backward_features_for_mapping = forward_ext_features

        # Use refactored function to calculate forward predictions
        forward_outputs_dict = calculate_back2front_predictions(
            backward_output=output,
            predictor_external_input_array=external_input_for_forward,
            forward_predictor=forward_predictor,
            predictor_horizon=predictor_horizon,
            dt=dt,
            routine=routine,
            hls=hls,
            forward_from_all_horizons=forward_from_all_horizons,
            test_len=test_len,
            param_name=param_name,
            param_value=param_value,
            backward_predictor_features=backward_features_for_mapping,
            backward_output_features=predictor.predictor.predictor_output_features,
        )
        
        # Package forward results (will be expanded below if strided)
        if forward_from_all_horizons:
            forward_prediction = [forward_outputs_dict, forward_predictor.predictor.predictor_output_features, abs(dt), True]
        else:
            forward_prediction = [forward_outputs_dict[predictor_horizon], forward_predictor.predictor.predictor_output_features, abs(dt), False]

    # Expand strided predictions to full size with NaN padding (for GUI compatibility)
    if stride_indices is not None and test_stride > 1:
        # Create full-size output filled with NaN
        full_output = np.full((test_len_full, output.shape[1], output.shape[2]), np.nan, dtype=output.dtype)
        # Place computed values at their original indices
        full_output[stride_indices[:len(output)]] = output
        output = full_output
        
        # Also expand forward predictions if they exist
        if forward_prediction is not None:
            if forward_from_all_horizons:
                # forward_outputs_dict is a dict of horizon -> array
                expanded_dict = {}
                for h, arr in forward_outputs_dict.items():
                    full_fwd = np.full((test_len_full, arr.shape[1], arr.shape[2]), np.nan, dtype=arr.dtype)
                    full_fwd[stride_indices[:len(arr)]] = arr
                    expanded_dict[h] = full_fwd
                forward_prediction[0] = expanded_dict
            else:
                # Single array
                arr = forward_prediction[0]
                full_fwd = np.full((test_len_full, arr.shape[1], arr.shape[2]), np.nan, dtype=arr.dtype)
                full_fwd[stride_indices[:len(arr)]] = arr
                forward_prediction[0] = full_fwd
        
        print(f"[Brunton] Expanded strided predictions: {len(stride_indices)} computed -> {test_len_full} total (with NaN padding)")

    prediction = [output, predictor.predictor.predictor_output_features, dt_predictions]
    prediction.append(forward_prediction)

    return prediction
