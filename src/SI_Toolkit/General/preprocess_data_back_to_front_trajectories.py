"""
Transformation function to calculate backward-then-forward trajectories.

This function takes a dataframe and calculates:
1. Backward predictions from each seed state
2. Forward reconstruction from the furthest backward point
3. Returns all trajectories in one dataframe with experiment_index column

IMPORTANT: Control-State Alignment
==================================
Backward mode: x(t) = g(x(t+1), q(t))
  - To predict state at time t from state at t+1, we use control q(t)
  - Controls are shifted by -1 in prepare_predictor_inputs (line 123 of get_prediction.py)
  
Forward mode: x(t+1) = g(x(t), q(t))
  - To predict state at time t+1 from state at t, we use control q(t)
  - Standard forward dynamics
  
Example trajectory (horizon=30):
  - Seed at time T (dataset index predictor_horizon + 1 + traj_idx)
  - Backward: T → T-1 → ... → T-30 using controls q(T-1), q(T-2), ..., q(T-30) [30 controls]
  - Forward: T-30 → T-29 → ... → T using controls q(T-30), q(T-29), ..., q(T-1) [30 controls produce 31 states]
  - For alignment: q(T) is appended from external_input_array to match all 31 states
  - The controls are extracted and flipped in calculate_back2front_predictions

Parameter Randomization (e.g., mu)
==================================
When randomize_param is set (e.g., 'mu'), the parameter is randomized BEFORE predictions:
  - Each trajectory gets a unique random value sampled from param_range
  - The SAME random value is used for both backward and forward predictions
  - This ensures physical consistency: the same friction coefficient is used throughout
  - The output data contains the actual randomized parameter values used in predictions
"""

import numpy as np
import pandas as pd
from tqdm import trange

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
from SI_Toolkit.Testing.Testing_Functions.get_prediction import (
    calculate_back2front_predictions, 
    prepare_predictor_inputs,
    predict_with_fixed_batch_size
)


def back_to_front_trajectories(
    df,
    df_name,
    backward_predictor_specification,
    forward_predictor_specification,
    test_horizon,
    dataset_sampling_dt=0.02,
    verbose=False,
    predictor=None,
    forward_predictor=None,
    randomize_param=None,
    param_range=None,
    random_seed=None,
    compute_error_score=True,
    normalization_info=None,
    **kwargs
):
    """
    Calculate backward-then-forward trajectories for a single dataframe.
    
    This function is called by transform_dataset for each CSV file.
    Always processes the full file.
    
    Args:
        df: Input dataframe (one CSV file)
        df_name: Name of the file being processed
        backward_predictor_specification: Specification for backward predictor (e.g., 'Dense-9IN-32H1-32H2-8OUT-1')
        forward_predictor_specification: Specification for forward predictor (e.g., 'ODE')
        test_horizon: Prediction horizon
        dataset_sampling_dt: Sampling time step
        verbose: If True, print diagnostic information (default: False)
        predictor: Optional pre-created backward predictor (for reuse across files)
        forward_predictor: Optional pre-created forward predictor (for reuse across files)
        randomize_param: Name of control parameter to randomize (e.g., 'mu'). If None, no randomization (default: None)
        param_range: Tuple (min, max) for random parameter sampling. If None, uses (0.1, 1.0) (default: None)
        random_seed: Random seed for reproducibility. If None, uses hash of df_name (default: None)
        compute_error_score: If True, compute trajectory error score comparing backward vs forward (default: True)
        normalization_info: DataFrame with normalization statistics (mean, std, min, max). If None, uses raw values (default: None)
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with all trajectories including 'trajectory_error_score' column, or None if processing failed
    """
    
    # Use full dataset
    dataset = df.copy()
    dataset.reset_index(drop=True, inplace=True)
    
    if dataset.shape[0] < test_horizon:
        print(f"  Warning: Dataset too short ({dataset.shape[0]} rows < {test_horizon} horizon). Skipping.")
        return None
    
    # Create time axis if not present
    if 'time' in dataset.columns:
        time_axis = dataset['time'].to_numpy()
    else:
        time_axis = np.arange(0, len(dataset) * dataset_sampling_dt, dataset_sampling_dt)
    
    # Setup predictors (create only if not provided)
    if predictor is None:
        predictor = PredictorWrapper()
        predictor.update_predictor_config_from_specification(
            predictor_specification=backward_predictor_specification
        )
        predictor_created = True
    else:
        predictor_created = False
    
    if forward_predictor is None:
        forward_predictor = PredictorWrapper()
        forward_predictor.update_predictor_config_from_specification(
            predictor_specification=forward_predictor_specification
        )
        forward_predictor_created = True
    else:
        forward_predictor_created = False
    
    # Calculate backward predictions
    predictor_horizon = test_horizon
    dt = -abs(dataset_sampling_dt)  # Negative for backward
    routine = 'autoregressive_backward'
    
    # Get test length accounting for backward mode
    test_len_calc = dataset.shape[0] - test_horizon - 1  # -1 for backward shift
    
    # Use fixed batch size for predictor reuse
    # Get max_batch_size from kwargs or use default
    max_batch_size = kwargs.get('max_batch_size', 512)
    
    # Configure predictor BEFORE prepare_predictor_inputs (it needs predictor.predictor to be initialized)
    needs_configuration = predictor_created or not hasattr(predictor, '_configured_batch_size')
    
    if needs_configuration:
        # Configure with fixed batch size for reuse
        predictor.configure_with_compilation(
            batch_size=max_batch_size,
            dt=dt,
            mode=routine,
            hls=False
        )
        predictor._configured_batch_size = max_batch_size
        predictor._configured_dt = dt
        predictor._configured_mode = routine
    else:
        # Predictor already configured - check if reconfiguration needed
        if (not hasattr(predictor, '_configured_dt') or predictor._configured_dt != dt or
            not hasattr(predictor, '_configured_mode') or predictor._configured_mode != routine):
            if verbose:
                print("  Note: Predictor already configured, reusing existing configuration")
    
    # Prepare inputs using refactored function (predictor must be configured first!)
    try:
        predictor_initial_input, predictor_external_input_array = prepare_predictor_inputs(
            dataset=dataset,
            predictor=predictor,
            predictor_horizon=predictor_horizon,
            backward_mode=True,
            routine=routine,
            test_max_horizon=None,
        )
    except KeyError as e:
        print(f"  Error: Required features not found in dataset: {e}")
        return None
    
    # Get control features for parameter randomization check
    control_features = predictor.predictor.predictor_external_input_features
    
    # Setup random parameter generation and apply BEFORE predictions
    # This ensures the same randomized parameter is used in both backward and forward predictions
    random_param_values = None  # Will store the random value per trajectory
    if randomize_param is not None:
        if param_range is None:
            param_range = (0.1, 1.0)
        if random_seed is None:
            # Use hash of df_name for reproducibility per file
            random_seed = hash(df_name) % (2**32)
        rng = np.random.RandomState(random_seed)
        
        # Check if the parameter is in control features
        param_idx = None
        for idx, feat in enumerate(control_features):
            if feat == randomize_param:
                param_idx = idx
                break
        
        if param_idx is None:
            if verbose:
                print(f"  Warning: randomize_param='{randomize_param}' not found in control features: {control_features}")
        else:
            # Generate one random value per trajectory
            random_param_values = rng.uniform(param_range[0], param_range[1], size=test_len_calc)
            
            # Apply randomization to predictor_external_input_array BEFORE predictions
            # Shape: [test_len, predictor_horizon+1, control_features]
            for traj_idx in range(test_len_calc):
                predictor_external_input_array[traj_idx, :, param_idx] = random_param_values[traj_idx]
            
            if verbose:
                print(f"  Randomizing '{randomize_param}' for each trajectory (range: {param_range})")
                print(f"    Applied to {test_len_calc} trajectories before prediction")
    
    # Predict backward trajectories
    if verbose:
        print("  Calculating backward trajectories...")
        print(f"    Initial input shape: {predictor_initial_input.shape}")
        print(f"    External input array shape: {predictor_external_input_array.shape}")
    
    # Use fixed batch size with padding or chunking
    configured_batch_size = predictor._configured_batch_size
    backward_output = predict_with_fixed_batch_size(
        predictor=predictor,
        initial_input=predictor_initial_input,
        external_input=predictor_external_input_array,
        fixed_batch_size=configured_batch_size,
        actual_test_len=test_len_calc,
    )
    
    if verbose:
        print(f"    Backward output shape: {backward_output.shape}")
    
    # Extract feature names BEFORE augmentation
    backward_features_original = predictor.predictor.predictor_output_features
    
    # Validate forward_predictor is provided (required for back-to-front)
    if forward_predictor is None:
        raise ValueError("forward_predictor is required for back_to_front_trajectories transformation")
    
    # Augment backward output to match ODE's expected full state vector
    # Uses shared augmentation function from predictors_customization
    from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import augment_states_numpy
    backward_output_augmented, backward_features_augmented = augment_states_numpy(
        states=backward_output,
        input_features=backward_features_original,
        target_features=None,  # Use FULL_STATE_ORDER from predictors_customization
        verbose=verbose
    )
    backward_features_augmented = np.array(backward_features_augmented)
    
    # Use original features for output dataframes (NN output features)
    backward_features = backward_features_original
    
    # Calculate forward predictions from max horizon using refactored function
    if verbose:
        print("  Calculating forward trajectories...")
    forward_outputs_dict = calculate_back2front_predictions(
        backward_output=backward_output_augmented,
        predictor_external_input_array=predictor_external_input_array,
        forward_predictor=forward_predictor,
        predictor_horizon=predictor_horizon,
        dt=dt,
        routine='autoregressive',
        hls=False,
        forward_from_all_horizons=False,
        test_len=test_len_calc,
        max_batch_size=max_batch_size,
        forward_predictor_created=forward_predictor_created,
    )
    
    # Extract forward output for max horizon
    forward_output = forward_outputs_dict[predictor_horizon]
    if verbose:
        print(f"    Forward output shape: {forward_output.shape}")
    
    # Extract forward feature names (backward_features already extracted before augmentation)
    forward_features = forward_predictor.predictor.predictor_output_features
    # control_features already extracted above for parameter randomization
    
    # Build output dataframe
    if verbose:
        print("  Building output dataframe...")
    all_trajectories = []
    trajectory_error_scores = []  # Store error score for each trajectory
    
    # Use tqdm only if verbose, otherwise silent iteration
    iterator = trange(test_len_calc, desc="  Processing trajectories", leave=False) if verbose else range(test_len_calc)
    for traj_idx in iterator:
        # Get trajectories for this seed point
        backward_traj = backward_output[traj_idx, :, :]  # [horizon+1, features] - but may be horizon+2 depending on predictor
        forward_traj = forward_output[traj_idx, :, :]  # [horizon or horizon+1, features] depending on predictor
        
        # Seed index in original dataframe
        seed_idx = predictor_horizon + 1 + traj_idx
        
        # Get control inputs for this trajectory
        # External input array has controls used in backward prediction (in backward order)
        # The array has shape [test_len, predictor_horizon+1, control_features]
        # For horizon=30: backward uses 30 controls (indices 0:30), forward produces 31 states
        # We need to handle the case where we have more states than controls used in prediction
        num_forward_steps = len(forward_traj)
        
        # Take predictor_horizon controls (these were actually used in prediction)
        # Note: if randomize_param was set, predictor_external_input_array already has the randomized values
        control_traj = predictor_external_input_array[traj_idx, :predictor_horizon, :]  # [predictor_horizon, control_features]
        control_traj_forward = np.flip(control_traj, axis=0)  # Flip to forward time order
        
        # If forward trajectory has one more state than controls (initial state + predictions),
        # we need to add one more control for alignment. Take it from the dataset.
        if num_forward_steps == predictor_horizon + 1:
            # Need one more control - take the control that belongs to the seed state directly from the dataset
            extra_control = dataset.loc[seed_idx, control_features].to_numpy(dtype=float)[np.newaxis, :]
            
            # If parameter was randomized, apply the same random value to the extra control
            if random_param_values is not None:
                param_idx = None
                for idx, feat in enumerate(control_features):
                    if feat == randomize_param:
                        param_idx = idx
                        break
                if param_idx is not None:
                    extra_control[0, param_idx] = random_param_values[traj_idx]
            
            control_traj_forward = np.concatenate([control_traj_forward, extra_control], axis=0)
        
        # Note: backward_df is computed but not saved (user only wants forward trajectories)
        # Keeping this for debugging/verification purposes
        backward_df = pd.DataFrame(backward_traj, columns=backward_features)
        # Fix: adjust phase list to match actual backward_traj length
        num_backward_steps = len(backward_traj)
        backward_df['phase'] = ['seed'] + [f'backward_{i}' for i in range(1, num_backward_steps)]
        backward_df['time_step'] = np.arange(num_backward_steps)
        backward_df['absolute_time'] = time_axis[seed_idx] - np.arange(num_backward_steps) * abs(dt)
        
        # Forward part - this is what gets saved
        forward_df = pd.DataFrame(forward_traj, columns=forward_features)
        num_forward_steps = len(forward_traj)
        forward_df['phase'] = [f'forward_{i}' for i in range(num_forward_steps)]
        forward_df['time_step'] = np.arange(num_forward_steps)
        # Original time starts from furthest backward point and goes forward
        # Row 0 is at time T-H, Row H is at time T (the seed time)
        forward_df['original_time'] = (
            time_axis[seed_idx] - predictor_horizon * abs(dt) 
            + np.arange(num_forward_steps) * abs(dt)
        )
        
        # Add control inputs
        # IMPORTANT ALIGNMENT: In forward mode x(t+1) = f(x(t), q(t))
        # Row i contains: state x(T-horizon+i) and control q(T-horizon+i)
        # 
        # Example with horizon=30 (31 states from autoregressive predictor with 30 controls):
        # - Backward used controls q(T-1), ..., q(T-30) to produce 31 states x(T), ..., x(T-30)
        # - Forward uses controls q(T-30), ..., q(T-1) (flipped) to produce 31 states x(T-30), ..., x(T)
        # - We append q(T) by reading it from the original dataset (so it also picks up any randomized param)
        # Row 0: state x(T-30), control q(T-30) → produces x(T-29) in Row 1
        # Row 1: state x(T-29), control q(T-29) → produces x(T-28) in Row 2  
        # ...
        # Row 29: state x(T-1), control q(T-1) → produces x(T) in Row 30
        # Row 30: state x(T) [seed], control q(T) - no next state in this trajectory
        
        control_df = pd.DataFrame(control_traj_forward, columns=control_features)
        for col in control_df.columns:
            forward_df[col] = control_df[col].values
        
        # Compute error score if requested
        # Score = L∞ over features of (L2 over time for each normalized feature)
        # i.e., the worst feature's RMS error after normalization
        # 
        # IMPORTANT: Compare BACKWARD trajectory vs FORWARD trajectory (NOT vs ground truth!)
        # Both backward and forward use the SAME randomized mu, so they should match.
        # Ground truth has a DIFFERENT mu (original from dataset), so comparison would be unfair.
        if compute_error_score:
            # Get backward trajectory states (reversed to forward time order)
            # backward_traj: [horizon+1, features] with order [s(T), s(T-1), ..., s(T-H)]
            # We need to reverse it to [s(T-H), s(T-H+1), ..., s(T)] to match forward trajectory
            backward_traj_fwd_order = np.flip(backward_traj, axis=0)  # Now: [s(T-H), ..., s(T)]
            
            # Use NN output features directly (backward_features_original has 8 states)
            # These are the features the NN actually predicts
            features_for_error = list(backward_features_original)
            
            # Get backward states (all features)
            backward_states = backward_traj_fwd_order.astype(np.float64)
            
            # Get forward trajectory states for same features
            forward_feature_indices = [list(forward_features).index(f) for f in features_for_error]
            forward_states = forward_traj[:, forward_feature_indices].astype(np.float64)
            
            # Ensure shapes match
            min_len = min(len(backward_states), len(forward_states))
            if min_len > 0:
                backward_states = backward_states[:min_len]
                forward_states = forward_states[:min_len]
                
                # Normalize if normalization_info provided
                if normalization_info is not None:
                    for feat_idx, feat_name in enumerate(features_for_error):
                        if feat_name in normalization_info.columns:
                            # Use minmax_sym normalization: scale to [-1, 1]
                            feat_min = normalization_info.loc['min', feat_name]
                            feat_max = normalization_info.loc['max', feat_name]
                            feat_range = feat_max - feat_min
                            if feat_range > 0:
                                backward_states[:, feat_idx] = -1.0 + 2.0 * (backward_states[:, feat_idx] - feat_min) / feat_range
                                forward_states[:, feat_idx] = -1.0 + 2.0 * (forward_states[:, feat_idx] - feat_min) / feat_range
                
                # Compute L2 over time for each feature (RMS)
                # error_per_feature[f] = sqrt(mean_t((backward[t,f] - forward[t,f])^2))
                squared_errors = (backward_states - forward_states) ** 2  # [time, features]
                rms_per_feature = np.sqrt(np.mean(squared_errors, axis=0))  # [features]
                
                # Take L∞ over features (max = worst feature)
                max_feature_error = np.max(rms_per_feature)
            else:
                max_feature_error = np.nan
            
            trajectory_error_scores.append(max_feature_error)
        else:
            trajectory_error_scores.append(np.nan)
        
        # Combine
        traj_df = forward_df
        traj_df['experiment_index'] = traj_idx
        traj_df['seed_absolute_time'] = time_axis[seed_idx]
        
        all_trajectories.append(traj_df)
    
    # Combine all trajectories
    combined_df = pd.concat(all_trajectories, ignore_index=True)
    
    # Add global time axis: 0 to total_rows * dt (ignoring experiment_index)
    combined_df['time'] = np.arange(len(combined_df)) * abs(dt)
    
    # Add trajectory error score column (same value for all rows in each trajectory)
    if compute_error_score:
        # Create a mapping from experiment_index to error score
        error_score_map = {i: score for i, score in enumerate(trajectory_error_scores)}
        combined_df['trajectory_error_score'] = combined_df['experiment_index'].map(error_score_map)
        
        if verbose:
            valid_scores = [s for s in trajectory_error_scores if not np.isnan(s)]
            if valid_scores:
                print(f"  Error scores: min={min(valid_scores):.6f}, max={max(valid_scores):.6f}, mean={np.mean(valid_scores):.6f}")
    
    # Reorder columns: metadata first, then states, then controls
    metadata_cols = [
        'experiment_index', 'seed_absolute_time', 'time', 'phase', 'time_step'
    ]
    if compute_error_score:
        metadata_cols.append('trajectory_error_score')
    state_cols = [col for col in forward_features if col in combined_df.columns]
    control_cols = [col for col in control_features if col in combined_df.columns]
    
    # Arrange: metadata | states | controls
    combined_df = combined_df[metadata_cols + state_cols + control_cols]
    
    # Add file metadata as attributes (will be written as header comments)
    combined_df.attrs['source_file'] = df_name
    combined_df.attrs['backward_predictor'] = backward_predictor_specification
    combined_df.attrs['forward_predictor'] = forward_predictor_specification
    combined_df.attrs['test_horizon'] = predictor_horizon
    combined_df.attrs['dataset_sampling_dt'] = abs(dt)
    
    if verbose:
        print(f"  Generated {len(all_trajectories)} trajectories ({len(combined_df)} total rows)")
    
    return combined_df

