import os
import numpy as np
from SI_Toolkit.load_and_normalize import load_yaml

# predictors config
config_testing = load_yaml(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'))

from SI_Toolkit.Testing.Testing_Functions.Brunton_GUI import run_test_gui
from SI_Toolkit.Testing.Testing_Functions.get_prediction import get_prediction
from SI_Toolkit.Testing.Testing_Functions.preprocess_for_brunton import preprocess_for_brunton

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper


def _compute_and_print_trajectory_rmse(title: str, predictions, ground_truth, max_horizon: int, stride: int = 1):
    """
    Compute and print trajectory RMSE for backward predictions.
    
    Args:
        title: Predictor name for display
        predictions: Tuple (predictions_array, dt_predictions, forward_prediction) from get_prediction()
        ground_truth: Tuple (data_array [N, features], feature_names) from preprocess_for_brunton
        max_horizon: Maximum prediction horizon
        stride: Test stride used
    """
    if predictions is None:
        return
    
    # Extract data array and feature names from ground_truth tuple
    if isinstance(ground_truth, (list, tuple)) and len(ground_truth) >= 2:
        gt_data = ground_truth[0]
        gt_feature_names = ground_truth[1]
    else:
        gt_data = ground_truth[0] if isinstance(ground_truth, (list, tuple)) else ground_truth
        gt_feature_names = None
    
    # Handle dict (param sweep) vs tuple (single prediction)
    if isinstance(predictions, dict):
        print(f"\n{'='*60}")
        print(f"TRAJECTORY RMSE: {title}")
        print(f"{'='*60}")
        for param_val, pred_tuple in predictions.items():
            if pred_tuple is not None and len(pred_tuple) >= 1:
                _compute_rmse_for_tuple(pred_tuple, gt_data, gt_feature_names, max_horizon, stride, label=f"  {param_val:.4f}")
        return
    
    print(f"\n{'='*60}")
    print(f"TRAJECTORY RMSE: {title}")
    print(f"{'='*60}")
    _compute_rmse_for_tuple(predictions, gt_data, gt_feature_names, max_horizon, stride, label="  ")


def _compute_rmse_for_tuple(pred_tuple, ground_truth, gt_feature_names, max_horizon: int, stride: int, label: str = ""):
    """
    Compute RMSE for a single prediction tuple.
    
    Tuple structure from get_prediction():
      [0]: output array [test_len, horizon+1, features]
      [1]: predictor_output_features (list of feature names)
      [2]: dt_predictions (scalar or array)
      [3]: forward_prediction (optional)
    
    For backward predictions:
      - predictions_array[k, 0, :] = anchor state at dataset index (max_horizon + k*stride)
      - predictions_array[k, h, :] = predicted state h steps BEFORE anchor
      - Ground truth for horizon h is at dataset index (max_horizon + k*stride - h)
    
    Args:
        pred_tuple: Prediction tuple from get_prediction()
        ground_truth: Ground truth data array [N, all_features]
        gt_feature_names: Array of ground truth feature names (to map predictor outputs to GT columns)
        max_horizon: Maximum prediction horizon
        stride: Test stride
        label: Label prefix for output
    """
    if pred_tuple is None or len(pred_tuple) < 1:
        print(f"{label}(no predictions)")
        return
    
    predictions_array = pred_tuple[0]  # [test_len, horizon+1, features]
    predictor_output_features = pred_tuple[1] if len(pred_tuple) > 1 else None  # Feature names
    dt_predictions = pred_tuple[2] if len(pred_tuple) > 2 else None
    forward_prediction = pred_tuple[3] if len(pred_tuple) > 3 else None
    
    # Backward mode: dt < 0, Forward mode: dt > 0
    # dt_predictions can be scalar or array - extract scalar value
    if dt_predictions is not None:
        dt_val = float(np.atleast_1d(dt_predictions).flatten()[0])
        backward_mode = dt_val < 0
    else:
        backward_mode = False
    
    test_len = predictions_array.shape[0]
    horizon_plus_1 = predictions_array.shape[1]
    num_features = predictions_array.shape[2]
    horizon = horizon_plus_1 - 1  # Actual prediction horizon (excluding anchor)
    
    # Map predictor output features to ground truth columns
    # This handles the case where ground truth has more columns than predictions
    gt_feature_indices = None
    if gt_feature_names is not None and predictor_output_features is not None:
        gt_feature_names_list = list(gt_feature_names)
        gt_feature_indices = []
        for pred_feat in predictor_output_features:
            pred_feat_str = str(pred_feat)
            if pred_feat_str in gt_feature_names_list:
                gt_feature_indices.append(gt_feature_names_list.index(pred_feat_str))
            else:
                # Feature not found - this is an error
                print(f"{label}Warning: predictor output feature '{pred_feat_str}' not found in ground truth")
                gt_feature_indices = None
                break
    
    # Build aligned ground truth array [test_len, horizon, features]
    # Note: After NaN expansion in get_prediction, predictions_array[k] corresponds to 
    # anchor at dataset index (max_horizon + k) for backward, or k for forward.
    # The stride is already incorporated via NaN padding, so we don't multiply by stride here.
    gt_aligned = np.full((test_len, horizon, num_features), np.nan, dtype=np.float32)
    
    for k in range(test_len):
        if backward_mode:
            # For backward: anchor at dataset index (max_horizon + k), predictions go backward
            anchor_idx = max_horizon + k
            for h in range(1, horizon_plus_1):  # h=1..horizon
                gt_idx = anchor_idx - h
                if 0 <= gt_idx < len(ground_truth):
                    if gt_feature_indices is not None:
                        gt_aligned[k, h-1, :] = ground_truth[gt_idx, gt_feature_indices]
                    else:
                        gt_aligned[k, h-1, :] = ground_truth[gt_idx, :num_features]
        else:
            # For forward: anchor at dataset index k, predictions go forward
            for h in range(1, horizon_plus_1):  # h=1..horizon
                gt_idx = k + h
                if 0 <= gt_idx < len(ground_truth):
                    if gt_feature_indices is not None:
                        gt_aligned[k, h-1, :] = ground_truth[gt_idx, gt_feature_indices]
                    else:
                        gt_aligned[k, h-1, :] = ground_truth[gt_idx, :num_features]
    
    # Extract predictions for horizons 1..horizon (skip anchor at index 0)
    pred_aligned = predictions_array[:, 1:, :]  # [test_len, horizon, features]
    
    # Compute errors
    errors = pred_aligned - gt_aligned  # [test_len, horizon, features]
    
    # Per-horizon RMSE (averaged over test samples and features)
    per_horizon_mse = np.nanmean(errors ** 2, axis=(0, 2))  # [horizon]
    per_horizon_rmse = np.sqrt(per_horizon_mse)
    
    # Overall RMSE (averaged over all dimensions)
    overall_mse = np.nanmean(errors ** 2)
    overall_rmse = np.sqrt(overall_mse)
    
    # Summary
    mode_str = "Backward" if backward_mode else "Forward"
    print(f"{label}{mode_str} trajectory (horizon=1..{horizon}, {test_len} samples, {num_features} features):")
    print(f"{label}  Overall RMSE:             {overall_rmse:.6f}")
    print(f"{label}  RMSE at horizon 1:        {per_horizon_rmse[0]:.6f}")
    print(f"{label}  RMSE at horizon {horizon}:       {per_horizon_rmse[-1]:.6f}")
    print(f"{label}  RMSE max over horizons:   {np.nanmax(per_horizon_rmse):.6f}")
    print(f"{label}  RMSE mean over horizons:  {np.nanmean(per_horizon_rmse):.6f}")
    
    # Forward reconstruction RMSE (if available, only for backward mode)
    # forward_prediction structure: [fwd_output, fwd_features, fwd_dt, from_all_horizons]
    if backward_mode and forward_prediction is not None and len(forward_prediction) >= 1:
        fwd_data = forward_prediction[0] if len(forward_prediction) > 0 else None
        
        # Handle dict format (from_all_horizons=True) vs array format
        if isinstance(fwd_data, dict):
            # Use max_horizon key from the dict
            fwd_array = fwd_data.get(horizon, None)
            if fwd_array is None and len(fwd_data) > 0:
                # Fallback to the largest available horizon
                fwd_array = fwd_data.get(max(fwd_data.keys()), None)
        else:
            fwd_array = fwd_data
        
        if fwd_array is not None and hasattr(fwd_array, 'shape') and fwd_array.shape[0] > 0:
            # Forward reconstruction: [test_len, fwd_horizon+1, features]
            # The final step (index -1) should reconstruct back to the anchor
            fwd_final = fwd_array[:, -1, :]  # Reconstructed anchor [test_len, features]
            
            # Compare to ground truth anchors
            gt_anchors = np.full((min(test_len, fwd_array.shape[0]), num_features), np.nan)
            for k in range(gt_anchors.shape[0]):
                anchor_idx = max_horizon + k  # No stride multiplication - already in full indices
                if anchor_idx < len(ground_truth):
                    if gt_feature_indices is not None:
                        gt_anchors[k] = ground_truth[anchor_idx, gt_feature_indices]
                    else:
                        gt_anchors[k] = ground_truth[anchor_idx, :num_features]
            
            fwd_error = fwd_final[:len(gt_anchors)] - gt_anchors
            fwd_rmse = np.sqrt(np.nanmean(fwd_error ** 2))
            print(f"{label}  Forward→Anchor RMSE:      {fwd_rmse:.6f}")


def run_brunton_test(test_hls=False):

    if not test_hls:
        try:
            test_hls = config_testing['test_hls']
        except KeyError:
            test_hls = False

    dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(**config_testing)

    # Check for parameter sweep configuration
    param_sweep_config = config_testing.get('param_sweep', {})
    param_sweep_enabled = param_sweep_config.get('enabled', False)
    
    param_values = None
    param_name = None
    true_param_idx = None  # Index of true parameter value in param_values
    
    if param_sweep_enabled:
        param_name = param_sweep_config.get('param_name', 'mu')
        param_range = param_sweep_config.get('param_range', [0.3, 1.1])
        param_step = param_sweep_config.get('param_step', 0.1)
        
        # Generate parameter values
        param_values = np.arange(param_range[0], param_range[1] + param_step / 2, param_step)
        param_values = np.round(param_values, decimals=6)  # Avoid floating point artifacts
        
        # Extract true parameter value from dataset (if available)
        true_param_value = None
        if param_name in dataset.columns:
            # Use mean value (in case it varies, though typically constant)
            true_param_value = np.round(dataset[param_name].mean(), decimals=6)
            
            # Check if true value is already in param_values (within tolerance)
            tolerance = param_step / 10
            close_idx = np.where(np.abs(param_values - true_param_value) < tolerance)[0]
            
            if len(close_idx) > 0:
                # True value is close to an existing value
                true_param_idx = int(close_idx[0])
            else:
                # Insert true value into param_values at correct position
                insert_pos = np.searchsorted(param_values, true_param_value)
                param_values = np.insert(param_values, insert_pos, true_param_value)
                true_param_idx = int(insert_pos)
            
            print(f"True {param_name} value from data: {true_param_value:.6f} (index {true_param_idx})")
        
        print(f"Parameter sweep enabled: {param_name} = {param_values.tolist()}")

    predictions_list = []
    predictors_list = config_testing['predictors_specifications_testing']
    predictor = PredictorWrapper()
    titles = []
    
    for predictor_specification_raw in predictors_list:
        if ';' in predictor_specification_raw:
            predictor_specification, forward_predictor_specification = [
                part.strip() for part in predictor_specification_raw.split(';', 1)
            ]
        else:
            predictor_specification = predictor_specification_raw.strip()
            forward_predictor_specification = None

        forward_predictor = None
        if forward_predictor_specification:
                forward_predictor = PredictorWrapper()
                forward_predictor.update_predictor_config_from_specification(
                    predictor_specification=forward_predictor_specification
                )

        routine = "autoregressive"
        
        # Check for S: prefix (simple evaluation)
        if predictor_specification[:2] == 'S:':
            routine = "simple evaluation"
            predictor_specification = predictor_specification[2:]
        
        # Check for B: prefix (backward trajectory)
        if predictor_specification[:2] == 'B:':
            routine = routine + "_backward"
            predictor_specification = predictor_specification[2:]
        
        predictor.update_predictor_config_from_specification(predictor_specification=predictor_specification)
        
        if param_sweep_enabled and param_values is not None:
            # Generate predictions for each parameter value
            predictions_by_param = {}
            for pv in param_values:
                print(f"  Computing predictions for {param_name}={pv:.4f}...")
                predictions_by_param[pv] = get_prediction(
                    dataset, predictor, dataset_sampling_dt, routine,
                    forward_predictor=forward_predictor,
                    param_name=param_name,
                    param_value=pv,
                    **config_testing
                )
            predictions_list.append(predictions_by_param)
        else:
            # Standard single prediction
            predictions_list.append(get_prediction(
                dataset, predictor, dataset_sampling_dt, routine,
                forward_predictor=forward_predictor,
                **config_testing
            ))
        
        titles.append(predictor_specification)
        
        if test_hls and predictor.predictor_type == 'neural':
            if param_sweep_enabled and param_values is not None:
                predictions_by_param_hls = {}
                for pv in param_values:
                    print(f"  Computing HLS predictions for {param_name}={pv:.4f}...")
                    predictions_by_param_hls[pv] = get_prediction(
                        dataset, predictor, dataset_sampling_dt, routine,
                        forward_predictor=forward_predictor,
                        param_name=param_name,
                        param_value=pv,
                        hls=True,
                        **config_testing
                    )
                predictions_list.append(predictions_by_param_hls)
            else:
                predictions_list.append(get_prediction(
                    dataset, predictor, dataset_sampling_dt, routine,
                    forward_predictor=forward_predictor,
                    hls=True,
                    **config_testing
                ))
            titles.append('HLS:'+predictor_specification)

    # Print trajectory RMSE for each predictor
    max_horizon = config_testing.get('test_max_horizon', 50)
    test_stride = config_testing.get('test_stride', 1)
    
    print("\n" + "=" * 80)
    print("BRUNTON TEST - TRAJECTORY RMSE DIAGNOSTICS")
    print("=" * 80)
    print(f"Max horizon: {max_horizon}, Stride: {test_stride}")
    
    for title, predictions in zip(titles, predictions_list):
        _compute_and_print_trajectory_rmse(title, predictions, ground_truth, max_horizon, test_stride)
    
    print("\n" + "=" * 80)
    print("Launching GUI...")
    print("=" * 80 + "\n")

    run_test_gui(
        titles=titles,
        ground_truth=ground_truth,
        predictions_list=predictions_list,
        time_axis=time_axis,
        param_values=param_values,
        param_name=param_name,
        true_param_idx=true_param_idx,
    )


if __name__ == '__main__':
    run_brunton_test()
