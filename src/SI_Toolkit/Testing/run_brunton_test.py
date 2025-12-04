import os
import numpy as np
from SI_Toolkit.load_and_normalize import load_yaml

# predictors config
config_testing = load_yaml(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'))

from SI_Toolkit.Testing.Testing_Functions.Brunton_GUI import run_test_gui
from SI_Toolkit.Testing.Testing_Functions.get_prediction import get_prediction
from SI_Toolkit.Testing.Testing_Functions.preprocess_for_brunton import preprocess_for_brunton

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper


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
