import os
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
        predictions_list.append(get_prediction(dataset, predictor, dataset_sampling_dt, routine, forward_predictor=forward_predictor, **config_testing))
        titles.append(predictor_specification)
        if test_hls and predictor.predictor_type == 'neural':
            predictions_list.append(get_prediction(dataset, predictor, dataset_sampling_dt, routine, forward_predictor=forward_predictor, **config_testing, hls=True))
            titles.append('HLS:'+predictor_specification)

    run_test_gui(titles=titles,
                 ground_truth=ground_truth, predictions_list=predictions_list, time_axis=time_axis
                 )


if __name__ == '__main__':
    run_brunton_test()
