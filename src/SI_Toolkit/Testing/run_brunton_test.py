import os
import yaml

from SI_Toolkit.Testing.Testing_Functions.Brunton_GUI import run_test_gui
from SI_Toolkit.Testing.Testing_Functions.get_prediction import get_prediction, get_predictor
from SI_Toolkit.Testing.Testing_Functions.preprocess_for_brunton import preprocess_for_brunton

# predictors config
config = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_predictors.yml'), 'r'), Loader=yaml.FullLoader)
config_testing = config['testing']

def run_brunton_test():

    dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(config_testing)

    predictions_list = []
    predictors_list = config_testing['predictor_names_testing']
    for test_idx in range(len(predictors_list)):
        predictor = get_predictor(predictor_specification=predictors_list[test_idx], config_predictors=config)
        predictions_list.append(get_prediction(config_testing, dataset, predictor))

    run_test_gui(config_testing['FEATURES_TO_PLOT'], titles=predictors_list,
                 ground_truth=ground_truth, predictions_list=predictions_list, time_axis=time_axis,
                 )


if __name__ == '__main__':
    run_brunton_test()
