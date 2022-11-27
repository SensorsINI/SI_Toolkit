import os
import yaml

# predictors config
config_testing = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'), Loader=yaml.FullLoader)

from SI_Toolkit.Testing.Testing_Functions.Brunton_GUI import run_test_gui
from SI_Toolkit.Testing.Testing_Functions.get_prediction import get_prediction
from SI_Toolkit.Testing.Testing_Functions.preprocess_for_brunton import preprocess_for_brunton

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

def run_brunton_test():

    dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(**config_testing)

    predictions_list = []
    predictors_list = config_testing['predictors_specifications_testing']
    predictor = PredictorWrapper()
    for predictor_specification in predictors_list:
        predictor.update_predictor_config_from_specification(predictor_specification=predictor_specification)
        predictions_list.append(get_prediction(dataset, predictor, **config_testing))

    run_test_gui(titles=predictors_list,
                 ground_truth=ground_truth, predictions_list=predictions_list, time_axis=time_axis,
                 shift_labels=config_testing['SHIFT_LABELS']
                 )


if __name__ == '__main__':
    run_brunton_test()
