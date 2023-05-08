import os
import yaml
import numpy as np

# predictors config
config_testing = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'), Loader=yaml.FullLoader)

#from SI_Toolkit.Testing.Testing_Functions.Brunton_GUI import run_overfitting_test
from SI_Toolkit.Testing.Testing_Functions.get_prediction import get_prediction
from SI_Toolkit.Testing.Testing_Functions.preprocess_for_brunton import preprocess_for_brunton

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

def run_overfitting_test():

    dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(**config_testing)

    predictions_list = []
    predictors_list = config_testing['predictors_specifications_testing']
    predictor = PredictorWrapper()
    for predictor_specification in predictors_list:
        predictor.update_predictor_config_from_specification(predictor_specification=predictor_specification)
        predictions_list.append(get_prediction(dataset, predictor, **config_testing))

    overfitting_test(config_testing['features_to_plot'], titles=predictors_list,
                 ground_truth=ground_truth, predictions_list=predictions_list, time_axis=time_axis,
                 )

    dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(**config_testing)

    predictions_list = []
    predictors_list = config_testing['predictors_specifications_testing']
    predictor = PredictorWrapper()
    for predictor_specification in predictors_list:
        predictor.update_predictor_config_from_specification(predictor_specification=predictor_specification)
        predictions_list.append(get_prediction(dataset, predictor, **config_testing))

    overfitting_test(config_testing['features_to_plot'], titles=predictors_list,
                         ground_truth=ground_truth, predictions_list=predictions_list, time_axis=time_axis,
                         )

    dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(**config_testing)

    predictions_list = []
    predictors_list = config_testing['predictors_specifications_testing']
    predictor = PredictorWrapper()
    for predictor_specification in predictors_list:
        predictor.update_predictor_config_from_specification(predictor_specification=predictor_specification)
        predictions_list.append(get_prediction(dataset, predictor, **config_testing))

    overfitting_test(config_testing['features_to_plot'], titles=predictors_list,
                         ground_truth=ground_truth, predictions_list=predictions_list, time_axis=time_axis,
                         )


def overfitting_test(features, titles, ground_truth, predictions_list, time_axis):

    #choose which gp to look at
    dataset = predictions_list[0]

    #choose which feature to look at (either name or index from 0 to 5)
    feature_idx = features.index("angle")

    print(get_sqrt_MSE_at_horizon(0, True, dataset, 50, ground_truth, 0))
    print("hi")


def get_sqrt_MSE_at_horizon(feature_idx, show_all, dataset, horizon, ground_truth, current_point_at_timeaxis):


    if feature_idx == 0:
        if show_all:
            predictions_at_horizon = dataset[..., :-horizon, horizon, feature_idx]
            horizon = 25

            for i in range(len(predictions_at_horizon)):
                if (ground_truth[horizon:, feature_idx] - predictions_at_horizon[i]) > 180:
                    predictions_at_horizon[i] = predictions_at_horizon[i] + 360
                elif (ground_truth[horizon:, feature_idx] - predictions_at_horizon[i]) < -180:
                    predictions_at_horizon[i] = predictions_at_horizon[i] - 360
                else:
                    continue
            MSE_at_horizon = np.mean((ground_truth[horizon:, feature_idx] - predictions_at_horizon) ** 2)

        else:
            predictions_at_horizon = dataset[..., current_point_at_timeaxis, horizon, feature_idx]

            for i in range(len(predictions_at_horizon)):
                if (ground_truth[current_point_at_timeaxis + horizon, feature_idx] - predictions_at_horizon[i]) > 180:
                    predictions_at_horizon[i] = predictions_at_horizon[i] + 360
                elif (ground_truth[current_point_at_timeaxis + horizon, feature_idx] - predictions_at_horizon[i]) < -180:
                    predictions_at_horizon[i] = predictions_at_horizon[i] - 360
                else:
                    continue
            MSE_at_horizon = np.mean((ground_truth[current_point_at_timeaxis + horizon, feature_idx] - predictions_at_horizon) ** 2)

        sqrt_MSE_at_horizon = np.sqrt(MSE_at_horizon)



    else:
        if show_all:
            horizon = 25
            predictions_at_horizon = dataset[:-horizon, horizon, feature_idx]
            MSE_at_horizon = np.mean((ground_truth[horizon:, feature_idx] - predictions_at_horizon) ** 2)

        else:
            predictions_at_horizon = dataset[..., current_point_at_timeaxis, horizon, feature_idx]
            MSE_at_horizon = np.mean((ground_truth[current_point_at_timeaxis + horizon, feature_idx] - predictions_at_horizon) ** 2)

        sqrt_MSE_at_horizon = np.sqrt(MSE_at_horizon)

    return sqrt_MSE_at_horizon




if __name__ == '__main__':
    run_overfitting_test()



