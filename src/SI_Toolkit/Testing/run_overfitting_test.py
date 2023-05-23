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

    #number of iterations
    iterations = 3

    #dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(**config_testing) #put here for run speed

    MSEs = []
    vars = []

    for i in range(iterations):

        dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(**config_testing)

        predictions_list = []
        predictors_list = config_testing['predictors_specifications_testing']
        predictor = PredictorWrapper()
        for predictor_specification in predictors_list:
            predictor.update_predictor_config_from_specification(predictor_specification=predictor_specification)
            predictions_list.append(get_prediction(dataset, predictor, **config_testing))

        MSE, var = overfitting_test(config_testing['features_to_plot'], titles=predictors_list,
                 ground_truth=ground_truth, predictions_list=predictions_list, time_axis=time_axis,
                 )

        MSEs.append(MSE)
        vars.append(var)

    avg_MSE = np.mean(MSEs)
    avg_var = np.mean(vars)

    print(avg_MSE)
    print(avg_var)

def overfitting_test(features, titles, ground_truth, predictions_list, time_axis):


    convert_units_inplace(ground_truth, predictions_list, features)

    # choose which feature to look at (0 to 5)
    feature_idx = 0

    #Show all True or False
    show_all = True

    #choose which gp to look at
    dataset = predictions_list[0]

    #Choose a horizon length (1 to 50)
    horizon = 25

    #Choose Current Point at Time Axis (doesnt do anything if show_all is True)
    current_point_at_timeaxis = 0

    sqrt_MSE_at_horizon = get_sqrt_MSE_at_horizon(feature_idx, show_all, dataset, horizon, ground_truth, current_point_at_timeaxis)
    var_at_horizon = get_var_at_horizon(feature_idx, show_all, dataset, horizon, ground_truth, current_point_at_timeaxis)

    print("Square Root of MSE at Horizon:", sqrt_MSE_at_horizon)
    print("Variance at Horizon:", var_at_horizon)

    return sqrt_MSE_at_horizon, var_at_horizon




def get_sqrt_MSE_at_horizon(feature_idx, show_all, dataset, horizon, ground_truth, current_point_at_timeaxis):


    if feature_idx == 0:
        if show_all:
            predictions_at_horizon = dataset[..., :-horizon, horizon, feature_idx]

            for i in range(len(predictions_at_horizon)):
                for j in range(len(predictions_at_horizon[i])):
                    if (ground_truth[horizon + j, feature_idx] - predictions_at_horizon[i][j]) > 180:
                        predictions_at_horizon[i][j] = predictions_at_horizon[i][j] + 360
                    elif (ground_truth[horizon + j, feature_idx] - predictions_at_horizon[i][j]) < -180:
                        predictions_at_horizon[i][j] = predictions_at_horizon[i][j] - 360
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
            predictions_at_horizon = dataset[..., :-horizon, horizon, feature_idx]
            MSE_at_horizon = np.mean((ground_truth[horizon:, feature_idx] - predictions_at_horizon) ** 2)

        else:
            predictions_at_horizon = dataset[..., current_point_at_timeaxis, horizon, feature_idx]
            MSE_at_horizon = np.mean((ground_truth[current_point_at_timeaxis + horizon, feature_idx] - predictions_at_horizon) ** 2)

        sqrt_MSE_at_horizon = np.sqrt(MSE_at_horizon)

    return sqrt_MSE_at_horizon




def get_mean_at_horizon(feature_idx, show_all, dataset, horizon, ground_truth, current_point_at_timeaxis):

    if feature_idx == 0:
        if show_all:
            predictions_at_horizon = dataset[..., :-horizon, horizon, feature_idx]

            for i in range(len(predictions_at_horizon)):
                for j in range(len(predictions_at_horizon[i])):
                    if (ground_truth[horizon + j, feature_idx] - predictions_at_horizon[i][j]) > 180:
                        predictions_at_horizon[i][j] = predictions_at_horizon[i][j] + 360
                    elif (ground_truth[horizon + j, feature_idx] - predictions_at_horizon[i][j]) < -180:
                        predictions_at_horizon[i][j] = predictions_at_horizon[i][j] - 360
                    else:
                        continue
            mean_at_horizon = np.mean(predictions_at_horizon, axis = 0)     #returns an array of means at each timestep for the given horizon

        else:
            predictions_at_horizon = dataset[..., current_point_at_timeaxis, horizon, feature_idx]

            for i in range(len(predictions_at_horizon)):
                if (ground_truth[current_point_at_timeaxis + horizon, feature_idx] - predictions_at_horizon[i]) > 180:
                    predictions_at_horizon[i] = predictions_at_horizon[i] + 360
                elif (ground_truth[current_point_at_timeaxis + horizon, feature_idx] - predictions_at_horizon[
                    i]) < -180:
                    predictions_at_horizon[i] = predictions_at_horizon[i] - 360
                else:
                    continue
            mean_at_horizon = np.mean(predictions_at_horizon)       #returns the mean at the horizon

    else:
        if show_all:
            predictions_at_horizon = dataset[..., :-horizon, horizon, feature_idx]
            mean_at_horizon = np.mean(predictions_at_horizon, axis = 0)     #returns an array of means at each timestep for the given horizon

        else:
            predictions_at_horizon = dataset[..., current_point_at_timeaxis, horizon, feature_idx]
            mean_at_horizon = np.mean(predictions_at_horizon)       #returns the mean at the horizon

    return mean_at_horizon




def get_var_at_horizon(feature_idx, show_all, dataset, horizon, ground_truth, current_point_at_timeaxis):

    means = get_mean_at_horizon(feature_idx, show_all, dataset, horizon, ground_truth, current_point_at_timeaxis)

    if feature_idx == 0:
        if show_all:
            predictions_at_horizon = dataset[..., :-horizon, horizon, feature_idx]

            for i in range(len(predictions_at_horizon)):
                for j in range(len(predictions_at_horizon[i])):
                    if (ground_truth[horizon + j, feature_idx] - predictions_at_horizon[i][j]) > 180:
                        predictions_at_horizon[i][j] = predictions_at_horizon[i][j] + 360
                    elif (ground_truth[horizon + j, feature_idx] - predictions_at_horizon[i][j]) < -180:
                        predictions_at_horizon[i][j] = predictions_at_horizon[i][j] - 360
                    else:
                        continue
            var_at_horizon = np.mean(np.var(predictions_at_horizon, axis = 0))

        else:
            predictions_at_horizon = dataset[..., current_point_at_timeaxis, horizon, feature_idx]

            for i in range(len(predictions_at_horizon)):
                if (ground_truth[current_point_at_timeaxis + horizon, feature_idx] - predictions_at_horizon[i]) > 180:
                    predictions_at_horizon[i] = predictions_at_horizon[i] + 360
                elif (ground_truth[current_point_at_timeaxis + horizon, feature_idx] - predictions_at_horizon[
                    i]) < -180:
                    predictions_at_horizon[i] = predictions_at_horizon[i] - 360
                else:
                    continue
            var_at_horizon = np.var(means - predictions_at_horizon)

    else:
        if show_all:
            predictions_at_horizon = dataset[..., :-horizon, horizon, feature_idx]
            var_at_horizon = np.var(means - predictions_at_horizon)

        else:
            predictions_at_horizon = dataset[..., current_point_at_timeaxis, horizon, feature_idx]
            var_at_horizon = np.var(means - predictions_at_horizon)

    return var_at_horizon


def convert_units_inplace(ground_truth, predictions_list, features):

    # Convert ground truth
    for feature in features:
        feature_idx = features.index(feature)

        if feature == 'angle':
            ground_truth[:, feature_idx] *= 180.0 / np.pi
        elif feature == 'angleD':
            ground_truth[:, feature_idx] *= 180.0 / np.pi
        elif feature == 'angle_cos':
            pass
        elif feature == 'angle_sin':
            pass
        elif feature == 'position':
            ground_truth[:, feature_idx] *= 100.0
        elif feature == 'positionD':
            ground_truth[:, feature_idx] *= 100.0
        else:
            pass

    # Convert predictions
    for i in range(len(predictions_list)):
        for feature in features:
            feature_idx = features.index(feature)

            predictions_array = predictions_list[i]

            if feature == 'angle':
                predictions_array[..., feature_idx] *= 180.0/np.pi
            elif feature == 'angleD':
                predictions_array[..., feature_idx] *= 180.0 / np.pi
            elif feature == 'angle_cos':
                pass
            elif feature == 'angle_sin':
                pass
            elif feature == 'position':
                predictions_array[..., feature_idx] *= 100.0
            elif feature == 'positionD':
                predictions_array[..., feature_idx] *= 100.0
            else:
                pass

            predictions_list[i] = predictions_array





if __name__ == '__main__':
    run_overfitting_test()



