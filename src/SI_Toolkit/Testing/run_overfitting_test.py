import os
import yaml
import numpy as np

# predictors config
config_testing = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'), Loader=yaml.FullLoader)

#from SI_Toolkit.Testing.Testing_Functions.Brunton_GUI import run_overfitting_test
from SI_Toolkit.Testing.Testing_Functions.get_prediction import get_prediction
from SI_Toolkit.Testing.Testing_Functions.preprocess_for_brunton import preprocess_for_brunton

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

#Train_GPR imports
from SI_Toolkit.GP.Models import MultiOutSGPR, MultiOutGPR

from SI_Toolkit.GP.Functions.save_and_load import get_normalized_data_for_training_overfitting_test, save_model, save_training_time, save_training_script
from SI_Toolkit.GP.Functions.plot import plot_samples, plot_test, state_space_prediction_error, plot_error
from SI_Toolkit.GP.Functions.other import reformat_data_for_vaidation, preselect_data_for_GPs
from SI_Toolkit.GP.Functions.GP_ASF import get_kernels

import os

import gpflow as gpf
import random
import tensorflow as tf

import matplotlib
from SI_Toolkit.Functions.General.load_parameters_for_training import args

def run_overfitting_test():

    #Train_GPR
    Train_GPR(100,400,0)

    #number of iterations for Testing
    iterations = 3

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

    avg_MSE = np.mean(MSEs, axis = 0)
    avg_var = np.mean(vars, axis = 0)

    avg_std = np.sqrt(avg_var)

    print(avg_MSE)
    print(avg_std)

    #plotting



def overfitting_test(features, titles, ground_truth, predictions_list, time_axis):


    convert_units_inplace(ground_truth, predictions_list, features)

    # choose which feature to look at (0 to 5). Update: don't need this anymore since it now does all features
    #feature_idx = 0

    #Show all True or False
    show_all = True

    #choose which gp to look at. Update: dont need this either
    #dataset = predictions_list[0]

    #Choose a horizon length (1 to 50)
    horizon = 25

    #Choose Current Point at Time Axis (doesnt do anything if show_all is True)
    current_point_at_timeaxis = 0

    sqrt_MSE_at_horizon = []
    var_at_horizon = []

    for i in range(len(predictions_list)):
        sqrt_MSE_at_horizon_list = []
        var_at_horizon_list = []
        dataset = predictions_list[i]
        for feature_idx in range(len(features)):
            sqrt_MSE_at_horizon_list.append(get_sqrt_MSE_at_horizon(feature_idx, show_all, dataset, horizon, ground_truth, current_point_at_timeaxis))
            var_at_horizon_list.append(get_var_at_horizon(feature_idx, show_all, dataset, horizon, ground_truth, current_point_at_timeaxis))
        sqrt_MSE_at_horizon.append(sqrt_MSE_at_horizon_list)
        var_at_horizon.append(var_at_horizon_list)

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


def Train_GPR(number_of_inducing_points, maxiter, no_of_experiments):
    GP_TYPE = 'SGPR'

    gpf.config.set_default_float(tf.float64)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF
    matplotlib.rcParams.update({'font.size': 24})

    ##  LOADING ARGUMENTS FROM CONFIG
    a = args()
    inputs = a.state_inputs + a.control_inputs
    a.wash_out_len = 0
    a.post_wash_out_len = 1
    outputs = a.outputs
    batch_size = a.batch_size

    #number_of_inducing_points = 100

    #save_dir = a.path_to_models + "SGP_{}/".format(number_of_inducing_points)
    save_dir = a.path_to_models + GP_TYPE[:-1] + "_{}/".format(number_of_inducing_points)
    #save_dir = a.path_to_models + GP_TYPE[:-1] + "_100_4/"
    print(save_dir)
    save_training_script(save_dir)

    data_train, data_val, data_test = get_normalized_data_for_training_overfitting_test(a,3)

    ## PRESELECTING DATA: See preselect_data_for_GPs docstring
    number_of_bins = 11
    training = True
    X, Y = preselect_data_for_GPs(data_train, inputs, outputs, number_of_bins, training, batch_size)

    ## DEFINING KERNELS
    kernels = get_kernels(inputs)

    # FROM PRESELECTED DATA RANDOMLY CHOOSE POINTS TO SERVE AS FIRST GUESS OF INDUCING POINTS
    sample_indices = random.sample(range(X.shape[0]), number_of_inducing_points)

    ## PLOTTING PHASE DIAGRAMS OF PRESELECTED DATA AND INDUCING POINTS GUESS.
    plot_samples(X[sample_indices], save_dir=save_dir+"info/initial_ip/")
    plot_samples(X, save_dir=save_dir+"info/training_ss/")

    ## DEFINING MULTI OUTPUT SGPR MODEL
    if GP_TYPE == 'GPR':
        model = MultiOutGPR(a)
        model.setup((X[sample_indices], Y[sample_indices]), kernels)
    elif GP_TYPE == 'SGPR':
        model = MultiOutSGPR(a)
        model.setup((X, Y), kernels, X[sample_indices])

    X_val, Y_val = reformat_data_for_vaidation(data_val, inputs)

    ## MODEL OPTIMIZATION
    #maxiter = 400   #determines how long training is, sweep maxiter choose set number of inducing points and experiments - expect MSE to go down and get stuck and var continuing
                    #inducing points: maxiter high ie 400 expect discrepency between MSE and var getting smaller, fixed no of exp start with this
                    #changing no of exp: var of var should get smaller and MSE variance, take a few different numbers of inducing points, big gp and small gp, start with this on presentation
    learning_rate = 0.08
    logf, logf_val, train_time = model.optimize("Adam", iters=maxiter, lr=learning_rate, val_data=(X_val, Y_val))
    save_training_time(train_time, save_dir)

    # PLOT FINAL INDUCING POINTS (FIXME: In current implementation one does not know which plot is for which kernel)
    # for i in len(kernels):
    #     plot_samples(model.models[i].inducing_variable.Z.numpy(), save_dir=save_dir + model.outputs[i], show=False)

    plot_error(model, maxiter, logf_val, save_dir)

    ## SAVE MODEL
    print("Saving...")
    save_model(model, save_dir, GP_TYPE)
    print("Done!")

    ## TESTING ON TEST DATA
    number_of_bins = 11
    training = False
    X, Y = preselect_data_for_GPs(data_test, inputs, outputs, number_of_bins, training, batch_size)

    test_indices = random.sample(range(X.shape[0]), 100)

    # PLOT error between true and predicted values
    errs = state_space_prediction_error(model, (X[test_indices], Y[test_indices]), save_dir=save_dir+"info/ss_error/")

    ## PLOTTING 1s CLOSED-LOOP PREDICTION FROM TEST RECORDING
    plot_test(model, data_test, closed_loop=True)




if __name__ == '__main__':
    run_overfitting_test()



