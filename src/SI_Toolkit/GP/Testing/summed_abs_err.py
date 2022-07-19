import numpy as np
import tensorflow as tf
import os

from SI_Toolkit.Testing.Parameters_for_testing import args
from SI_Toolkit.Testing.Testing_Functions.preprocess_for_brunton import preprocess_for_brunton
from SI_Toolkit.Testing.Testing_Functions.get_prediction import get_prediction
from SI_Toolkit.load_and_normalize import normalize_numpy_array, load_normalization_info
from SI_Toolkit.GP.Parameters import args as args_GP
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange



try:
    from SI_Toolkit_ASF.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, CONTROL_INDICES
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')

from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP

matplotlib.rcParams.update({'font.size': 24})

def summed_normed_error(Y_pred, Y):

    errs = np.linalg.norm(Y_pred - Y, axis=1)

    return np.sum(errs)

if __name__ == '__main__':
    # NOTE: quickly coded and very slow

    a = args()
    a_GP = args_GP()

    norm_info = load_normalization_info(a_GP.path_to_normalization_info)

    save_dir = '/'.join(a.path_to_models.split('/')[:-2]) + '/Tests/Model_error'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    horizons = [1, 10, 20, 30, 40, 50]
    mae = np.zeros(shape=[len(horizons), len(a.tests)])
    std = np.zeros(shape=[len(horizons), len(a.tests)])

    test_files = os.listdir(a.default_locations_for_testfile[0])
    dataset_list = []
    ground_truth_list = []
    for test_file in test_files:
        a.test_file = test_file
        dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(a)
        dataset_list.append(dataset)
        ground_truth_list.append(ground_truth)

    predictors = []
    for predictor_name in a.tests:
        if 'Euler' in predictor_name:
            predictors.append(predictor_ODE_tf(horizon=50, dt=0.02))
        elif 'GP' in predictor_name:
            predictors.append(predictor_autoregressive_GP(model_name=predictor_name, horizon=50))
        else:
            predictors.append(predictor_autoregressive_tf(horizon=50, batch_size=1, net_name=predictor_name))

    i, j = 0, 0
    for predictor in predictors:
        predictor_name = a.tests[i]
        print("PREDICTOR: {}".format(predictor_name))
        errs = np.zeros(shape=[len(horizons), len(test_files)])
        for k in range(len(dataset_list)):
            a.test_file = test_files[k]
            dataset = dataset_list[k]
            ground_truth = ground_truth_list[k]

            states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]

            Q = dataset[CONTROL_INPUTS].to_numpy()
            Q_array = [Q[..., i:-a.test_max_horizon + i, :] for i in range(a.test_max_horizon)]
            Q_array = np.stack(Q_array, axis=1)

            output_array = np.zeros([a.test_len, a.test_max_horizon + 1, len(a.features)],
                                    dtype=np.float32)

            stateful_components = ['RNN', 'GRU', 'LSTM']
            if any(stateful_component in predictor_name for stateful_component in stateful_components):
                output = np.zeros([a.test_len, a.test_max_horizon + 1, len(a.features)],
                                  dtype=np.float32)
                for timestep in trange(a.test_len):
                    Q_current_timestep = Q_array[np.newaxis, timestep, :, :]
                    s0 = states_0[np.newaxis, timestep, :]
                    output[timestep, :, :] = predictor.predict(s0, Q_current_timestep)
                    if a.test_max_horizon > 1:
                        predictor.update_internal_state(Q_current_timestep[:, np.newaxis, 1, :])

                output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in a.features]]
            else:
                output = predictor.predict(states_0, Q_array)
                output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in a.features]]

            j = 0
            for horizon in horizons:
                print("HORIZON: {}".format(horizon))
                ground_truth = dataset[STATE_VARIABLES].to_numpy()[horizon:a.test_len+horizon, :]
                predictions = output_array[:, horizon, :]

                predictions = normalize_numpy_array(predictions,
                                                    features=STATE_VARIABLES,
                                                    normalization_info=norm_info)
                ground_truth = normalize_numpy_array(ground_truth,
                                                     features=STATE_VARIABLES,
                                                     normalization_info=norm_info)

                err = summed_normed_error(predictions, ground_truth)
                errs[j, k] = err
                j += 1

        mae[:, i] = errs.mean(axis=1)
        std[:, i] = errs.std(axis=1)
        i += 1

    np.savetxt(save_dir + "/mae_results.csv", mae, delimiter=", ")
    np.savetxt(save_dir + "/std_results.csv", std, delimiter=", ")

    width = 1.8
    bars = []

    tests = ['RNN', 'SGP', 'Euler']

    horizons = [1, 10, 20, 30, 40, 50]

    plt.figure(figsize=(12, 10))
    horizons = [h-2 for h in horizons]
    for i in range(len(tests)):
        b = plt.bar(horizons, mae[:, i], width=width, yerr=std[:, i], capsize=5)
        bars.append(b)
        horizons = [h + 2 for h in horizons]

    plt.xticks([1, 10, 20, 30, 40, 50])
    plt.xlabel("Horizon steps")
    plt.ylabel('Error')

    plt.legend([bar for bar in bars], tests)
    plt.tight_layout()
    plt.grid()
    plt.ylim(0,220)
    plt.savefig('./error_bars.png', dpi=400)
    plt.show()