import numpy as np
import tensorflow as tf
import os

from SI_Toolkit.Testing.Parameters_for_testing import args
from SI_Toolkit.Testing.Testing_Functions.preprocess_for_brunton import preprocess_for_brunton
from SI_Toolkit.Testing.Testing_Functions.get_prediction import get_prediction
from SI_Toolkit.load_and_normalize import normalize_numpy_array, load_normalization_info
from SI_Toolkit.GP.Parameters import args as args_GP
import matplotlib.pyplot as plt
from tqdm import trange



try:
    from SI_Toolkit_ASF_global.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, CONTROL_INDICES
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')

from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP


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

    horizons = range(0, 51)
    results = np.zeros(shape=[len(horizons)+1, len(a.tests)])

    np.savetxt(save_dir + "/error_results.csv", results, delimiter=",")

    i = 1
    for horizon in horizons[1:]:
        a.test_max_horizon = horizon
        print("HORIZON: {}".format(horizon))
        j = 0
        for predictor_name in a.tests:
            if 'EulerTF' in predictor_name:
                predictor = predictor_ODE_tf(horizon=horizon, dt=0.02)
            elif 'GP' in predictor_name:
                predictor = predictor_autoregressive_GP(model_name=predictor_name, horizon=horizon)
            else:
                predictor = predictor_autoregressive_tf(horizon=horizon, batch_size=1, net_name=predictor_name)

            a.test_start_idx = 50 - horizon

            test_files = os.listdir(a.default_locations_for_testfile[0])
            avg_err = 0

            for test_file in test_files:
                a.test_file = test_file
                dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(a)

                states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]

                Q = dataset[CONTROL_INPUTS].to_numpy()
                Q_array = [Q[..., i:-a.test_max_horizon + i, :] for i in range(a.test_max_horizon)]
                Q_array = np.stack(Q_array, axis=1)

                output_array = np.zeros([a.test_len, a.test_max_horizon + 1, len(a.features)],
                                        dtype=np.float32)

                stateful_components = ['RNN', 'GRU', 'LSTM', 'GP']
                if any(stateful_component in predictor_name for stateful_component in stateful_components):
                    mode = 'sequential'
                else:
                    mode = 'batch'

                if mode == 'batch':
                    output = predictor.predict(states_0, Q_array)
                    output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in a.features]]

                else:

                    output = np.zeros([a.test_len, a.test_max_horizon + 1, len(a.features)],
                                      dtype=np.float32)
                    for timestep in trange(a.test_len):
                        Q_current_timestep = Q_array[np.newaxis, timestep, :, :]
                        s0 = states_0[np.newaxis, timestep, :]
                        output[timestep, :, :] = predictor.predict(s0, Q_current_timestep)
                        if a.test_max_horizon > 1:
                            predictor.update_internal_state(Q_current_timestep[:, np.newaxis, 1, :])

                    output_array[:, :, :] = output[..., [STATE_INDICES.get(key) for key in a.features]]

                ground_truth = dataset[STATE_VARIABLES].to_numpy()[horizon:, :]

                predictions = output_array[:, horizon, :]

                predictions = normalize_numpy_array(predictions,
                                                    features=STATE_VARIABLES,
                                                    normalization_info=norm_info)
                ground_truth = normalize_numpy_array(ground_truth,
                                                     features=STATE_VARIABLES,
                                                     normalization_info=norm_info)

                err = summed_normed_error(predictions, ground_truth)
                avg_err += err

            avg_err /= len(test_files)

            results[i, j] = avg_err
            print("{}: {}".format(predictor_name, avg_err))
            j += 1

        i += 1

    np.savetxt(save_dir + "/error_results.csv", results, delimiter=", ")

    plt.plot(np.linspace(0, horizons[-1], len(horizons)), results)
    plt.legend(a.tests)
    plt.grid()
    plt.xlabel("Horizon [s]")
    plt.ylabel("Error")
    plt.savefig(save_dir + '/plot.pdf')
    plt.show()
