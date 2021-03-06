import numpy as np
import tensorflow as tf
import os
import timeit

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


if __name__ == '__main__':
    a = args()
    a_GP = args_GP()

    initialization = '''
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
import numpy as np
import tensorflow as tf

predictor_name = {}
horizon = {}
num_rollouts = {}

if 'Euler' in predictor_name:
    predictor = predictor_ODE_tf(horizon=horizon, dt=0.02)
elif 'GP' in predictor_name:
    predictor = predictor_autoregressive_GP(model_name=predictor_name, horizon=horizon)
else:
    predictor = predictor_autoregressive_tf(horizon=horizon, batch_size=num_rollouts, net_name=predictor_name)

initial_state = tf.random.uniform(shape=[num_rollouts, 6], dtype=tf.float32)
Q = tf.random.uniform(shape=[num_rollouts, horizon, 1], dtype=tf.float32)

predictor.predict_tf(initial_state, Q)
predictor.update_internal_state(Q)
predictor.predict_tf(initial_state, Q)
'''

    code = '''\
predictor.predict_tf(initial_state, Q)
'''

    norm_info = load_normalization_info(a_GP.path_to_normalization_info)

    save_dir = '/'.join(a.path_to_models.split('/')[:-2]) + '/Tests/Prediction_times'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    horizons = [k for k in range(2, 51, 2)]
    rollouts = [3000]
    results = np.zeros(shape=[len(horizons)+1, len(a.tests)])

    for r in rollouts:
        i = 1
        for h in horizons:
            print("HORIZONS: {}".format(h))
            j = 0
            for predictor_name in a.tests:
                prediction_time = timeit.timeit(code, number=100, setup=initialization.format('"'+predictor_name+'"', h, r))/ 100.0

                results[i, j] = prediction_time * 1000  # convert to milliseconds
                print("{}: {}".format(predictor_name, prediction_time))
                j += 1

            i += 1

        np.savetxt(save_dir + "/prediction_times_rollouts_{}.csv".format("3000"), results, delimiter=", ")

        plt.plot(rollouts, results)
        # plt.legend(a.tests)
        plt.grid()
        plt.title(str(h))
        plt.xlabel("Rollouts")
        plt.ylabel("Prediction time [ms]")
        plt.savefig(save_dir + '/plot_horizon_{}.pdf'.format(h))
        plt.show()
