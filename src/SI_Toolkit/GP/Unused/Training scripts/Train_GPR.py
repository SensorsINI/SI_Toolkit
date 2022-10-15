from SI_Toolkit.GP.Models import MultiOutGPR, save_model, load_model, plot_samples, plot_test, state_space_pred_err

import os
import timeit

import gpflow as gpf
import random
import tensorflow as tf
import numpy as np

from SI_Toolkit.GP import args
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles, load_normalization_info, normalize_df
from SI_Toolkit.GP.DataSelector import DataSelector


gpf.config.set_default_float(tf.float64)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF

##  LOADING ARGUMENTS FROM CONFIG
a = args()

## LOADING NORMALIZATION INFO AND DATA
norm_info = load_normalization_info(a.path_to_normalization_info)

path_train = get_paths_to_datafiles(a.training_files)
path_test = get_paths_to_datafiles(a.test_files)

data_train = load_data(path_train)
data_train = normalize_df(data_train, norm_info)

data_test = load_data(path_test)
data_test = normalize_df(data_test, norm_info)

## SAMPLING FROM STATE TRAJECTORY
DS = DataSelector(a)
DS.load_data_into_selector(data_train)
X, Y = DS.return_dataset_for_training(shuffle=True,
                                      inputs=a.state_inputs + a.control_inputs,
                                      outputs=a.outputs,
                                      raw=True)
X = X.squeeze().astype(np.float64)
Y = Y.squeeze().astype(np.float64)

## SUBSAMPLING FOR GP
random.seed(10)
sample_indices = random.sample(range(X.shape[0]), 10)
X_sub = X[sample_indices]
Y_sub = Y[sample_indices]
data_train_subsampled = (X_sub, Y_sub)

## PLOTTING PHASE DIAGRAMS OF SUBSAMPLED DATA
# plot_samples(data_train_subsampled, show_output=False)

## DEFINING KERNELS
inputs = a.state_inputs + a.control_inputs
indices = {key: inputs.index(key) for key in inputs}
kernels = {"position": gpf.kernels.RBF(lengthscales=[1, 1, 1],
                                       active_dims=[indices["position"],
                                                    # indices["angleD"],
                                                    indices["positionD"],
                                                    indices["Q"]
                                                    ]),

           # "positionD": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1, 1],
           #                              active_dims=[indices["angle_sin"],
           #                                           indices["angle_cos"],
           #                                           indices["angleD"],
           #                                           indices["positionD"],
           #                                           indices["Q"]
           #                                           ]),

           # "angle_sin": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1, 1],
           #                              active_dims=[indices["angle_sin"],
           #                                           indices["angle_cos"],
           #                                           indices["angleD"],
           #                                           indices["positionD"],
           #                                           indices["Q"]
           #                                           ]),

           # "angle_cos": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1, 1],
           #                              active_dims=[indices["angle_sin"],
           #                                           indices["angle_cos"],
           #                                           indices["angleD"],
           #                                           indices["positionD"],
           #                                           indices["Q"]
           #                                           ]),

           # "angleD": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1, 1],
           #                           active_dims=[indices["angle_sin"],
           #                                        indices["angle_cos"],
           #                                        indices["angleD"],
           #                                        indices["positionD"],
           #                                        indices["Q"]
           #                                        ])

}

## DEFINING MULTI OUTPUT GPR MODEL
model = MultiOutGPR(a)
model.setup(data_train_subsampled, kernels)

# plot_gp_test(model, data_train)  # plot prediction with kernel priors

## MODEL OPTIMIZATION
# model.batch_optimize("Adam", iters=1000, lr=0.01, data=data_train)
model.optimize("Adam", iters=1000, lr=0.01)

## SAMPLING FROM STATE TRAJECTORY
DS = DataSelector(a)
DS.load_data_into_selector(data_test)
X, Y = DS.return_dataset_for_training(shuffle=True,
                                      inputs=a.state_inputs + a.control_inputs,
                                      outputs=a.outputs,
                                      raw=True)
X = X.squeeze().astype(np.float64)
Y = Y.squeeze().astype(np.float64)
data = (X, Y)

# errs = state_space_pred_err(model, data)
# print(errs)

## PLOTTING 1s CLOSED-LOOP PREDICTION FROM TEST RECORDING
# plot_test(model, data_test, closed_loop=True)

# save model
save_dir = a.path_to_models + "GPR_model"
print("Saving...")
save_model(model, save_dir)
print("Done!")

## TIMING PREDICTION
initialization = '''
import tensorflow as tf
import numpy as np
from SI_Toolkit.GP.Models import load_model
from SI_Toolkit.GP import args

a = args()
save_dir = a.path_to_models + "GPR_model"

# load model
print("Loading...")
m_loaded = load_model(save_dir)
print("Done!")

num_rollouts = 2000
horizon = 50

s = tf.zeros(shape=[num_rollouts, 3], dtype=tf.float64)
m_loaded.predict_f(s)
'''

code = '''\
mn, _ = m_loaded.predict_f(s)
'''

print(timeit.timeit(code, number=50, setup=initialization))

# plot_test(m_loaded, data_test, closed_loop=True)  # plot posterior predictions with loaded trained model

