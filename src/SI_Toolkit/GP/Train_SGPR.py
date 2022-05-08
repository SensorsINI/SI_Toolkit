from SI_Toolkit.GP.Models import MultiOutSGPR, run_tf_optimization, save_model, \
    load_model, plot_samples, plot_test, state_space_pred_err

import os
import timeit

import gpflow as gpf
import random
import tensorflow as tf
import numpy as np

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX
from SI_Toolkit.GP.Parameters import args
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles, load_normalization_info, \
    normalize_df, denormalize_df, normalize_numpy_array, denormalize_numpy_array
from SI_Toolkit.GP.DataSelector import DataSelector
import matplotlib.pyplot as plt

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
X = X.squeeze()
Y = Y.squeeze()

data = (X, Y)

## SUBSAMPLING FOR GP
# random.seed(10)
# sample_indices = random.sample(range(X.shape[0]), 100000)
X_samples = X  # [sample_indices]
Y_samples = Y  # [sample_indices]
data_samples = (X_samples, Y_samples)

## DEFINING KERNELS
inputs = a.state_inputs + a.control_inputs
indices = {key: inputs.index(key) for key in inputs}
kernels = {"position": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1],
                                       active_dims=[indices["position"],
                                                    indices["angleD"],
                                                    indices["positionD"],
                                                    indices["Q"]
                                                    ]),

           "positionD": gpf.kernels.Matern32(lengthscales=[1, 1, 1, 1, 1],
                                        active_dims=[indices["angle_sin"],
                                                     indices["angle_cos"],
                                                     indices["angleD"],
                                                     indices["positionD"],
                                                     indices["Q"]
                                                     ]),

           "angle_sin": gpf.kernels.Matern32(lengthscales=[1, 1, 1, 1, 1],
                                        active_dims=[indices["angle_sin"],
                                                     indices["angle_cos"],
                                                     indices["angleD"],
                                                     indices["positionD"],
                                                     indices["Q"]
                                                     ]),

           "angle_cos": gpf.kernels.Matern32(lengthscales=[1, 1, 1, 1, 1],
                                        active_dims=[indices["angle_sin"],
                                                     indices["angle_cos"],
                                                     indices["angleD"],
                                                     indices["positionD"],
                                                     indices["Q"]
                                                     ]),

           "angleD": gpf.kernels.Matern12(lengthscales=[1, 1, 1, 1, 1],
                                     active_dims=[indices["angle_sin"],
                                                  indices["angle_cos"],
                                                  indices["angleD"],
                                                  indices["positionD"],
                                                  indices["Q"]
                                                  ])

}

## DEFINING MULTI OUTPUT SGPR MODEL
sample_indices = random.sample(range(X_samples.shape[0]), 10)
data_subsampled = (data_samples[0][sample_indices], data_samples[1][sample_indices])

## PLOTTING PHASE DIAGRAMS OF SUBSAMPLED DATA
plot_samples(data_subsampled, show_output=False)

# X = np.empty(shape=[0, 6])
# Y = np.empty(shape=[0, 5])
# for df in data_train:
#     df = df[a.state_inputs + a.control_inputs].to_numpy()
#     X = np.vstack([X, df[:-1, :]])
#     Y = np.vstack([Y, df[1:, :-1]])

model = MultiOutSGPR(a)
# model.setup(data_samples, kernels, X_samples[sample_indices])
model.setup((X, Y), kernels, X_samples[sample_indices])

# plot_gp_test(model, data_train)  # plot prediction with kernel priors

X_val = np.empty(shape=[0, 6])
Y_val = np.empty(shape=[0, 5])
for df in data_test:
    df = df[a.state_inputs + a.control_inputs].to_numpy()
    X_val = np.vstack([X_val, df[:-1, :]])
    Y_val = np.vstack([Y_val, df[1:, :-1]])

## MODEL OPTIMIZATION
maxiter = 800
logf, logf_val = model.optimize("Adam", iters=maxiter, lr=0.08, val_data=(X_val, Y_val))

plt.figure(figsize=(12, 12))
for i in range(len(model.outputs)):
    plt.plot(np.arange(maxiter)[::10], logf_val[i])
plt.legend(model.outputs)
plt.xlabel("iteration")
plt.ylabel("VAL ERROR")
plt.show()

# for i in range(len(model.outputs)):
#     plt.plot(np.arange(maxiter)[::10], logf[i])
# plt.legend(model.outputs)
# plt.xlabel("iteration")
# plt.ylabel("TRAIN ERROR")
# plt.ylim(5000, 9000)
#plt.show()

## SAMPLING FROM STATE TRAJECTORY
DS = DataSelector(a)
DS.load_data_into_selector(data_test)
X, Y = DS.return_dataset_for_training(shuffle=True,
                                      inputs=a.state_inputs + a.control_inputs,
                                      outputs=a.outputs,
                                      raw=True)
X = X.squeeze()
Y = Y.squeeze()
data = (X, Y)

errs = state_space_pred_err(model, data)
print(errs)

## PLOTTING 1s CLOSED-LOOP PREDICTION FROM TEST RECORDING
plot_test(model, data_test, closed_loop=True)

# save model
save_dir = a.path_to_models + "SGPR_model"
print("Saving...")
save_model(model, save_dir)
print("Done!")

## TIMING PREDICTION WITH LOADED MODEL
initialization = '''
import tensorflow as tf
import numpy as np
from SI_Toolkit.GP.Models import load_model
from SI_Toolkit.GP.Parameters import args

a = args()
save_dir = a.path_to_models + "SGPR_model"

# load model
print("Loading...")
m_loaded = load_model(save_dir)
print("Done!")

num_rollouts = 1500
horizon = 22

s = tf.zeros(shape=[num_rollouts, 5], dtype=tf.float64)
m_loaded.predict_f(s)
'''

code = '''\
mn, _ = m_loaded.predict_f(s)
'''

print(timeit.timeit(code, number=22, setup=initialization))

# plot_test(m_loaded, data_test, closed_loop=True)  # plot posterior predictions with loaded trained model

