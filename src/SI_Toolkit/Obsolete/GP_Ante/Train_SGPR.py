from SI_Toolkit.GP.Models import MultiOutSGPR, run_tf_optimization
from SI_Toolkit.GP.Functions.save_and_load import load_model, save_model
from SI_Toolkit.GP.Functions.plot import plot_samples, plot_test, state_space_pred_err

import os
import timeit
import shutil

import gpflow as gpf
import random
import tensorflow as tf
import numpy as np

import matplotlib
from SI_Toolkit.Functions.General.load_parameters_for_training import args
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles, load_normalization_info, \
    normalize_df, denormalize_df, normalize_numpy_array, denormalize_numpy_array
from SI_Toolkit.GP.DataSelector import DataSelector
import matplotlib.pyplot as plt

gpf.config.set_default_float(tf.float64)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF
matplotlib.rcParams.update({'font.size': 24})

##  LOADING ARGUMENTS FROM CONFIG
a = args()

## LOADING NORMALIZATION INFO AND DATA
norm_info = load_normalization_info(a.path_to_normalization_info)

path_train = get_paths_to_datafiles(a.training_files)
path_val = get_paths_to_datafiles(a.validation_files)
path_test = get_paths_to_datafiles(a.test_files)

data_train = load_data(path_train)
data_train = normalize_df(data_train, norm_info)

data_val = load_data(path_val)
data_val = normalize_df(data_val, norm_info)

data_test = load_data(path_test)
data_test = normalize_df(data_test, norm_info)

## SAMPLING FROM STATE TRAJECTORY
a.num = 10
a.training = True
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
kernels = {"position": gpf.kernels.RBF(lengthscales=[1, 1, 1],
                                       active_dims=[indices["position"],
                                                    # indices["angleD"],
                                                    indices["positionD"],
                                                    indices["Q"]
                                                    ]),

           "positionD": gpf.kernels.RBF(lengthscales=[1, 1],
                                        active_dims=[# indices["angle_sin"],
                                                     # indices["angle_cos"],
                                                     # indices["angleD"],
                                                     indices["positionD"],
                                                     indices["Q"]
                                                     ]),

           "angle_sin": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1, 1],
                                        active_dims=[indices["angle_sin"],
                                                     indices["angle_cos"],
                                                     indices["angleD"],
                                                     indices["positionD"],
                                                     indices["Q"]
                                                     ]),

           "angle_cos": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1, 1],
                                        active_dims=[indices["angle_sin"],
                                                     indices["angle_cos"],
                                                     indices["angleD"],
                                                     indices["positionD"],
                                                     indices["Q"]
                                                     ]),

           "angleD": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1, 1],
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
save_dir = a.path_to_models + "SGP_{}/".format(len(sample_indices))
if not os.path.exists(save_dir):
    os.makedirs(save_dir+'info')
shutil.copyfile("SI_Toolkit/src/SI_Toolkit/GP/Train_SGPR.py", save_dir+"info/training_file.py")
plot_samples(data_subsampled[0], save_dir=save_dir+"info/initial_ip/")
plot_samples(data[0], save_dir=save_dir+"info/training_ss/")


# X = np.empty(shape=[0, 6])
# Y = np.empty(shape=[0, 5])
# for df in data_train:
#     df = df[a.state_inputs + a.control_inputs].to_numpy()
#     X = np.vstack([X, df[:-1, :]])
#     Y = np.vstack([Y, df[1:, :-1]])

model = MultiOutSGPR(a)
model.setup(data_samples, kernels, X_samples[sample_indices])
# inducing_variables = {
#     "position": X_samples[random.sample(range(X_samples.shape[0]), 5)],
#     "positionD": X_samples[random.sample(range(X_samples.shape[0]), 5)],
#     "angle_sin": X_samples[random.sample(range(X_samples.shape[0]), 5)],
#     "angle_cos": X_samples[random.sample(range(X_samples.shape[0]), 5)],
#     "angleD": X_samples[random.sample(range(X_samples.shape[0]), 5)]
# }
# model.setup((X, Y), kernels, inducing_variables)

# plot_gp_test(model, data_train)  # plot prediction with kernel priors

X_val = np.empty(shape=[0, 6])
Y_val = np.empty(shape=[0, 5])
for df in data_val:
    df = df[a.state_inputs + a.control_inputs].to_numpy()
    X_val = np.vstack([X_val, df[:-1, :]])
    Y_val = np.vstack([Y_val, df[1:, :-1]])

## MODEL OPTIMIZATION
maxiter = 400
logf, logf_val, train_time = model.optimize("Adam", iters=maxiter, lr=0.08, val_data=(X_val, Y_val))
with open(save_dir+'info/training_time.txt', "w") as f:
    f.write(str(train_time))

plt.figure(figsize=(10, 10))
for i in range(len(model.outputs)):
    plt.plot(np.arange(maxiter+1)[::10], logf_val[i])
plt.legend(model.outputs)
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.grid()
plt.savefig(save_dir+'info/val_MAE.pdf')
plt.show()

# for i in range(len(model.outputs)):
#     plt.plot(np.arange(maxiter)[::10], logf[i])
# plt.legend(model.outputs)
# plt.xlabel("iteration")
# plt.ylabel("TRAIN ERROR")
# plt.ylim(5000, 9000)
#plt.show()

## SAMPLING FROM STATE TRAJECTORY
a.num = 10
a.training = False
DS = DataSelector(a)
DS.load_data_into_selector(data_test)
X, Y = DS.return_dataset_for_training(shuffle=True,
                                      inputs=a.state_inputs + a.control_inputs,
                                      outputs=a.outputs,
                                      raw=True)
X = X.squeeze()
Y = Y.squeeze()
data = (X, Y)
test_indices = random.sample(range(X.shape[0]), 100)
data_subsampled = (data[0][test_indices], data[1][test_indices])

errs = state_space_pred_err(model, data_subsampled, save_dir=save_dir+"info/ss_error/")
# print(errs)

## PLOTTING 1s CLOSED-LOOP PREDICTION FROM TEST RECORDING
plot_test(model, data_test, closed_loop=True)

# save model
print("Saving...")
save_model(model, save_dir)
print("Done!")

## TIMING PREDICTION WITH LOADED MODEL
initialization = '''
import tensorflow as tf
import numpy as np
from SI_Toolkit.GP.Models import load_model
from SI_Toolkit.Functions.General.load_parameters_for_training import args

a = args()
save_dir = a.path_to_models + "/SGP_{}/"

# load model
print("Loading...")
m_loaded = load_model(save_dir)
print("Done!")

num_rollouts = 2000
horizon = 35

s = tf.zeros(shape=[num_rollouts, 6], dtype=tf.float64)
m_loaded.predict_f(s)
'''.format(len(sample_indices))

code = '''\
mn = m_loaded.predict_f(s)
'''

print(timeit.timeit(code, number=35, setup=initialization))

# plot_test(model, data_val, closed_loop=True)  # plot posterior predictions with loaded trained model

