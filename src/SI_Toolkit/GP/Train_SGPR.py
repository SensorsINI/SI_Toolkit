
from SI_Toolkit.GP.Models import MultiOutSGPR
from SI_Toolkit.GP.Functions.save_and_load import get_normalized_data_for_training, save_model
from SI_Toolkit.GP.DataSelector import DataSelector
from SI_Toolkit.GP.Functions.plot import plot_samples, plot_test, state_space_prediction_error, plot_error

import os
import shutil

import gpflow as gpf
import random
import tensorflow as tf
import numpy as np

import matplotlib
from SI_Toolkit.Functions.General.load_parameters_for_training import args

gpf.config.set_default_float(tf.float64)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF
matplotlib.rcParams.update({'font.size': 24})

##  LOADING ARGUMENTS FROM CONFIG
a = args()
inputs = a.state_inputs + a.control_inputs
a.wash_out_len = 0
a.post_wash_out_len = 1
data_train, data_val, data_test = get_normalized_data_for_training(a)

## SAMPLING FROM STATE TRAJECTORY
a.num = 10
a.training = True
DS = DataSelector(a)
DS.load_data_into_selector(data_train)
X, Y = DS.return_dataset_for_training(shuffle=True,
                                      inputs=inputs,
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

model = MultiOutSGPR(a)
model.setup(data_samples, kernels, X_samples[sample_indices])

X_val = np.empty(shape=[0, 6])
Y_val = np.empty(shape=[0, 5])
for df in data_val:
    df = df[inputs].to_numpy()
    X_val = np.vstack([X_val, df[:-1, :]])
    Y_val = np.vstack([Y_val, df[1:, :-1]])

## MODEL OPTIMIZATION
maxiter = 400
logf, logf_val, train_time = model.optimize("Adam", iters=maxiter, lr=0.08, val_data=(X_val, Y_val))
with open(save_dir+'info/training_time.txt', "w") as f:
    f.write(str(train_time))

plot_error(model, maxiter, logf_val, save_dir)

## SAMPLING FROM STATE TRAJECTORY
a.num = 10
a.training = False
DS = DataSelector(a)
DS.load_data_into_selector(data_test)
X, Y = DS.return_dataset_for_training(shuffle=True,
                                      inputs=inputs,
                                      outputs=a.outputs,
                                      raw=True)
X = X.squeeze()
Y = Y.squeeze()
data = (X, Y)
test_indices = random.sample(range(X.shape[0]), 100)
data_subsampled = (data[0][test_indices], data[1][test_indices])

errs = state_space_prediction_error(model, data_subsampled, save_dir=save_dir+"info/ss_error/")

## PLOTTING 1s CLOSED-LOOP PREDICTION FROM TEST RECORDING
plot_test(model, data_test, closed_loop=True)

# save model
print("Saving...")
save_model(model, save_dir)
print("Done!")

