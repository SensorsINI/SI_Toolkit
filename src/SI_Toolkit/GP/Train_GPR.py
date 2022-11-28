from SI_Toolkit.GP.Models import MultiOutSGPR, MultiOutGPR

from SI_Toolkit.GP.Functions.save_and_load import get_normalized_data_for_training, save_model, save_training_time, save_training_script
from SI_Toolkit.GP.Functions.plot import plot_samples, plot_test, state_space_prediction_error, plot_error
from SI_Toolkit.GP.Functions.other import reformat_data_for_vaidation, preselect_data_for_GPs
from SI_Toolkit.GP.Functions.GP_ASF import get_kernels

import os

import gpflow as gpf
import random
import tensorflow as tf

import matplotlib
from SI_Toolkit.Functions.General.load_parameters_for_training import args

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

number_of_inducing_points = 10

save_dir = a.path_to_models + "SGP_{}/".format(number_of_inducing_points)
save_training_script(save_dir)

data_train, data_val, data_test = get_normalized_data_for_training(a)

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
maxiter = 400
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
