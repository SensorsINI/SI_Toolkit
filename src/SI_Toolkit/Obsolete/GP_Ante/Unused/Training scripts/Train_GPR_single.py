from SI_Toolkit.GP.Models import SingleOutGPRWrapper, run_tf_optimization
from SI_Toolkit.GP.Functions.save_and_load import load_model, save_model
from SI_Toolkit.GP.Functions.plot import plot_samples, plot_test, state_space_pred_err

import os
import timeit

import gpflow as gpf
import random
import tensorflow as tf
import numpy as np
from gpflow.ci_utils import ci_niter
import matplotlib.pyplot as plt


from SI_Toolkit.GP import args
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles, load_normalization_info, \
    normalize_df, denormalize_df, normalize_numpy_array, denormalize_numpy_array
from SI_Toolkit.GP.DataSelector import DataSelector

gpf.config.set_default_float(tf.float64)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF


if __name__ == '__main__':
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
    # DS = DataSelector(a)
    # DS.load_data_into_selector(data_train)
    # X, Y = DS.return_dataset_for_training(shuffle=True,
    #                                       inputs=["position", "positionD", "Q"],
    #                                       outputs=["position"],
    #                                       raw=True)
    X = X.squeeze().astype(np.float64)
    Y = Y.squeeze().astype(np.float64)

    data = (X, Y)

    ## SUBSAMPLING FOR GP
    random.seed(10)
    sample_indices = random.sample(range(X.shape[0]), 30)
    X_samples = X[sample_indices]
    Y_samples = Y[sample_indices]
    data_samples = (X_samples, Y_samples.reshape(-1, 1))

    ## PLOTTING PHASE DIAGRAMS OF SUBSAMPLED DATA
    # plot_samples(data_samples, show_output=False)

    ## DEFINING KERNELS
    inputs = a.state_inputs + a.control_inputs
    indices = {key: inputs.index(key) for key in inputs}
    kernel = gpf.kernels.RBF(lengthscales=[1, 1, 1],
                             active_dims=[indices["position"],
                                          indices["positionD"],
                                          indices["Q"]
                                          ])

    ## DEFINING MULTI OUTPUT GPR MODEL
    model = gpf.models.GPR(
        data=data_samples,
        kernel=kernel
    )

    # plot_gp_test(model, data_train)  # plot prediction with kernel priors

    # gpf.set_trainable(model.inducing_variable, False)

    ## MODEL OPTIMIZATION
    opt = gpf.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=1000))

    model = SingleOutGPRWrapper(a, model)

    # save model
    save_dir = a.path_to_models + "GPR_model"
    print("Saving...")
    save_model(model, save_dir)
    print("Done!")

    ## TIMING PREDICTION WITH LOADED MODEL
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
mn = m_loaded.predict_f(s)
'''

print(timeit.timeit(code, number=50, setup=initialization))