from SI_Toolkit.GP.Models import SVGPWrapper, run_tf_optimization, save_model, \
    load_model, plot_samples, plot_test, state_space_pred_err

import os
import timeit

import gpflow as gpf
import random
import tensorflow as tf
import numpy as np
from gpflow.ci_utils import ci_niter
import matplotlib.pyplot as plt


from SI_Toolkit.GP.Parameters import args
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles, load_normalization_info, \
    normalize_df, denormalize_df, normalize_numpy_array, denormalize_numpy_array
from SI_Toolkit_ASF.DataSelector import DataSelector
from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX

gpf.config.set_default_float(tf.float64)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF


def run_opt(model, iterations, train_iter, test_iter, lr=0.01):
    logf = []
    logf_val = []
    training_loss = model.training_loss_closure(train_iter, compile=True)
    validation_loss = model.training_loss_closure(test_iter, compile=True)

    variational_params = [(model.q_mu, model.q_sqrt)]

    optimizer = tf.optimizers.Adam(lr)
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)
        natgrad_opt.minimize(training_loss, variational_params)

    for step in range(iterations):
        optimization_step()

        if step % 100 == 0:
            print("Epoch: {}".format(step))
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            elbo_val = -validation_loss().numpy()
            print("TRAIN: {} - VAL: {}".format(elbo, elbo_val))
            logf.append(elbo)
            logf_val.append(elbo_val)

    return logf, logf_val


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
    DS = DataSelector(a)
    DS.load_data_into_selector(data_train)
    X, Y = DS.return_dataset_for_training(shuffle=True,
                                          inputs=a.state_inputs + a.control_inputs,
                                          outputs=a.outputs,
                                          raw=True)
    X = X.squeeze().astype(np.float64)
    Y = Y.squeeze().astype(np.float64)

    data = (X, Y)

    ## SUBSAMPLING FOR GP
    random.seed(10)
    sample_indices = random.sample(range(X.shape[0]), 30)
    X_samples = X[sample_indices]
    Y_samples = Y[sample_indices]
    data_samples = (X_samples, Y_samples)

    ## PLOTTING PHASE DIAGRAMS OF SUBSAMPLED DATA
    plot_samples(data_samples, show_output=False)

    ## DEFINING KERNELS
    inputs = a.state_inputs + a.control_inputs
    indices = {key: inputs.index(key) for key in inputs}
    kernels = {"position": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1],
                                           active_dims=[indices["position"],
                                                        indices["angleD"],
                                                        indices["positionD"],
                                                        indices["Q"]
                                                        ]),

               "positionD": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1, 1],
                                            active_dims=[indices["angle_sin"],
                                                         indices["angle_cos"],
                                                         indices["angleD"],
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

    kernel = gpf.kernels.SeparateIndependent([kernels[k] for k in a.outputs])
    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [gpf.inducing_variables.InducingPoints(X_samples) for i in range(len(a.outputs))]
    )
    # inducing_variable = gpf.inducing_variables.SharedIndependentInducingVariables(
    #     gpf.inducing_variables.InducingPoints(X_samples)
    # )

    ## DEFINING MULTI OUTPUT GPR MODEL
    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=gpf.likelihoods.Gaussian(),
        inducing_variable=inducing_variable,
        num_latent_gps=5,
    )

    # plot_gp_test(model, data_train)  # plot prediction with kernel priors

    # gpf.set_trainable(model.inducing_variable, False)

    ## MODEL OPTIMIZATION
    # optimize_model_with_scipy(model)

    X = np.empty(shape=[0, 6])
    Y = np.empty(shape=[0, 5])
    for df in data_train:
        df = df[a.state_inputs + a.control_inputs].to_numpy()
        X = np.vstack([X, df[:-1, :]])
        Y = np.vstack([Y, df[1:, :-1]])

    # data_train = tf.data.Dataset.from_tensor_slices((X, Y))
    # train_iter = iter(data_train.batch(32, drop_remainder=True).repeat(5000))

    # gpf.set_trainable(model.inducing_variable, False)
    gpf.set_trainable(model.q_mu, False)
    gpf.set_trainable(model.q_sqrt, False)

    X_val = np.empty(shape=[0, 6])
    Y_val = np.empty(shape=[0, 5])
    for df in data_test:
        df = df[a.state_inputs + a.control_inputs].to_numpy()
        X_val = np.vstack([X_val, df[:-1, :]])
        Y_val = np.vstack([Y_val, df[1:, :-1]])

    maxiter = ci_niter(10000)
    logf, logf_val = run_opt(model, maxiter, (X, Y), (X_val, Y_val), lr=0.01)

    model = SVGPWrapper(a, model)

    plt.plot(np.arange(maxiter)[::10], logf)
    plt.xlabel("iteration")
    plt.ylabel("ELBO")
    plt.show()

    plt.plot(np.arange(maxiter)[::10], logf_val)
    plt.xlabel("iteration")
    plt.ylabel("ELBO")
    plt.show()

    DS = DataSelector(a)
    DS.load_data_into_selector(data_test)
    X, Y = DS.return_dataset_for_training(shuffle=True,
                                          inputs=a.state_inputs + a.control_inputs,
                                          outputs=a.outputs,
                                          raw=True)
    X = X.squeeze().astype(np.float64)
    Y = Y.squeeze().astype(np.float64)
    data = (X, Y)

    state_space_pred_err(model, data, SVGP=True)
    plot_test(model, data_test, closed_loop=True)

    # save model
    save_dir = a.path_to_models + "SVGP_model"
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
save_dir = a.path_to_models + "SVGP_model"

# load model
print("Loading...")
m_loaded = load_model(save_dir)
print("Done!")

num_rollouts = 2000
horizon = 50

s = tf.zeros(shape=[num_rollouts, 5], dtype=tf.float64)
m_loaded.predict_f(s)
'''

    code = '''\
mn, _ = m_loaded.predict_f(s)
'''

print(timeit.timeit(code, number=50, setup=initialization))