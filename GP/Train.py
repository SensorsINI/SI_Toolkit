import copy
from typing import Optional

import os

import pandas as pd
import gpflow as gpf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.models.training_mixins import InputData
from gpflow.types import MeanAndVariance

from SI_Toolkit.GP.Parameters import args
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles, load_normalization_info, \
    normalize_df, denormalize_numpy_array
# from SI_Toolkit_ApplicationSpecificFiles.DataSelector import DataSelector

gpf.config.set_default_float(np.float64)


class MultiGPRWrapper(tf.Module):
    """
    MIMO GPR with independent kernels (separate models) for each output.
    A native (faster) version might be implemented in future versions of GPFlow
    This is a temporary workaround.

    Note: GPFlow supports independent kernels for certain GP variations such as SVGP
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.state_inputs = args.state_inputs
        self.control_inputs = args.control_inputs
        self.outputs = args.outputs
        self.normalize = args.normalize
        if self.normalize:
            self.norm_info = load_normalization_info(args.path_to_normalization_info)
        self.models = []

        inputs = self.state_inputs + self.control_inputs
        indices = {key: inputs.index(key) for key in inputs}

        # TODO load kernels from config
        self.kernels = {# "position": gpf.kernels.SquaredExponential(lengthscales=[1, 1, 1],
                        #                                             active_dims=[indices["position"],
                        #                                                          indices["positionD"],
                        #                                                          indices["Q"]]),

                        "positionD": gpf.kernels.Linear(variance=[1, 1],
                                                        active_dims=[indices["positionD"],
                                                                     indices["Q"]
                                                                     ]),

                        # "angle_sin": gpf.kernels.SquaredExponential(lengthscales=[1, 1, 1],
                        #                                             active_dims=[indices["angle_sin"],
                        #                                                          indices["angle_cos"],
                        #                                                          indices["angleD"]
                        #                                                          ]),

                        # "angle_cos": gpf.kernels.SquaredExponential(lengthscales=[1, 1, 1, 1, 1],
                        #                                             active_dims=[indices["angle_sin"],
                        #                                                          indices["angle_cos"],
                        #                                                          indices["angleD"],
                        #                                                          indices["positionD"],
                        #                                                          indices["Q"]
                        #                                                          ]),

                        "angleD": gpf.kernels.Matern32(lengthscales=1, active_dims=[indices["angle"],
                                                                                    indices["angleD"],
                                                                                    indices["positionD"],
                                                                                    indices["Q"]]),


                        "angle": gpf.kernels.Linear(variance=1, active_dims=[indices["angle"],
                                                                             indices["angleD"]])
                        }

    def setup(self, data):
        # self.global_indices = data.columns.get_indexer(self.state_inputs)
        self.global_indices = [0, 1, 5]  # FIXME get some actual way

        X = data[self.state_inputs + self.control_inputs][:-1]
        Y = data[self.outputs][1:]

        if self.normalize:
            X = normalize_df(X, self.norm_info)
            Y = normalize_df(Y, self.norm_info)

        for output in self.outputs:
            ## model definition
            m = gpf.models.GPR(data=(X.to_numpy(dtype=np.float64),
                                     Y[output].to_numpy(dtype=np.float64).reshape(-1, 1)),
                               kernel=self.kernels[output])
            # print_summary(m)
            self.models.append(m)

    def optimize(self):
        opt = gpf.optimizers.Scipy()
        for m in self.models:
            opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float64)],
                 jit_compile=True)
    def predict_f(
        self, xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:

        means = tf.TensorArray(tf.float64, size=len(self.models))
        vars = tf.TensorArray(tf.float64, size=len(self.models))

        i = 0
        for model in self.models:
            mn, vr = model.predict_f(xnew, full_cov=full_cov, full_output_cov=full_output_cov)
            means = means.write(i, mn)
            vars = vars.write(i, vr)
            i += 1
        means = means.stack()
        vars = vars.stack()

        return means, vars


    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float64),
                                  tf.TensorSpec(shape=None, dtype=tf.int32)])
    def predict_f_samples(
        self,
        xnew: InputData,
        num_samples: Optional[int] = None,
        full_cov: bool = True,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        samples_ = []
        for model in self.models:
            sample = model.predict_f_samples(xnew, num_samples, full_cov=full_cov, full_output_cov=full_output_cov)
            samples_.append(sample)
        samples_ = tf.squeeze(tf.stack(samples_, axis=1))
        return samples_


def save_model(model, save_dir):
    m = copy.deepcopy(model)
    m.state_inputs = tf.Variable(m.state_inputs)
    m.control_inputs = tf.Variable(m.control_inputs)
    m.outputs = tf.Variable(m.outputs)
    m.normalize = tf.Variable(m.normalize)
    m.global_indices = tf.Variable(m.global_indices)
    if m.normalize:
        m.norm_header = tf.Variable(m.norm_info.columns)
        m.norm_index = tf.Variable(m.norm_info.index)
        m.norm_info = tf.Variable(m.norm_info)
    tf.saved_model.save(m, save_dir)


def load_model(save_dir):
    model = tf.saved_model.load(save_dir)
    model.state_inputs = [x.decode() for x in model.state_inputs.numpy().tolist()]
    model.control_inputs = [x.decode() for x in model.control_inputs.numpy().tolist()]
    model.outputs = [x.decode() for x in model.outputs.numpy().tolist()]
    model.global_indices = [x for x in model.global_indices.numpy().tolist()]
    if model.normalize:
        model.norm_header = [x.decode() for x in model.norm_header.numpy().tolist()]
        model.norm_index = [x.decode() for x in model.norm_index.numpy().tolist()]
        model.norm_info = pd.DataFrame(model.norm_info.numpy(), columns=model.norm_header,
                                       index=model.norm_index)
    return model


def plot_gp_test(model, data, closed_loop=False):

    t = data["time"][1:]
    X = data[model.state_inputs + model.control_inputs][:-1]
    Y = data[model.outputs][1:]

    if model.normalize:
        X = normalize_df(X, model.norm_info)

    t = t.to_numpy(dtype=np.float64).reshape(-1, 1)
    X = X.to_numpy(dtype=np.float64)
    Y = Y.to_numpy(dtype=np.float64)

    ## predict mean and variance of latent GPs at test points
    if closed_loop:
        t = t[:10]
        Y = Y[:10]
        mean = np.ndarray(shape=[10, len(model.outputs)])
        var = np.ndarray(shape=[10, len(model.outputs)])
        s = X[0, :]
        for i in range(10):
            s, v = model.predict_f(tf.convert_to_tensor(s.reshape(-1, 1).T))
            s = s.numpy().squeeze()
            # s[0] = X[i+1, 0]  # use ground truth for some state variable
            mean[i, :] = s
            var[i, :] = v.numpy().squeeze()
            s = np.concatenate((s, X[i+1, -1].reshape(1)))
    else:
        mean, var = model.predict_f(X)
        mean = mean.numpy().squeeze().T
        var = var.numpy().squeeze().T

    ## generate 10 samples from posterior
    tf.random.set_seed(1)  # for reproducibility
    samples = model.predict_f_samples(X, 10)  # shape (10, 100, 1)
    samples = samples.numpy()

    if model.normalize:
        mean = denormalize_numpy_array(mean, model.outputs, model.norm_info)
        var = denormalize_numpy_array(var, model.outputs, model.norm_info)
        samples = denormalize_numpy_array(samples, model.outputs, model.norm_info)

    for i in range(len(model.outputs)):
        plt.figure(figsize=(12, 6))

        # PLOT MEAN AND VARIANCE
        # alternative for small amounts of data
        # plt.errorbar(
        #    t_test[1:], mean[:, i],
        #    yerr=1.96 * np.sqrt(var[:, i]),
        #    fmt='co', capsize=5, zorder=1, label="mean, var"
        # )
        plt.plot(t, mean[:, i], "C0", zorder=3, label="mean")
        plt.fill_between(
            t[:, 0],
            mean[:, i] - 1.96 * np.sqrt(var[:, i]),
            mean[:, i] + 1.96 * np.sqrt(var[:, i]),
            color="C0",
            alpha=0.2,
            label="var"
        )

        # PLOT SAMPLES
        # alternative for small amounts of data
        # plt.plot(t_test[1:], samples[:, :, i].numpy().T, "r.", zorder=2, label="samples")
        # plt.plot(t_test[1:], samples[:, i, :].squeeze().T, "C0", lw=0.5, zorder=1)

        # PLOT GROUND TRUTH
        # alternative for small amounts of data
        # plt.plot(t_test[1:], Y_test[:, i], "kx", mew=2, zorder=3, label="ground truth")
        plt.plot(t, Y[:, i], "r-", zorder=2, label="ground truth")


        plt.xlabel('time [s]')
        plt.title(model.outputs[i])
        plt.legend(loc="upper right")
        plt.grid(True)
        # p.savefig(save_dir + 'feature_{}.png'.format(i))
        plt.show()

def plot_data(data, features):
    for i in range(len(data)):
        for f in features:
            plt.figure(figsize=(12, 6))

            # PLOT GROUND TRUTH
            # alternative for small amounts of data
            # plt.plot(t_test[1:], Y_test[:, i], "kx", mew=2, zorder=3, label="ground truth")
            plt.plot(data[i]["time"], data[i][f], "r-", label="ground truth")

            plt.xlabel('time [s]')
            plt.title(f)
            plt.legend(loc="upper right")
            plt.grid(True)
            # p.savefig(save_dir + 'feature_{}.png'.format(i))
            plt.show()

if __name__ == '__main__':
    a = args()

    path_train = get_paths_to_datafiles(a.training_files)
    path_test = get_paths_to_datafiles(a.test_files)

    data_train = load_data(path_train)[0]  # take middle slice of first datafile
    data_train = data_train[50:200]  # load sample points from data
    data_test = load_data(path_test)[0]

    m_multi = MultiGPRWrapper(a)

    # plot_data(data_train, m_multi.outputs)
    # plot_data(data_test, m_multi.outputs)

    m_multi.setup(data_train)
    # plot_gp_test(m_multi, data_test)  # plot prediction with kernel priors

    m_multi.optimize()

    # plt.plot(data_train["Q"], data_train["positionD"], "r-")
    # plt.show()

    # plot_gp_test(m_multi, data_test, closed_loop=True)  # plot posterior predictions with loaded trained model

    # save model
    save_dir = a.path_to_models + "GP_model"
    save_model(m_multi, save_dir)

    # load model
    m_loaded = load_model(save_dir)

    # s = tf.zeros(shape=(2000, 1, 4), dtype=tf.float64)
    # m, v = m_loaded.predict_f(s)

    plot_gp_test(m_loaded, data_test, closed_loop=True)  # plot posterior predictions with loaded trained model
