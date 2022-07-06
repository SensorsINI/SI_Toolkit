import copy
from typing import Optional

import os
import timeit
import time
import csv

import pandas as pd
import gpflow as gpf
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.models.training_mixins import InputData
#from gpflow.types import MeanAndVariance
from gpflow.utilities import print_summary
from gpflow import posteriors

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX
from SI_Toolkit.GP.Parameters import args
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles, load_normalization_info, \
    normalize_df, denormalize_df, normalize_numpy_array, denormalize_numpy_array
from SI_Toolkit.GP.DataSelector import DataSelector
from SI_Toolkit.TF.TF_Functions.Compile import Compile

gpf.config.set_default_float(tf.float64)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF
matplotlib.rcParams.update({'font.size': 24})

class MultiOutGPR(tf.Module):
    """
    MIMO GPR with independent kernels (separate models) for each output.
    A native (faster) version might be implemented in future versions of GPFlow
    This is a temporary workaround.

    Note: GPFlow supports independent kernels for certain GP variations such as SVGP
    """
    def __init__(self, args):
        super().__init__()
        self.models = []
        self.posteriors = []
        self.args = args
        self.state_inputs = args.state_inputs
        self.control_inputs = args.control_inputs
        self.outputs = args.outputs
        self.normalize = args.normalize
        if self.normalize:
            self.norm_info = load_normalization_info(args.path_to_normalization_info)
        self.global_indices = [ANGLED_IDX,
                               ANGLE_COS_IDX,
                               ANGLE_SIN_IDX,
                               POSITION_IDX,
                               POSITIOND_IDX]

    def setup(self, data, kernels):
        for i in range(len(self.outputs)):
            m = gpf.models.GPR(data=(data[0],
                                     data[1].reshape(-1, 1)),
                               kernel=kernels[self.outputs[i]]
                               )
            self.models.append(m)

    def optimize(self, optimizer, iters=1000, lr=0.01, val_data=None):
        if optimizer == "Scipy":
            opt = gpf.optimizers.Scipy()
            for m in self.models:
                # iters is used as max number of iterations for L-BFGS-B
                opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=iters))
                print_summary(m)
        else:
            if optimizer == "SGD":
                opt = tf.optimizers.SGD(lr)
            elif optimizer == "Adam":
                opt = tf.optimizers.Adam(lr)
            start = time.time()
            logs = []
            logs_val = []
            for i in range(len(self.models)):
                logf, logf_val = run_tf_optimization(self.models[i], opt, iters, val_data, i, self)
                logs.append(logf)
                logs_val.append(logf_val)
                print_summary(self.models[i])
            end = time.time()
            training_time = end-start

        self.posteriors = []
        for m in self.models:
            self.posteriors.append(m.posterior(precompute_cache=posteriors.PrecomputeCacheType.VARIABLE))  # posteriors allow for caching and faster prediction

        return logs, logs_val, training_time

    def batch_optimize(self, optimizer, data, iters=1000, lr=0.01, batch_size=16):
        X = np.empty(shape=[0, len(self.state_inputs+self.control_inputs)])
        Y = np.empty(shape=[0, len(self.outputs)])
        for df in data:
            df = df[self.state_inputs + self.control_inputs].to_numpy()
            X = np.vstack([X, df[:-1, :]])
            Y = np.vstack([Y, df[1:, :-1]])

        if optimizer == "SGD":
            opt = tf.optimizers.SGD(lr)
        elif optimizer == "Adam":
            opt = tf.optimizers.Adam(lr)
        logs = []
        for i in range(len(self.models)):
            data = tf.data.Dataset.from_tensor_slices((X, Y[:, i].reshape(-1, 1)))
            train_iter = iter(data.batch(batch_size, drop_remainder=True))
            logs.append(run_tf_optimization_batch(self.models[i], opt, iters, train_iter))
            print_summary(self.models[i])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=gpf.default_float())],
                 jit_compile=False)  # predictor runs faster on MPPI if only outer predictor function uses XLA; set to True if you use predict_f directly
    def predict_f(self, x):
        # means = tf.TensorArray(gpf.default_float(), size=len(self.models))
        # vars = tf.TensorArray(gpf.default_float(), size=len(self.models))

        # i = 0
        """
        for p in self.posteriors:
            mn, _ = p.predict_f(x)
            # mn, _ = p._conditional_with_precompute(x)
            means = means.write(i, mn)
            # vars = vars.write(i, vr)
            i += 1
        """
        mn1, var1 = self.posteriors[0].predict_f(x)
        mn2, var2 = self.posteriors[1].predict_f(x)
        mn3, var3 = self.posteriors[2].predict_f(x)
        mn4, var4 = self.posteriors[3].predict_f(x)
        mn5, var5 = self.posteriors[4].predict_f(x)

        # means = tf.concat([p.predict_f(x)[0] for p in self.posteriors], axis=1)
        means = tf.concat([mn1, mn2, mn3, mn4, mn5], axis=1)
        vars = tf.concat([var1, var2, var3, var4, var5], axis=1)
        # means = tf.squeeze(tf.transpose(means.stack(), perm=[1, 0, 2]))
        # vars = tf.squeeze(tf.transpose(vars.stack(), perm=[1, 0, 2]))

        return means, vars

"""
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=gpf.default_float())],
                 jit_compile=False)  # predictor runs faster on MPPI if only outer predictor function uses XLA; set to True if you use predict_f directly
    def predict_mean(self, x):
        # means = tf.TensorArray(gpf.default_float(), size=len(self.models))
        # vars = tf.TensorArray(gpf.default_float(), size=len(self.models))

        # i = 0
        '''
        for p in self.posteriors:
            mn, _ = p.predict_f(x)
            # mn, _ = p._conditional_with_precompute(x)
            means = means.write(i, mn)
            # vars = vars.write(i, vr)
            i += 1
        '''
        mn1 = self.posteriors[0].predict_mean(x)
        mn2 = self.posteriors[1].predict_mean(x)
        mn3 = self.posteriors[2].predict_mean(x)
        mn4 = self.posteriors[3].predict_mean(x)
        mn5 = self.posteriors[4].predict_mean(x)

        # means = tf.concat([p.predict_f(x)[0] for p in self.posteriors], axis=1)
        means = tf.concat([mn1, mn2, mn3, mn4, mn5], axis=1)
        # means = tf.squeeze(tf.transpose(means.stack(), perm=[1, 0, 2]))
        # vars = tf.squeeze(tf.transpose(vars.stack(), perm=[1, 0, 2]))

        return means
"""

class MultiOutSGPR(MultiOutGPR):
    """
    MIMO SGPR with independent kernels (separate models) for each output.
    A native (faster) version might be implemented in future versions of GPFlow
    This is a temporary workaround.

    Note: GPFlow supports independent kernels for certain GP variations such as SVGP
    """
    def __init__(self, args):
        super().__init__(args)

    def setup(self, data, kernels, inducing_variables):
        for i in range(len(self.outputs)):
            # TODO: make each model take only relevant inputs by default, instead of parsing them itself
            # might lead to speedup
            # m = gpf.models.SGPR(data=(data[0][:, kernels[self.outputs[i]].active_dims].astype(dtype=np.float64),
            #                           data[1][:, i].astype(dtype=np.float64).reshape(-1, 1)),
            #                     kernel=kernels[self.outputs[i]],
            #                     inducing_variable=inducing_variables[:, kernels[self.outputs[i]].active_dims].astype(dtype=np.float64)
            #                     )
            m = gpf.models.SGPR(data=(data[0].astype(dtype=np.float64),
                                      data[1][:, i].astype(dtype=np.float64).reshape(-1, 1)),
                                kernel=kernels[self.outputs[i]],
                                inducing_variable=inducing_variables.astype(dtype=np.float64)
                                )
            self.models.append(m)
            # gpf.set_trainable(m.inducing_variable, False)

class SVGPWrapper(MultiOutGPR):
    def __init__(self, args, model):
        super().__init__(args)
        self.model = model
        self.posterior = model.posterior()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=gpf.default_float())],
                 jit_compile=False)
    def predict_f(self, x):
        means, _ = self.model.posterior().predict_f(x)
        return means

class SingleOutGPRWrapper(MultiOutGPR):
    def __init__(self, args, model):
        super().__init__(args)
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=gpf.default_float())],
                 jit_compile=False)
    def predict_f(
        self, x,
        full_cov: bool = False,
        full_output_cov: bool = False):
        means, _ = self.model.posterior().predict_f(x, full_cov=full_cov, full_output_cov=full_output_cov)
        return means


def run_tf_optimization(model, optimizer, iterations, val_data, i, wrapper):
    logf = []
    logf_val = []
    training_loss = model.training_loss_closure(compile=True)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    # @tf.function
    def validation_loss(data, i):
        X, Y = data
        Y_pred, _ = model.predict_f(X)

        Y = normalize_numpy_array(Y[:, i], features=[wrapper.outputs[i]], normalization_info=wrapper.norm_info)
        Y_pred = normalize_numpy_array(Y_pred.numpy().T, features=[wrapper.outputs[i]], normalization_info=wrapper.norm_info)

        err = np.linalg.norm(Y_pred - Y).mean()
        return err

    elbo = -training_loss().numpy()
    elbo_val = validation_loss(val_data, i)
    print("TRAIN (ELBO): {} - VAL (MAE): {}".format(elbo, elbo_val))
    logf.append(elbo)
    logf_val.append(elbo_val)
    for step in range(0, iterations):
        optimization_step()

        if step % 100 == 0:
            print("Epoch: {}".format(step))
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            elbo_val = validation_loss(val_data, i)
            print("TRAIN (ELBO): {} - VAL (MAE): {}".format(elbo, elbo_val))
            logf.append(elbo)
            logf_val.append(elbo_val)

    return logf, logf_val


def run_tf_optimization_batch(model, optimizer, iterations, data_minibatch_it):
    X = tf.Variable(np.zeros((16, 6)), shape=(16, 6), dtype=tf.float64)
    Y = tf.Variable(np.zeros((16, 1)), shape=(16, 1), dtype=tf.float64)
    tmp_model = gpf.models.GPR((X, Y), kernel=model.kernel)

    @tf.function
    def loss(data_batch):
        tmp_model.data[0].assign(data_batch[0])
        tmp_model.data[1].assign(data_batch[1])
        return tmp_model.training_loss()

    for epoch in range(iterations):
        for batch_num, batch in enumerate(data_minibatch_it):
            with tf.GradientTape() as tape:
                l = loss(batch)
            grads = tape.gradient(l, tmp_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, tmp_model.trainable_variables))

        if epoch % 10 == 0:
            print(l)

    model.kernel = tmp_model.kernel



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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_params(model, save_dir+'info/')
    tf.saved_model.save(m, save_dir+'model')

def save_params(model, save_dir):
    param_names = list(gpf.utilities.parameter_dict(model.models[0]).keys())
    param_names = [" ".join(p.split(".")[1:]) for p in param_names]
    for i in range(len(model.models)):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir+model.outputs[i])
        out = []
        out.append(model.outputs[i])
        # out.append(", ".join(model.outputs))
        out.append("-----------------------------------")
        out.append(param_names[0])
        out.append(np.array2string(model.models[i].kernel.variance.numpy(), separator=", "))
        out.append("-----------------------------------")
        out.append(param_names[1])
        out.append(np.array2string(model.models[i].kernel.lengthscales.numpy(), separator=", "))
        out.append("-----------------------------------")
        out.append(param_names[2])
        out.append(np.array2string(model.models[i].likelihood.variance.numpy(), separator=", "))
        out.append("-----------------------------------")
        out.append(param_names[3])
        out.append(np.array2string(model.models[i].inducing_variable.Z.numpy(), separator=", "))
        out.append("-----------------------------------")

        plot_samples(model.models[i].inducing_variable.Z.numpy(), save_dir=save_dir+model.outputs[i], show=False)

        with open(save_dir+model.outputs[i]+'/params.csv', 'w') as f:
            wr = csv.writer(f, delimiter="\n")
            wr.writerow(out)


def load_model(save_dir):
    model = tf.saved_model.load(save_dir+'/model')
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


def plot_samples(data, save_dir=None, show=True):
    X = data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 12))
    plt.plot(X[:, 2], X[:, 1], "bo")
    plt.xlabel(r"sin$\theta$")
    plt.ylabel(r"cos$\theta$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    if save_dir is not None: plt.savefig(save_dir+'/angle_ss.pdf')
    if show: plt.show()

    plt.figure(figsize=(12, 12))
    angle_normed = np.arctan2(X[:, 2], X[:, 1])
    angle_normed = -1.0 + (angle_normed + np.pi) / np.pi
    plt.plot(X[:, 0], angle_normed, "bo")
    plt.xlabel(r"$\dot{\theta}$")
    plt.ylabel(r"$\theta}$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    if save_dir is not None: plt.savefig(save_dir+'/angular_ss.pdf')
    if show: plt.show()

    plt.figure(figsize=(12, 12))
    plt.plot(X[:, 4], X[:, 3], "bo")
    plt.xlabel(r"$\dot{x}$")
    plt.ylabel(r"$x$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    if save_dir is not None: plt.savefig(save_dir+'/position_ss.pdf')
    if show: plt.show()

    plt.figure(figsize=(12, 12))
    plt.plot(X[:, 5], X[:, 4], "bo")
    plt.xlabel(r"$Q$")
    plt.ylabel(r"$\dot{x}$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    if save_dir is not None: plt.savefig(save_dir+'/input_ss.pdf')
    if show: plt.show()


def plot_test(model, data, closed_loop=False):
    data = data[0]  # TODO take random file from test folder
    t = data["time"][1:]
    X = data[model.state_inputs + model.control_inputs][:-1]
    Y = data[model.outputs][1:]

    t = t.to_numpy(dtype=np.float64).reshape(-1, 1)
    X = X.to_numpy(dtype=np.float64)
    Y = Y.to_numpy(dtype=np.float64)

    ## predict mean and variance of latent GPs at test points
    if closed_loop:
        t = t[100:150]  # TODO take random bit of recording
        Y = Y[100:150]
        X = X[100:150]
        mean = np.empty(shape=[0, len(model.outputs)])
        var = np.empty(shape=[0, len(model.outputs)])

        s = X[0, :-1].reshape(-1, len(model.outputs))
        for i in range(50):
            s = np.concatenate((s, X[i, -1].reshape(1, 1)), axis=1)
            s, v = model.predict_f(tf.convert_to_tensor(s.reshape(-1, 1).T))
            s = s.numpy().reshape(-1, len(model.outputs))
            v = v.numpy().reshape(-1, len(model.outputs))
            # s[0, :] = Y[i, :]  # use ground truth for some state variables
            mean = np.vstack([mean, s])
            var = np.vstack([var, v])
    else:
        mean, var = model.predict_f(X)
        mean = mean.numpy()
        var = var.numpy()

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

        # PLOT GROUND TRUTH
        # alternative for small amounts of data
        # plt.plot(t, Y[:, i], "kx", mew=2, zorder=3, label="ground truth")
        plt.plot(t, Y[:, i], "r-", zorder=2, label="ground truth")

        plt.xlabel('time [s]')
        plt.title(model.outputs[i])
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()


def state_space_pred_err(model, data, save_dir=None):
    X, Y = data
    Y_pred, _ = model.predict_f(tf.cast(X, dtype=tf.float64))
    errs = np.linalg.norm(Y_pred.numpy() - Y, axis=1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 10))
    plt.scatter(X[:, 2], X[:, 1], s=150, c=errs)
    plt.colorbar()
    plt.xlabel(r"sin$\theta$")
    plt.ylabel(r"cos$\theta$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir + '/angle_ss_err.pdf')
        plt.savefig(save_dir + '/angle_ss_err.pdf')
    plt.show()

    plt.figure(figsize=(12, 10))
    angle_normed = np.arctan2(X[:, 2], X[:, 1])
    angle_normed = -1.0 + (angle_normed + np.pi) / np.pi
    plt.scatter(X[:, 0], angle_normed, s=150, c=errs)
    plt.colorbar()
    plt.xlabel(r"$\dot{\theta}$")
    plt.ylabel(r"$\theta}$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir + '/angular_ss_err.pdf')
        plt.savefig(save_dir + '/angular_ss_err.pdf')
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.scatter(X[:, 4], X[:, 3], s=150, c=errs)
    plt.colorbar()
    plt.xlabel(r"$\dot{x}$")
    plt.ylabel(r"$x$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir + '/position_ss_err.pdf')
        plt.savefig(save_dir + '/position_ss_err.pdf')
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.scatter(X[:, 5], Y[:, 4], s=150, c=errs)
    plt.colorbar()
    plt.xlabel(r"$Q$")
    plt.ylabel(r"$\dot{x}$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir + '/input_ss_err.pdf')
        plt.savefig(save_dir + '/input_ss_err.pdf')
    plt.show()

    return np.sum(errs)

