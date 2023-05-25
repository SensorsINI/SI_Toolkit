import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_samples(data, save_dir=None, show=True):
    """
    (cartpole specific)
    Plots given data in different projections of the state space:
    - cos(theta) wrt sin(theta) (illustrates different pole angles)
    - theta wrt dtheta/dt
    - x wrt dx/dt
    - dx/dt wrt Q
    """
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
    if show: plt.show(block = False)

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
    if show: plt.show(block = False)

    plt.figure(figsize=(12, 12))
    plt.plot(X[:, 4], X[:, 3], "bo")
    plt.xlabel(r"$\dot{x}$")
    plt.ylabel(r"$x$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    if save_dir is not None: plt.savefig(save_dir+'/position_ss.pdf')
    if show: plt.show(block = False)

    plt.figure(figsize=(12, 12))
    plt.plot(X[:, 5], X[:, 4], "bo")
    plt.xlabel(r"$Q$")
    plt.ylabel(r"$\dot{x}$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    if save_dir is not None: plt.savefig(save_dir+'/input_ss.pdf')
    if show: plt.show(block = False)


def plot_test(model, data, closed_loop=False):
    """
    Plots model outputs over time for given inputs
    if closed_loop is set to True, previous predictions will be used as input instead of ground truth
    """
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
        plt.show(block = False)


def state_space_prediction_error(model, data, save_dir=None):
    """
    (cartpole specific)
    Plots normed difference between prediction and ground truth in different projections of the state space:
    - cos(theta) wrt sin(theta) (illustrates different pole angles)
    - theta wrt dtheta/dt
    - x wrt dx/dt
    - dx/dt wrt Q
    Error is represented by color
    """
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
    plt.show(block = False)

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
    plt.show(block = False)

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
    plt.show(block = False)

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
    plt.show(block = False)

    return np.sum(errs)


def plot_error(model, maxiter, logf, save_dir):
    """
    Plots logged errors through training epochs
    """
    plt.figure(figsize=(10, 10))
    for i in range(len(model.outputs)):
        plt.plot(np.arange(maxiter+1)[::10], logf[i])
    plt.legend(model.outputs)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.grid()
    plt.savefig(save_dir+'info/Error.pdf')
    plt.show(block = False)
