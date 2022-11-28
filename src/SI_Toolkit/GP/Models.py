import os
import time

import gpflow as gpf
import numpy as np
import matplotlib
import tensorflow as tf

from gpflow.utilities import print_summary
from gpflow import posteriors

from CartPole.state_utilities import ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX
from SI_Toolkit.load_and_normalize import load_normalization_info, normalize_numpy_array

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
            m = gpf.models.GPR(data=(data[0].astype(dtype=np.float64),
                                      data[1][:, i].astype(dtype=np.float64).reshape(-1, 1)),
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


    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=gpf.default_float())],
                 jit_compile=False)  # predictor runs faster on MPPI if only outer predictor function uses XLA; set to True if you use predict_f directly
    def predict_f(self, x):
        means = tf.TensorArray(gpf.default_float(), size=len(self.models))
        vars = tf.TensorArray(gpf.default_float(), size=len(self.models))

        i = 0

        for p in self.posteriors:
            mn, vr = p.predict_f(x)
            # mn, _ = p._conditional_with_precompute(x)
            means = means.write(i, mn)
            vars = vars.write(i, vr)
            i += 1

        means = tf.squeeze(tf.transpose(means.stack(), perm=[1, 0, 2]))
        vars = tf.squeeze(tf.transpose(vars.stack(), perm=[1, 0, 2]))

        return means, vars


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


