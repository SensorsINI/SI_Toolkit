from SI_Toolkit.Testing.Parameters_for_testing import args
from SI_Toolkit.GP.Parameters import args as args_GP

import matplotlib.pyplot as plt
import yaml
from types import SimpleNamespace

from SI_Toolkit.GP.DataSelector import DataSelector

from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles, load_normalization_info, \
    normalize_df

from SI_Toolkit.Functions.General.Initialization import get_net

import os
import numpy as np
import random

def state_space_pred_err(net, data, save_dir=None):
    X, Y = data

    X_input = np.hstack([X[:, -1].reshape(-1, 1), X[:, :-1]])

    Y_pred = net(X_input[:, np.newaxis, :])
    errs = np.linalg.norm(Y_pred.numpy().squeeze() - Y, axis=1)

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

if __name__ == '__main__':
    a = args()
    a_GP = args_GP()

    net_name = "GRU-6IN-32H1-32H2-5OUT-3"

    norm_info = load_normalization_info(a_GP.path_to_normalization_info)
    config = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'),
                       Loader=yaml.FullLoader)

    PATH_TO_NN = config['testing']['PATH_TO_NN']
    save_dir = PATH_TO_NN + net_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    """
    if 'predictor_ODE_tf' in predictor_name:
        predictor = predictor_ODE_tf(horizon=1, dt=0.02)
    elif 'predictor_autoregressive_GP' in predictor_name:
        predictor = predictor_autoregressive_GP(model_name=predictor_name, horizon=1)
    else:
        predictor = predictor_autoregressive_tf(horizon=1, batch_size=1000, net_name=predictor_name)
    """

    path_test = get_paths_to_datafiles(a_GP.test_files)

    data_test = load_data(path_test)
    data_test = normalize_df(data_test, norm_info)

    a_GP.num = 10
    a_GP.training = False
    DS = DataSelector(a_GP)
    DS.load_data_into_selector(data_test)
    X, Y = DS.return_dataset_for_training(shuffle=True,
                                          inputs=a_GP.state_inputs + a_GP.control_inputs,
                                          outputs=a_GP.outputs,
                                          raw=True)
    X = X.squeeze()
    Y = Y.squeeze()
    data = (X, Y)
    test_indices = random.sample(range(X.shape[0]), 1000)
    data_subsampled = (data[0][test_indices], data[1][test_indices])

    a = SimpleNamespace()

    if '/' in net_name:
        a.path_to_models = os.path.join(*net_name.split("/")[:-1]) + '/'
        a.net_name = net_name.split("/")[-1]
    else:
        a.path_to_models = PATH_TO_NN
        a.net_name = net_name

    net, _ = \
        get_net(a, time_series_length=1,
                batch_size=1000, stateful=True, library='TF')

    state_space_pred_err(net, data_subsampled, save_dir=save_dir + '/info/ss_error/')
