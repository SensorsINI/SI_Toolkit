try:
    import nni
except ModuleNotFoundError:
    print('Module nni not found - only needed to run training with NNI Framework')
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import yaml, os
from SI_Toolkit.Functions.General.load_parameters_for_training import args
from SI_Toolkit.load_and_normalize import load_data, normalize_df, get_paths_to_datafiles
from SI_Toolkit.Functions.General.Initialization import get_norm_info_for_net, get_net
from SI_Toolkit.Functions.TF.Dataset import Dataset
from SI_Toolkit.Functions.TF.Normalising import get_denormalization_function_tf
from others.globals_and_utils import load_config


'''
To see pole length prediction specify the exact name of the network in config_training.yml and the experiment 
length in config_data_gen.yml
'''

config = load_config('config_data_gen.yml')
dt = config['dt']['saving']
time_of_exp = config['length_of_experiment']
each_exp_length = time_of_exp/dt

config_training = yaml.load(open(os.path.join("SI_Toolkit_ASF", "config_training.yml"), "r"), yaml.FullLoader)


def testing(net, net_info, test_dfs_norm, a):

    test_dataset = Dataset(test_dfs_norm, a, shuffle=False, inputs=net_info.inputs, outputs=net_info.outputs)

    del test_dfs_norm

    net.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(a.lr)
    )
    loss = net.evaluate(test_dataset)
    y_pred = net.predict(test_dataset)

    return y_pred, loss


def denormalize(array, normalization_info, type ='minmax_sym', what='pole_length'):

    new_array = []
    if type == 'minmax_sym':
        minim = normalization_info._get_value('min', what)
        maxim = normalization_info._get_value('max', what)
        for elem in array:
            new_elem = minim+(elem+1)*(maxim-minim)*0.5
            new_array.append(new_elem)

    return np.array(new_array)


def test_network_pole_length():

    a = args()  # argument

    net, net_info = get_net(a, library=a.library)   # load network

    # get info
    normalization_info = get_norm_info_for_net(net_info, files_for_normalization=a.training_files)
    paths_to_datafiles_test = get_paths_to_datafiles(a.test_files)
    test_dfs = load_data(paths_to_datafiles_test)

    # ground truth
    y = np.zeros((int(each_exp_length), 0))
    for experiment in test_dfs:
        experiment_array = np.array(experiment)
        y_experiment = np.array(experiment_array[1:, -1])
        y_experiment = np.reshape(y_experiment, (len(y_experiment), 1))
        y = np.hstack((y, y_experiment))
    # end section

    # normalize test data
    test_dfs_norm = normalize_df(test_dfs, normalization_info)
    # end section

    # test network
    y_pred_norm, loss = testing(net, net_info, test_dfs_norm, a)
    # end section

    # pre-denormalization
    if np.shape(y_pred_norm)[2] is None:
        y_pred_vec_norm = np.zeros((len(y_pred_norm), 1))
        count = 0
        for i in y_pred_norm:
            y_pred_vec_norm[count] = i
            count += 1
    else:
        y_pred_vec_norm = []
        for i in y_pred_norm[:, 0]:
            y_pred_vec_norm.append(i[-1])
        y_pred_vec_norm = np.array(y_pred_vec_norm)
    # end section

    # denormalize test data
    y_pred_vec = denormalize(y_pred_vec_norm, normalization_info, what='pole_length')
    # end section

    # plot data
    y_pred = np.zeros((int(each_exp_length), int(len(y_pred_vec)/each_exp_length)))
    for i in range(len(y_pred_vec)):
        y_pred[int(i % each_exp_length), int(i // each_exp_length)] = y_pred_vec[i]

    if config_training['testing']['test_experiments_to_show'] == 'all':
        num_exp = [i for i in range(np.shape(y)[1])]
    else:
        num_exp = config_training['testing']['test_experiments_to_show']

    fig, axs = plt.subplots(len(num_exp))
    fig.suptitle(net_info.net_full_name)
    fig.supxlabel("Time")
    fig.supylabel("Length")
    plt.xlim()
    for i in range(len(num_exp)):
        axs[i].plot([j*dt for j in range(len(y[:, num_exp[i]]))], y[:, num_exp[i]], label='true pole length')
        axs[i].plot([j*dt for j in range(len(y_pred[:, num_exp[i]]))], y_pred[:, num_exp[i]], label='estimated pole length')
        axs[i].set_ylim([0, normalization_info._get_value('max', 'pole_length')+0.3])
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.savefig(net_info.path_to_net + '/Pole_length_predictions' + '.png')
    plt.show()
    # end section

def test_network_control_input():

    a = args()  # argument

    net, net_info = get_net(a, library=a.library)  # load network

    # get info
    normalization_info = get_norm_info_for_net(net_info, files_for_normalization=a.training_files)
    paths_to_datafiles_test = get_paths_to_datafiles(a.test_files)
    test_dfs = load_data(paths_to_datafiles_test)

    # ground truth
    y = np.zeros((int(each_exp_length), 0))
    for experiment in test_dfs:
        experiment_array = np.array(experiment)
        y_experiment = np.array(experiment_array[1:, -4])
        y_experiment = np.reshape(y_experiment, (len(y_experiment), 1))
        y = np.hstack((y, y_experiment))
    # end section

    # normalize test data
    test_dfs_norm = normalize_df(test_dfs, normalization_info)
    # end section

    # test network
    y_pred_norm, loss = testing(net, net_info, test_dfs_norm, a)
    # end section

    # pre-denormalization
    if np.shape(y_pred_norm)[2] is None:
        y_pred_vec_norm = np.zeros((len(y_pred_norm), 1))
        count = 0
        for i in y_pred_norm:
            y_pred_vec_norm[count] = i
            count += 1
    else:
        y_pred_vec_norm = []
        for i in y_pred_norm[:, 0]:
            y_pred_vec_norm.append(i[0])
        y_pred_vec_norm = np.array(y_pred_vec_norm)
    # end section

    # denormalize test data
    y_pred_vec = denormalize(y_pred_vec_norm, normalization_info, what='Q')
    # end section

    # plot data
    y_pred = np.zeros((int(each_exp_length), int(len(y_pred_vec) / each_exp_length)))
    for i in range(len(y_pred_vec)):
        y_pred[int(i % each_exp_length), int(i // each_exp_length)] = y_pred_vec[i]

    if config_training['testing']['test_experiments_to_show'] == 'all':
        num_exp = [i for i in range(np.shape(y)[1])]
    else:
        num_exp = config_training['testing']['test_experiments_to_show']

    fig, axs = plt.subplots(len(num_exp))
    fig.suptitle(net_info.net_full_name)
    fig.supxlabel("Time")
    fig.supylabel("Q")
    plt.xlim()
    for i in range(len(num_exp)):
        axs[i].plot([j * dt for j in range(len(y[:, num_exp[i]]))], y[:, num_exp[i]], label='mppi_tf Q')
        axs[i].plot([j * dt for j in range(len(y_pred[:, num_exp[i]]))], y_pred[:, num_exp[i]],
                    label='GRU Q')
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.savefig(net_info.path_to_net + '/Q_predictions' + '.png')
    plt.show()
    # end section






