try:
    import nni
except ModuleNotFoundError:
    print('Module nni not found - only needed to run training with NNI Framework')

import matplotlib.pyplot as plt

from SI_Toolkit.Functions.General.load_parameters_for_training import args

from SI_Toolkit.load_and_normalize import load_data, normalize_df, get_paths_to_datafiles

from SI_Toolkit.Functions.General.Initialization import get_norm_info_for_net, \
    get_net, set_seed

from SI_Toolkit.Functions.TF.Dataset import Dataset

from Normalising import get_denormalization_function_tf


def testing(net, net_info, test_dfs_norm, a):

    test_dataset = Dataset(test_dfs_norm, a, shuffle=False, inputs=net_info.inputs, outputs=net_info.outputs)

    del test_dfs_norm

    loss = net.evaluate(test_dataset)
    y_pred = net.predict(test_dataset)
    y = test_dataset.outputs

    return y, y_pred, loss


def test_network():

    a = args()  # argument

    net, net_info = get_net(a, library=a.library)   # load network

    # normalize test data
    normalization_info = get_norm_info_for_net(net_info, files_for_normalization=a.training_files)
    paths_to_datafiles_test = get_paths_to_datafiles(a.test_files)
    test_dfs = load_data(paths_to_datafiles_test)
    test_dfs_norm = normalize_df(test_dfs, normalization_info)

    # test network
    y_norm, y_pred_norm, loss = testing(net, net_info, test_dfs_norm, a)

    # denormalize data
    denormalize = get_denormalization_function_tf(normalization_info, variables_names='pole_length')
    y = denormalize(y_norm)
    y_pred = denormalize(y_pred_norm)

    # plot data

    print('Test Loss: %f' % loss)

    for test_y, test_y_pred in y, y_pred:
        plt.figure()
        plt.plot(test_y, label='true_pole_length')
        plt.plot(test_y_pred, label='estimated_pole_length')
        plt.xlabel("Training Epoch")
        plt.ylabel("Length")
        plt.legend()
        plt.title(net_info.net_full_name)

test_network()






