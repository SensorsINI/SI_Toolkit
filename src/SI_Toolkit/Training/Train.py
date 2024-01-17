import os.path
import time
import timeit
import shutil

try:
    import nni
except ModuleNotFoundError:
    print('Module nni not found - only needed to run training with NNI Framework')

import matplotlib.pyplot as plt

from SI_Toolkit.Functions.General.load_parameters_for_training import args

from SI_Toolkit.load_and_normalize import load_data, normalize_df, get_paths_to_datafiles

from SI_Toolkit.Functions.General.Initialization import create_full_name, create_log_file, get_norm_info_for_net, get_net, set_seed



def train_network():

    # region Import and print "command line" arguments
    # print('')
    a = args()  # 'a' like arguments
    # print(a.__dict__)
    # print('')
    # endregion

    print('Path to experiment {}'.format(a.path_to_models))

    # The following lines help to determine if the file on Google Colab was updated
    file = os.path.realpath(__file__)
    print("Training script last modified: %s" % time.ctime(os.path.getmtime(file)))

    if a.use_nni:
        nni_parameters = nni.get_next_parameter()
    else:
        nni_parameters = None

    # region Start measuring time - to evaluate performance of the training function
    start = timeit.default_timer()
    # endregion

    # region If NNI enabled load new parameters
    if nni_parameters is not None:
        a.net_name = 'GRU-' + str(nni_parameters['h1']) + 'H1-' + str(nni_parameters['h2']) + 'H2'
        a.wash_out_len = int(nni_parameters['wash_out_len'])
    # endregion

    # region Set seeds to make experiment reproducible
    set_seed(a)
    # endregion

    # region Make folder to keep trained models and their logs if not yet exist
    try:
        os.makedirs(a.path_to_models[:-1])
    except FileExistsError:
        pass
    # endregion

    net, net_info = get_net(a)

    if net_info.library == 'TF':  # If loading pretrained network this has precedence against a.library
        from SI_Toolkit.Functions.TF.Training import train_network_core
    else:
        from SI_Toolkit.Functions.Pytorch.Training import train_network_core

    # Create new full name for the pretrained net
    create_full_name(net_info, a.path_to_models)
    normalization_info = get_norm_info_for_net(net_info, files_for_normalization=a.training_files)

    # Copy training config
    src = os.path.join('SI_Toolkit_ASF', 'config_training.yml')
    dst = os.path.join(a.path_to_models, net_info.net_full_name)
    shutil.copy2(src, dst)
    if net_info.library == 'TF':
        shutil.copy('SI_Toolkit/src/SI_Toolkit/Functions/TF/Training.py', dst)
    elif net_info.library == 'Pytorch':
        shutil.copy('SI_Toolkit/src/SI_Toolkit/Functions/Pytorch/Training.py', dst)

    # region Load data and prepare datasets

    paths_to_datafiles_training = get_paths_to_datafiles(a.training_files)
    paths_to_datafiles_validation = get_paths_to_datafiles(a.validation_files)
    paths_to_datafiles_test = get_paths_to_datafiles(a.test_files)

    training_dfs = load_data(paths_to_datafiles_training)
    validation_dfs = load_data(paths_to_datafiles_validation)
    test_dfs = load_data(paths_to_datafiles_test)

    if net_info.normalize:
        training_dfs = normalize_df(training_dfs, normalization_info)
        validation_dfs = normalize_df(validation_dfs, normalization_info)
        test_dfs = normalize_df(test_dfs, normalization_info)

    create_log_file(net_info, a, training_dfs)

    # endregion

    # Run the training function
    loss, validation_loss = train_network_core(net, net_info, training_dfs, validation_dfs, test_dfs, a)

    # region Plot loss change during training
    plt.figure()
    plt.plot(loss, label='train')
    plt.plot(validation_loss, label='validation')
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.title(net_info.net_full_name)
    plt.savefig(net_info.path_to_net + 'training_curve' + '.png')
    plt.show()
    # endregion

    # region If NNI enabled send final report
    if nni_parameters is not None:
        nni.report_final_result(validation_loss[-1])
    # endregion

    # When finished the training print the final message
    print("Training Completed...                                               ")
    print(" ")

    # region Calculate and print the total time it took to train the network

    stop = timeit.default_timer()
    total_time = stop - start

    # Print the total time it took to run the function
    print('Total time of training the network: ' + str(total_time))

    # endregion