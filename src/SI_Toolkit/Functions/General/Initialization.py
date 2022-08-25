import os

import numpy as np
import random as rnd

from shutil import copy as shutil_copy
import shutil

from datetime import datetime

try:
    # Use gitpython to get a current revision number and use it in description of experimental data
    from git import Repo
except:
    pass

from SI_Toolkit.load_and_normalize import load_normalization_info, calculate_normalization_info

# Set seeds everywhere required to make results reproducible
def set_seed(args):
    seed = args.seed
    rnd.seed(seed)
    np.random.seed(seed)
    if args.library == 'TF':
        import tensorflow as tf
        tf.random.set_seed(seed)
    else:  # Pytorch
        pass


def get_net(a,
            # If any of arguments provided it overwrite what is given in a
            time_series_length=None,
            batch_size=None,
            stateful=False,
            ):
    """
    A quite big (too big?) chunk of creating a network, its associated net_info variable
    and loading associated normalization info.
    It accepts an object a (like SimpleNamespace) which dependent on the mode of operation
        must have following attributes:
    1) Creating new network:
        a.net_name (short name), a.inputs, a.outputs, a.wash_out_len, a.path_to_normalization_info,
    2) Reload:
        a.net_name ('last' or full-name with index suffix), a.path_to_models, a.wash_out_len

    The action to take is decided based on provided net_name.
    It also deletes the folder if txt or ckpt file is missing.
    """

    # region If length of timeseries to be fed into net not provided get it as a sum: wash_out_len + post_wash_out_len
    if time_series_length is None:
        time_series_length = a.wash_out_len + a.post_wash_out_len
    # endregion

    # region Load/create rnn instance, its log and normalization

    # Check if the last part of the name is a sole number
    # If yes user provided full name and this RNN should be loaded

    last_part_of_net_name = a.net_name.split('-')[-1]
    net_name_is_a_full_name = all(c in "0123456789" for c in last_part_of_net_name)

    if net_name_is_a_full_name or a.net_name == 'last':
        load_pretrained = True
    else:
        load_pretrained = False

    # We load a pretrained network
    if load_pretrained:
        # In case net_name is 'last' iterate till a valid file is found
        while True:  # Exit from while loop is done with break statement or when getting an error
            # region In case net_name is 'last' we have to first find (full) name of the last trained net
            if a.net_name == 'last':
                try:
                    directory = a.path_to_models
                    path_to_latest_model_directory = \
                        max([os.path.join(directory, d) for d in os.listdir(directory)], key=os.path.getctime)
                    # The net full name is the same as folder name in which it is stored
                    parent_net_name = os.path.basename(os.path.normpath(path_to_latest_model_directory))
                except ValueError:
                    raise ValueError('No information about any pretrained network found at {}'.format(a.path_to_models))
            else:
                parent_net_name = a.net_name

            # After above if statement we have parent_net_name and can load it
            print('Loading a pretrained network with the full name: {}'.format(parent_net_name))
            print('')

            # endregion

            # region Ensure that needed txt file are present in the indicated folder
            # They might be missing e.g. if a previous training session was terminated prematurely
            txt_filename = parent_net_name + '.txt'
            txt_path = a.path_to_models + parent_net_name + '/' + txt_filename
            if not os.path.isfile(txt_path):
                txt_not_found_str = 'The corresponding .txt file is missing' \
                                    '(information about inputs and outputs) at the location {}' \
                    .format(txt_path)
                if a.net_name == 'last':
                    print(txt_not_found_str)
                    print('I delete the corresponding folder and try to search again')
                    print('')
                    os.remove(path_to_latest_model_directory)
                    continue
                else:
                    raise FileNotFoundError(txt_not_found_str)

            # endregion

            # region Get information about the pretrained network from the associated txt file
            with open(txt_path, newline='') as f:
                lines = f.read().splitlines()

            for i in range(len(lines)):
                if lines[i] == 'LIBRARY:':
                    library = lines[i + 1].rstrip("\n")
                    continue
                if lines[i] == 'NET NAME:':
                    net_name = lines[i + 1].rstrip("\n")
                    continue
                if lines[i] == 'NET FULL NAME:':
                    net_full_name = lines[i + 1].rstrip("\n")
                    continue
                if lines[i] == 'INPUTS:':
                    inputs = lines[i + 1].rstrip("\n").split(sep=', ')
                    continue
                if lines[i] == 'OUTPUTS:':
                    outputs = lines[i + 1].rstrip("\n").split(sep=', ')
                    continue
                if lines[i] == 'TYPE:':
                    net_type = lines[i + 1].rstrip("\n").split(sep=', ')
                    continue
                if lines[i] == 'NORMALIZATION:':
                    path_to_normalization_info_old = lines[i + 1].rstrip("\n")
                    path_to_normalization_info = os.path.join(a.path_to_models, parent_net_name,
                                                         os.path.basename(path_to_normalization_info_old))
                    continue
                if lines[i] == 'SAMPLING INTERVAL:':
                    net_sampling_interval = float(lines[i + 1].rstrip("\n")[:-2])
                    continue
                if lines[i] == 'WASH OUT LENGTH:':
                    net_wash_out_len = float(lines[i + 1].rstrip("\n"))
                    continue
                if lines[i] == 'CONSTRUCT NETWORK':
                    construct_network = lines[i + 1].rstrip("\n")
                    continue

            print('Inputs to the loaded network: {}'.format(', '.join(map(str, inputs))))
            print('Outputs from the loaded network: {}'.format(', '.join(map(str, outputs))))
            print('')

            # endregion

            if 'library' not in locals():
                print()
                print('No information about library of pretrained network (TF/Pytorch) found \n'
                      'Set to default: TF \n'
                      "We suggest adding the information manually to network's txt file \n")
                library = 'TF'
                print()

            # region Load weights from checkpoint file
            if library == 'TF':
                ckpt_filenames = [parent_net_name + '.ckpt',
                                  'ckpt.ckpt']  # First is old, second is new way of naming ckpt files. The old way resulted in two long paths for Windows
            else:  # Pytorch
                ckpt_filenames = [parent_net_name + '.pt',
                                  'ckpt.pt']  # First is old, second is new way of naming ckpt files. The old way resulted in two long paths for Windows
            ckpt_found = False

            ckpt_path = a.path_to_models + parent_net_name + '/' + ckpt_filenames[0]
            if os.path.isfile(ckpt_path + '.index') or os.path.isfile(ckpt_path):
                ckpt_found = True
            if not ckpt_found:
                ckpt_path = a.path_to_models + parent_net_name + '/' + ckpt_filenames[1]
                if os.path.isfile(ckpt_path + '.index') or os.path.isfile(ckpt_path):
                    ckpt_found = True
            if not ckpt_found:
                ckpt_not_found_str = 'The corresponding .ckpt file is missing' \
                                     '(information about weights and biases). \n' \
                                     'it was not found neither at the location {} nor at {}' \
                    .format(a.path_to_models + parent_net_name + '/' + ckpt_filenames[0], ckpt_path)

                if a.net_name == 'last':
                    print(ckpt_not_found_str)
                    print('I delete the corresponding folder and try to search again')
                    print('')
                    shutil.rmtree(path_to_latest_model_directory)
                    continue
                else:
                    raise FileNotFoundError(ckpt_not_found_str)

            if 'construct_network' not in locals():
                print()
                print('No information about weather to construct network with modules or cells found \n'
                      'Set to default: construct with cells \n'
                      "We suggest adding the information manually to network's txt file \n")
                construct_network = 'with cells'
                print()

            break  # If we got to this point without hitting "continue" statement, network is found

        # endregion

    else:

        # region Create a new network according to provided parameters

        print('')
        print('No pretrained network specified. I will train a network from scratch.')
        print('')

        net_name = a.net_name
        inputs = a.inputs
        outputs = a.outputs

        library = a.library
        construct_network = a.construct_network



        # endregion



    if library == 'TF':
        from SI_Toolkit.Functions.TF.Network import compose_net_from_net_name, load_pretrained_net_weights
    else:
        from SI_Toolkit.Functions.Pytorch.Network import compose_net_from_net_name, load_pretrained_net_weights

    # Create network architecture
    net, net_info = compose_net_from_net_name(net_name, inputs, outputs,
                                              time_series_length=time_series_length,
                                              batch_size=batch_size, stateful=stateful,
                                              construct_network=construct_network)


    # We load a pretrained network
    if load_pretrained:

        # Load the pretrained weights
        load_pretrained_net_weights(net, ckpt_path)

        # net_info.wash_out_len = a.wash_out_len

        # endregion
        print('Model loaded from a checkpoint.')

        # endregion

        # region Save the path to associated normalization file to net_info
        net_info.path_to_normalization_info = path_to_normalization_info
        # endregion

        net_info.parent_net_name = parent_net_name
        # This is the full name of pretrained net. A new full name will be given if the training is resumed
        net_info.net_full_name = net_full_name

        net_info.path_to_net = a.path_to_models + parent_net_name

        # endregion

    else:
        # region Save the path to associated normalization file to net_info
        if a.path_to_normalization_info is not None:
            net_info.path_to_normalization_info = a.path_to_normalization_info
        else:
            net_info.path_to_normalization_info = None
        # endregion

        net_info.parent_net_name = 'Network trained from scratch'
        net_info.path_to_net = None  # Folder for net not yer created


    net_info.library = library
    net_info.construct_network = construct_network

    try:
        net_info.wash_out_len = a.wash_out_len
    except AttributeError:
        print('Wash out not defined.')
    # endregion

    return net, net_info


def get_norm_info_for_net(net_info, files_for_normalization=None):
    if net_info.parent_net_name == 'Network trained from scratch':
        # In this case I can either calculate a new normalization info based on training data
        if net_info.path_to_normalization_info is None:
            if files_for_normalization is None:
                raise ValueError('You have to provide either normalization info or data files based in which it should be calculated.')
            normalization_info, net_info.path_to_normalization_info = calculate_normalization_info(files_for_normalization,
                                                                                                   plot_histograms=False,
                                                                                                   user_correction=False,
                                                                                                   path_to_norm_info=net_info.path_to_net)
        else:
            normalization_info = load_normalization_info(net_info.path_to_normalization_info)
            shutil_copy(net_info.path_to_normalization_info, net_info.path_to_net)
            net_info.path_to_normalization_info = net_info.path_to_net + os.path.basename(net_info.path_to_normalization_info)
    else:
        # In this case (retraining) we need to provide a normalization info.
        # This normalization info should in general come from the folder of retrained network,
        #  however it is also compatible with older version of the program with normalization info placed in a different folder
        if net_info.path_to_normalization_info is None:
            raise ValueError('You must provide normalization info for retraining existing network')
        shutil_copy(net_info.path_to_normalization_info, net_info.path_to_net)
        net_info.path_to_normalization_info = os.path.join(net_info.path_to_net, os.path.basename(
            net_info.path_to_normalization_info))
        normalization_info = load_normalization_info(net_info.path_to_normalization_info)

    # region Get sampling interval from normalization info
    net_info.sampling_interval = None
    # endregion

    return normalization_info


def create_full_name(net_info, path_to_models):
    idx_end_prefix = net_info.net_name.find('-')  # finds first occurrence

    net_full_name = net_info.net_name[:idx_end_prefix + 1] \
                    + str(len(net_info.inputs)) + 'IN-' \
                    + net_info.net_name[idx_end_prefix + 1:] \
                    + '-' + str(len(net_info.outputs)) + 'OUT'

    net_index = 0
    while True:
        path_to_dir = path_to_models + net_full_name + '-' + str(net_index)
        if os.path.isdir(path_to_dir):
            pass
        else:
            net_full_name += '-' + str(net_index)
            os.makedirs(path_to_dir)
            break

        net_index += 1

    print('Full name given to the currently trained network is {}.'.format(net_full_name))
    print('')
    net_info.net_full_name = net_full_name
    net_info.path_to_net = path_to_dir + '/'


def create_log_file(net_info, a):
    date_now = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%H:%M:%S')
    try:
        repo = Repo()
        git_revision = repo.head.object.hexsha
    except:
        git_revision = 'unknown'

    txt_path = a.path_to_models + net_info.net_full_name + '/' + net_info.net_full_name + '.txt'
    f = open(txt_path, 'w')
    f.write('CREATED:\n')
    f.write(date_now + ' at time ' + time_now)
    f.write('\n\nWITH GIT REVISION:\n')
    f.write(git_revision)
    f.write('\n\nLIBRARY:\n')
    f.write(net_info.library)
    f.write('\n\nNET NAME:\n')
    f.write(net_info.net_name)
    f.write('\n\nNET FULL NAME:\n')
    f.write(net_info.net_full_name)
    f.write('\n\nINPUTS:\n')
    f.write(', '.join(map(str, net_info.inputs)))
    f.write('\n\nOUTPUTS:\n')
    f.write(', '.join(map(str, net_info.outputs)))
    f.write('\n\nTYPE:\n')
    f.write(net_info.net_type)
    f.write('\n\nNORMALIZATION:\n')
    f.write(net_info.path_to_normalization_info)
    f.write('\n\nPARENT NET:\n')
    f.write(net_info.parent_net_name)
    f.write('\n\nWASH OUT LENGTH:\n')
    f.write(str(net_info.wash_out_len))
    f.write('\n\nCONSTRUCT NETWORK:\n')
    f.write(net_info.construct_network)

    f.write('\n\nTRAINING_FILES:\n')
    if type(a.training_files) is list:
        for path in a.training_files:
            f.write('     ' + path + '\n')
    else:
        f.write(a.training_files)

    f.write('\n\nVALIDATION_FILES:\n')
    if type(a.validation_files) is list:
        for path in a.validation_files:
            f.write('     ' + path + '\n')
    else:
        f.write(a.validation_files)

    f.write('\n\nTEST_FILES:\n')
    if type(a.test_files) is list:
        for path in a.test_files:
            f.write('     ' + path + '\n')
    else:
        f.write(a.test_files)

    f.close()
