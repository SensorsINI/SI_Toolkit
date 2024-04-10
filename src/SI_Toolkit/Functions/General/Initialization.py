import os

import numpy as np
import random as rnd

from shutil import copy as shutil_copy
import shutil
import copy

from datetime import datetime

from types import SimpleNamespace

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


def load_net_info_from_txt_file(txt_path, parent_net_name=None, convert_to_delta=False, net_info=None):
    # region Get information about the pretrained network from the associated txt file
    if net_info is None:
        net_info = SimpleNamespace()

    with open(txt_path, newline='') as f:
        lines = f.read().splitlines()

    for i in range(len(lines)):
        if lines[i] == 'CREATED:':
            created = lines[i + 1].rstrip("\n")
            date = created[:10]
            time = created[-8:]
            continue
        if lines[i] == 'LIBRARY:':
            net_info.library = lines[i + 1].rstrip("\n")
            continue
        if lines[i] == 'NET NAME:':
            net_info.net_name = lines[i + 1].rstrip("\n")
            if convert_to_delta:
                net_info.net_name = 'Delta' + net_info.net_name
            continue
        if lines[i] == 'NET FULL NAME:':
            net_info.net_full_name = lines[i + 1].rstrip("\n")
            continue
        if lines[i] == 'INPUTS:':
            net_info.inputs = lines[i + 1].rstrip("\n").split(sep=', ')
            continue
        if lines[i] == 'OUTPUTS:':
            net_info.outputs = lines[i + 1].rstrip("\n").split(sep=', ')
            continue
        if lines[i] == 'TRANSLATION INVARIANT VARIABLES:':
            net_info.translation_invariant_variables = lines[i + 1].rstrip("\n").split(sep=', ')
            if len(net_info.translation_invariant_variables) == 1 and net_info.translation_invariant_variables[0] == '':
                net_info.translation_invariant_variables = []
            continue
        if lines[i] == 'TYPE:':
            net_info.net_type = lines[i + 1].rstrip("\n").split(sep=', ')
            continue
        if lines[i] == 'NORMALIZATION:':
            path_to_normalization_info = lines[i + 1].rstrip("\n")
            if parent_net_name is not None:
                path_to_normalization_info = os.path.join(net_info.path_to_models, parent_net_name, os.path.basename(path_to_normalization_info))
            net_info.path_to_normalization_info = path_to_normalization_info
            continue
        if lines[i] == 'NORMALIZE:':
            net_info.normalize = lines[i + 1].rstrip('\n') == 'True'
        if lines[i] == 'SAMPLING INTERVAL:':
            net_info.sampling_interval = float(lines[i + 1].rstrip("\n")[:-2])
            continue
        if lines[i] == 'WASH OUT LENGTH:':
            net_info.wash_out_len = int(lines[i + 1].rstrip("\n"))
            continue
        if lines[i] == 'POST WASH OUT LENGTH:':
            net_info.post_wash_out_len = int(lines[i + 1].rstrip("\n"))
            continue
        if lines[i] == 'SHIFT LABELS:':
            net_info.shift_labels = int(lines[i + 1].rstrip("\n"))
            continue
        if lines[i] == 'CONSTRUCT NETWORK:':
            net_info.construct_network = lines[i + 1].rstrip("\n")
            continue
        if lines[i] == 'TIMESTEP MEAN [s]:':
            net_info.dt = float(lines[i + 1].rstrip("\n"))
            continue
        if lines[i] == 'TIMESTEP STD [s]:':
            net_info.dt_std = float(lines[i + 1].rstrip("\n"))
            continue

    return net_info


def load_pretrained_network(net_info, time_series_length, batch_size, stateful, remove_redundant_dimensions=False):
    # In case net_name is 'last' iterate till a valid file is found
    while True:  # Exit from while loop is done with break statement or when getting an error
        # region In case net_name is 'last' we have to first find (full) name of the last trained net
        if net_info.net_name == 'last':
            try:
                directory = net_info.path_to_models
                path_to_latest_model_directory = \
                    max([os.path.join(directory, d) for d in os.listdir(directory)], key=os.path.getctime)
                # The net full name is the same as folder name in which it is stored
                parent_net_name = os.path.basename(os.path.normpath(path_to_latest_model_directory))
            except ValueError:
                raise ValueError('No information about any pretrained network found at {}'.format(net_info.path_to_models))
        else:
            parent_net_name = net_info.net_name

        # After above if statement we have parent_net_name and can load it

        # region check for DeltaGRU, and alternatively load normal GRU printing a warning
        convert_to_delta = False
        if os.path.isdir(os.path.join(net_info.path_to_models, parent_net_name)):
            print(f'Loading a pretrained network with the full name  {parent_net_name}  from  {net_info.path_to_models}')
        else:
            if parent_net_name[:5] == 'Delta':
                if os.path.isdir(os.path.join(net_info.path_to_models, parent_net_name[5:])):
                    convert_to_delta = True
                    print('{} not found, loading {} instead'.format(parent_net_name, parent_net_name[5:]))
                    parent_net_name = parent_net_name[5:]
                else:
                    raise FileNotFoundError('Neither specified DeltaGRU nor a normal GRU (Delta){} was found'.format(parent_net_name))
            else:
                raise FileNotFoundError('{} not found'.format(parent_net_name))
        print('')

        # region Ensure that needed txt file are present in the indicated folder
        # They might be missing e.g. if a previous training session was terminated prematurely
        txt_filename = parent_net_name + '.txt'
        txt_path = os.path.join(net_info.path_to_models, parent_net_name, txt_filename)
        if not os.path.isfile(txt_path):
            txt_not_found_str = 'The corresponding .txt file is missing' \
                                '(information about inputs and outputs) at the location {}' \
                .format(txt_path)
            if net_info.net_name == 'last':
                print(txt_not_found_str)
                print('I delete the corresponding folder and try to search again')
                print('')
                os.remove(path_to_latest_model_directory)
                continue
            else:
                raise FileNotFoundError(txt_not_found_str)

        net_info = load_net_info_from_txt_file(txt_path, parent_net_name, convert_to_delta, net_info)
        print('Inputs to the loaded network: {}'.format(', '.join(map(str, net_info.inputs))))
        print('Outputs from the loaded network: {}'.format(', '.join(map(str, net_info.outputs))))
        print('')

        net_info.parent_net_name = parent_net_name

        net_info.path_to_net = os.path.join(net_info.path_to_models, parent_net_name)

        if not hasattr(net_info, 'wash_out_len'):
            print('Wash out not defined.')

        if not hasattr(net_info, 'library'):
            print()
            print('No information about library of pretrained network (TF/Pytorch) found \n'
                  'Set to default: TF \n'
                  "We suggest adding the information manually to network's txt file \n")
            net_info.library = 'TF'
            print()

        # region Load weights from checkpoint file
        if net_info.library == 'TF':
            ckpt_filenames = [parent_net_name + '.ckpt',
                              'ckpt.ckpt']  # First is old, second is new way of naming ckpt files. The old way resulted in two long paths for Windows
        else:  # Pytorch
            ckpt_filenames = [parent_net_name + '.pt',
                              'ckpt.pt']  # First is old, second is new way of naming ckpt files. The old way resulted in two long paths for Windows
        ckpt_found = False

        ckpt_path = os.path.join(net_info.path_to_models, parent_net_name, ckpt_filenames[0])
        if os.path.isfile(ckpt_path + '.index') or os.path.isfile(ckpt_path):
            ckpt_found = True
        if not ckpt_found:
            ckpt_path = os.path.join(net_info.path_to_models, parent_net_name, ckpt_filenames[1])
            if os.path.isfile(ckpt_path + '.index') or os.path.isfile(ckpt_path):
                ckpt_found = True
        if not ckpt_found:
            ckpt_not_found_str = 'The corresponding .ckpt file is missing' \
                                 '(information about weights and biases). \n' \
                                 'it was not found neither at the location {} nor at {}' \
                .format(os.path.join(net_info.path_to_models, parent_net_name, ckpt_filenames[0], ckpt_path))

            if net_info.net_name == 'last':
                print(ckpt_not_found_str)
                print('I delete the corresponding folder and try to search again')
                print('')
                shutil.rmtree(path_to_latest_model_directory)
                continue
            else:
                raise FileNotFoundError(ckpt_not_found_str)

        if not hasattr(net_info, 'construct_network'):
            print()
            print('No information about weather to construct network with modules or cells found \n'
                  'Set to default: construct with cells \n'
                  "We suggest adding the information manually to network's txt file \n")
            net_info.construct_network = 'with cells'
            print()

        break  # If we got to this point without hitting "continue" statement, network is found

    if net_info.library == 'TF':
        from SI_Toolkit.Functions.TF.Network import compose_net_from_module, compose_net_from_net_name, load_pretrained_net_weights
    else:
        from SI_Toolkit.Functions.Pytorch.Network import compose_net_from_net_name, load_pretrained_net_weights

    if net_info.net_name.split('-')[0] == 'Custom':
        net, net_info = compose_net_from_module(net_info,
                                                time_series_length=time_series_length,
                                                batch_size=batch_size, stateful=stateful)
    else:
        net, net_info = compose_net_from_net_name(net_info,
                                                  time_series_length=time_series_length,
                                                  batch_size=batch_size, stateful=stateful,
                                                  remove_redundant_dimensions=remove_redundant_dimensions,
                                                  construct_network=net_info.construct_network,
                                                  )

    # Load the pretrained weights
    load_pretrained_net_weights(net, ckpt_path)
    print('Model loaded from a checkpoint.')

    return net, net_info


def load_new_network(net_info, time_series_length, batch_size, stateful, remove_redundant_dimensions=False):
    '''Create a new network according to provided parameters'''

    print('')
    print('No pretrained network specified. I will train a network from scratch.')
    print('')

    net_info.parent_net_name = 'Network trained from scratch'
    net_info.path_to_net = None  # Folder for net not yet created

    if not hasattr(net_info, 'wash_out_len'):
        print('Wash out not defined.')

    if not hasattr(net_info, 'path_to_normalization_info'):
        net_info.path_to_normalization_info = None

    if net_info.library == 'TF':
        from SI_Toolkit.Functions.TF.Network import compose_net_from_module, compose_net_from_net_name
    else:
        from SI_Toolkit.Functions.Pytorch.Network import compose_net_from_net_name

    # Create network architecture
    if net_info.net_name.split('-')[0] == 'Custom':
        net, net_info = compose_net_from_module(net_info,
                                                time_series_length=time_series_length,
                                                batch_size=batch_size, stateful=stateful)
    else:
        net, net_info = compose_net_from_net_name(net_info,
                                                  time_series_length=time_series_length,
                                                  batch_size=batch_size, stateful=stateful)

    return net, net_info


def get_net(a,
            # If any of arguments provided it overwrite what is given in a
            time_series_length=None,
            batch_size=None,
            stateful=False,
            remove_redundant_dimensions=False,
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
    a = copy.copy(a)  # Needed, since in predictor_autoregressive_neural this function is called multiple times. Without this, it fails.
    # region If length of timeseries to be fed into net not provided get it as a sum: wash_out_len + post_wash_out_len
    if time_series_length is None:
        time_series_length = a.wash_out_len + a.post_wash_out_len
    # endregion

    last_part_of_net_name = a.net_name.split('-')[-1]
    net_name_is_a_full_name = all(c in "0123456789" for c in last_part_of_net_name)

    if net_name_is_a_full_name or a.net_name == 'last':
        net, net_info = load_pretrained_network(a, time_series_length, batch_size, stateful, remove_redundant_dimensions)
    else:
        net, net_info = load_new_network(a, time_series_length, batch_size, stateful, remove_redundant_dimensions)

    return net, net_info


def get_norm_info_for_net(net_info, files_for_normalization=None, copy_files=True):
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
            if copy_files:
                shutil_copy(net_info.path_to_normalization_info, net_info.path_to_net)
                net_info.path_to_normalization_info = os.path.join(net_info.path_to_net, os.path.basename(net_info.path_to_normalization_info))
    else:
        # In this case (retraining) we need to provide a normalization info.
        # This normalization info should in general come from the folder of retrained network,
        #  however it is also compatible with older version of the program with normalization info placed in a different folder
        if net_info.path_to_normalization_info is None:
            raise ValueError('You must provide normalization info for retraining existing network')
        if copy_files:
            try:
                shutil_copy(net_info.path_to_normalization_info, net_info.path_to_net)
            except shutil.SameFileError:
                pass
            net_info.path_to_normalization_info = os.path.join(net_info.path_to_net, os.path.basename(
                net_info.path_to_normalization_info))
        normalization_info = load_normalization_info(net_info.path_to_normalization_info)

    # region Get sampling interval from normalization info
    net_info.sampling_interval = None
    # endregion

    return normalization_info


def create_full_name(net_info, path_to_models):
    if net_info.parent_net_name == 'Network trained from scratch':
        idx_end_prefix = net_info.net_name.find('-')  # finds first occurrence

        net_full_name = net_info.net_name[:idx_end_prefix + 1] \
                        + str(len(net_info.inputs)) + 'IN-' \
                        + net_info.net_name[idx_end_prefix + 1:] \
                        + '-' + str(len(net_info.outputs)) + 'OUT'
    else:
        net_full_name = net_info.parent_net_name

    net_index = 0
    while True:
        path_to_dir = os.path.join(path_to_models, net_full_name + '-' + str(net_index))
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
    net_info.path_to_net = path_to_dir


def create_log_file(net_info, a, dfs):
    date_now = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%H:%M:%S')
    try:
        repo = Repo()
        git_revision = repo.head.object.hexsha
    except:
        git_revision = 'unknown'

    txt_path = os.path.join(a.path_to_models, net_info.net_full_name, net_info.net_full_name + '.txt')
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
    f.write('\n\nTRANSLATION INVARIANT VARIABLES:\n')
    f.write(', '.join(map(str, net_info.translation_invariant_variables)))
    f.write('\n\nTYPE:\n')
    f.write(net_info.net_type)
    f.write('\n\nNORMALIZATION:\n')
    f.write(net_info.path_to_normalization_info)
    f.write('\n\nNORMALIZE:\n')
    f.write(str(a.normalize))
    f.write('\n\nPARENT NET:\n')
    f.write(str(net_info.parent_net_name))
    f.write('\n\nWASH OUT LENGTH:\n')
    f.write(str(net_info.wash_out_len))
    f.write('\n\nPOST WASH OUT LENGTH:\n')
    f.write(str(net_info.post_wash_out_len))
    f.write('\n\nSHIFT LABELS:\n')
    f.write(str(net_info.shift_labels))
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

    # Get dt:
    if all( 'time' in dfs[i].columns for i in range(len(dfs)) ):
        all_dt = []
        for i in range(len(dfs)):
            time = dfs[i]['time'].to_numpy()
            dt = time[1:]-time[:-1]
            all_dt.append(dt)
        all_dt = np.concatenate(all_dt)
        dt_mean = np.mean(all_dt)*a.shift_labels
        dt_std = np.std(all_dt)*a.shift_labels
    else:
        dt_mean = None
        dt_std = None

    if dt_mean is None:
        f.write('\n\nTIMESTEP MEAN [datafile rows]:\n')
        f.write(str(a.shift_labels))

        f.write('\n\nTIMESTEP STD [datafile rows]:\n')
        f.write(str(0))
    else:
        f.write('\n\nTIMESTEP MEAN [s]:\n')
        f.write(str(dt_mean))

        f.write('\n\nTIMESTEP STD [s]:\n')
        f.write(str(dt_std))


    f.close()

    # Save config for Delta Network

    if hasattr(net_info, 'delta_gru_dict') and net_info.delta_gru_dict:
        import yaml
        yaml_path = os.path.join(a.path_to_models, net_info.net_full_name, 'delta_gru_hyperparameters' + '.yaml')
        file = open(yaml_path, "w")
        yaml.dump(net_info.delta_gru_dict, file)
        file.close()

