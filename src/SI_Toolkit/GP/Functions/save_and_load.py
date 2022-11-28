import shutil
import copy
import csv
import os

import gpflow as gpf
import numpy as np
import pandas as pd

import tensorflow as tf

from SI_Toolkit.GP.Functions.plot import plot_samples
from SI_Toolkit.load_and_normalize import get_paths_to_datafiles, load_data, load_normalization_info, normalize_df


def save_model(model, save_dir, gp_type):
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
    save_params(model, save_dir+'info/', gp_type)
    tf.saved_model.save(m, save_dir+'model')


def save_params(model, save_dir, gp_type):
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
        if gp_type == 'SGPR':
            out.append(param_names[3])
            out.append(np.array2string(model.models[i].inducing_variable.Z.numpy(), separator=", "))
            out.append("-----------------------------------")

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


def get_normalized_data_for_training(args_training):

    norm_info = load_normalization_info(args_training.path_to_normalization_info)

    path_train = get_paths_to_datafiles(args_training.training_files)
    path_val = get_paths_to_datafiles(args_training.validation_files)
    path_test = get_paths_to_datafiles(args_training.test_files)

    data_train = load_data(path_train)
    data_train = normalize_df(data_train, norm_info)

    data_val = load_data(path_val)
    data_val = normalize_df(data_val, norm_info)

    data_test = load_data(path_test)
    data_test = normalize_df(data_test, norm_info)

    return data_train, data_val, data_test


def save_training_time(train_time, save_dir):
    with open(save_dir+'info/training_time.txt', "w") as f:
        f.write(str(train_time))


def save_training_script(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir+'info')
    shutil.copyfile("SI_Toolkit/src/SI_Toolkit/GP/Train_GPR.py", save_dir+"info/training_file.py")