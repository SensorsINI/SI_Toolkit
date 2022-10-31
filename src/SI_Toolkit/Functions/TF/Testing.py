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
import tensorflow as tf
from SI_Toolkit.Functions.TF.CustomTraining import giveme_idx


'''
To see pole length amd Q predictions, specify the exact name of the network in config_training.yml and the experiment 
length in config_data_gen.yml
'''

config = load_config('config_data_gen.yml')
config_training = yaml.load(open(os.path.join("SI_Toolkit_ASF", "config_training.yml"), "r"), yaml.FullLoader)

dt = config['dt']['saving']

def testing(net, net_info, test_dfs_norm, a):

    a.post_wash_out_len = 1
    test_dataset = Dataset(test_dfs_norm, a, shuffle=False, inputs=net_info.inputs, outputs=net_info.outputs)

    del test_dfs_norm

    net.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(a.lr)
    )
    loss = net.evaluate(test_dataset)
    y_pred = net.predict(test_dataset)

    return y_pred, loss

def testing_autoregressive(net, net_info, test_dataset, a):

    epochs = 1
    loss_fn = keras.losses.MeanSquaredError()
    loss = []
    pole_lengths = {}
    total_predictions = 0
    for i in range(len(test_dataset.df_lengths)):
        pole_lengths[i] = a.first_guess_pole_length

    for epoch in range(epochs):
        # validation_loss.append(validation_step(net, validation_dataset, a))
        for batch in tf.range(len(test_dataset)):  # Iterate over the batches of the dataset.

            x_batch, y_batch = test_dataset[batch]
            current_batch_size = np.shape(x_batch)[0]
            training_ids = test_dataset.indexes[
                           batch * test_dataset.batch_size:batch * test_dataset.batch_size + current_batch_size]
            exp_ids = giveme_idx(training_ids, test_dataset)
            net_input = x_batch[:, :, 1:]
            temp = np.copy(x_batch[:, :, 0])
            input_pole_length = np.expand_dims(temp, axis=2)
            for i in exp_ids:
                input_pole_length[exp_ids.index(i), :, 0] = pole_lengths[i]

            net_input = np.concatenate((input_pole_length, net_input), axis=2)
            net_output = net(net_input, training=False)

            # Use the mean across washout+post_washout predictions
            real_PL = np.expand_dims(np.expand_dims(net_output[:, 0, -1], axis=1), axis=2)
            real_Q = np.expand_dims(np.expand_dims(net_output[:, 0, 0], axis=1), axis=2)
            predictions = np.concatenate((real_Q, real_PL), axis=2)
            for i in range(current_batch_size):
                pole_lengths[exp_ids[i]] = real_PL[i]

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch, net_output)

            loss.append(loss_value)
            if batch == 0:
                total_predictions = predictions
            else:
                total_predictions = np.concatenate((total_predictions, predictions), axis=0)
    print("Loss: ", np.mean(loss))

    return total_predictions, np.mean(loss)


def denormalize(array, normalization_info, type ='minmax_sym', what='pole_length'):

    new_array = []
    if type == 'minmax_sym':
        minim = normalization_info._get_value('min', what)
        maxim = normalization_info._get_value('max', what)
        for elem in array:
            new_elem = minim+(elem+1)*(maxim-minim)*0.5
            new_array.append(new_elem)

    return np.array(new_array)


def test_network():

    a = args()  # argument

    net, net_info = get_net(a, library=a.library)   # load network

    # get info
    normalization_info = get_norm_info_for_net(net_info, files_for_normalization=a.training_files)
    paths_to_datafiles_test = get_paths_to_datafiles(a.test_files)
    test_dfs = load_data(paths_to_datafiles_test)

    # normalize test data
    test_dfs_norm = normalize_df(test_dfs, normalization_info)
    test_dataset = Dataset(test_dfs_norm, a, shuffle=False, inputs=net_info.inputs, outputs=net_info.outputs)
    # end section

    # ground truth
    y = test_dataset.labels
    y = np.array(y)
    for i in range(np.shape(y)[0]):
        for j in range(len(test_dataset.outputs)):
                y[i, :, j] = denormalize(y[i, :, j], normalization_info, what=test_dataset.outputs[j])
    # end section

    # test network
    prefix = a.net_name.split('-')[0]
    if prefix == 'Autoregressive':

        y_pred_norm, loss = testing_autoregressive(net, net_info, test_dataset, a)
    else:
        y_pred_norm, loss = testing(net, net_info, test_dfs_norm, a)
        # end section

    # denormalization
    y_pred = None
    for i in range(len(test_dataset.outputs)):
        normalized_prediction = denormalize(y_pred_norm[:, 0, i], normalization_info, what=test_dataset.outputs[i])
        normalized_prediction = np.expand_dims(np.expand_dims(normalized_prediction, axis=1), axis=2)
        if i == 0:
            y_pred = normalized_prediction
        else:
            y_pred = np.concatenate((y_pred, normalized_prediction), axis=2)

    find_Q = False
    find_PL = False
    for output in test_dataset.outputs:
        if output == 'Q':
            find_Q = True
        if output == 'pole_length':
            find_PL = True

    if find_Q is True:
        Q_pred_vec = y_pred[:, 0, list(test_dataset.outputs).index('Q')]
    else:
        Q_pred_vec = None

    if find_PL is True:
        PL_pred_vec = y_pred[:, 0, list(test_dataset.outputs).index('pole_length')]
    else:
        PL_pred_vec = None
    # end section


    # plot data
    true_exp_length = test_dataset.df_lengths[0]
    if config_training['testing']['test_experiments_to_show'] == 'all':
        num_exp = [i for i in range(np.shape(y)[0])]
    else:
        num_exp = config_training['testing']['test_experiments_to_show']

    if PL_pred_vec is not None:
        PL_pred = np.zeros((int(true_exp_length), int(len(PL_pred_vec)/true_exp_length)))
        for i in range(len(PL_pred_vec)):
            PL_pred[int(i % true_exp_length), int(i // true_exp_length)] = PL_pred_vec[i]

        fig, axs = plt.subplots(len(num_exp))
        fig.suptitle(net_info.net_full_name)
        fig.supxlabel("Time [s]")
        fig.supylabel("Length [m]")
        plt.xlim()
        for i in range(len(num_exp)):
            axs[i].plot([j*dt for j in range(len(y[num_exp[i], :, -1]))], y[num_exp[i], :, -1], label='True pole length')
            axs[i].plot([j*dt for j in range(len(PL_pred[:, num_exp[i]]))], PL_pred[:, num_exp[i]], label='Estimated pole length')
            axs[i].set_ylim([0, 1])
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        plt.savefig(net_info.path_to_net + '/Pole_length_predictions' + '.png')
        plt.show()

    if Q_pred_vec is not None:
        Q_pred = np.zeros((int(true_exp_length), int(len(Q_pred_vec) / true_exp_length)))
        for i in range(len(PL_pred_vec)):
            Q_pred[int(i % true_exp_length), int(i // true_exp_length)] = Q_pred_vec[i]

        fig, axs = plt.subplots(len(num_exp))
        fig.suptitle(net_info.net_full_name)
        fig.supxlabel("Time")
        fig.supylabel("Q")
        plt.xlim()
        for i in range(len(num_exp)):
            axs[i].plot([j * dt for j in range(len(y[num_exp[i], :, 0]))], y[num_exp[i], :, 0], label='Mppi_tf Q')
            axs[i].plot([j * dt for j in range(len(Q_pred[:, num_exp[i]]))], Q_pred[:, num_exp[i]],
                        label='Estimated Q')
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        plt.savefig(net_info.path_to_net + '/Q_predictions' + '.png')
        plt.show()
    # end section