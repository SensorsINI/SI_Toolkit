import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from SI_Toolkit.Functions.TF.Dataset import Dataset

try:
    from SI_Toolkit_ASF.DataSelector import DataSelector
except:
    print('No DataSelector found.')


def giveme_idx(training_ids, dataset):

    idx_list = dataset.df_lengths_cs
    exp_ids = []
    for j in training_ids:
        for i in idx_list:
            if j < i:
                exp_ids.append(idx_list.index(i))
                break
        if j == max(idx_list):
            exp_ids.append(len(idx_list) - 1)

    return exp_ids

def plot_data(data_to_plot, net_info, what: str):

    fig = plt.figure()
    plt.plot(data_to_plot)
    plt.ylabel(what)
    plt.xlabel('Training Batches')
    fig.savefig(net_info.path_to_net + '/' + what + '.png')
    plt.show()

def validation_step_1(net, validation_dataset, a):

    loss_fn = keras.losses.MeanSquaredError()
    loss = []
    pole_lengths = {}
    for i in range(len(validation_dataset.df_lengths)):
        pole_lengths[i] = a.first_guess_pole_length

    for batch in tf.range(len(validation_dataset)):  # Iterate over the batches of the dataset.

        x_batch, y_batch = validation_dataset[batch]
        current_batch_size = np.shape(x_batch)[0]
        training_ids = validation_dataset.indexes[
                       batch * validation_dataset.batch_size:batch * validation_dataset.batch_size + current_batch_size]
        exp_ids = giveme_idx(training_ids, validation_dataset)
        net_input = x_batch[:, :, 1:]
        input_pole_length = np.expand_dims(x_batch[:, :, 0], axis=2)

        for i in exp_ids:
            input_pole_length[exp_ids.index(i), :, 0] = pole_lengths[i]

        net_input = np.concatenate((input_pole_length, net_input), axis=2)
        net_output = net(net_input, training=False)
        predictions = np.array(net_output[:, :, 1])

        # Use the mean across washout+post_washout predictions
        means = np.mean(predictions, axis=1)
        for i in range(current_batch_size):
            pole_lengths[exp_ids[i]] = means[i]

        # Compute the loss value for this minibatch.
        loss_value = loss_fn(y_batch, net_output)
        loss.append(loss_value)

    return np.mean(loss)

'''

1 - Autoregressive Training with dictionary
2 - Autoregressive Training without dictionary, iterating along washout length
3 - Autoregressive Training with dictionary and iterating along washout length

'''

# WITH DICTIONARY
def fit_autoregressive_1(net, net_info, training_dataset, a, validation_dataset=None):

    epochs = a.num_epochs
    optimizer = keras.optimizers.Adam(a.lr)
    loss_fn = keras.losses.MeanSquaredError()
    loss = []
    training_loss = []
    validation_loss = []
    epoch_loss = []
    weight_loss = weights = prec_weights = []
    grads_loss = total_grads = prec_grads = []

    pole_lengths = {}
    for i in range(len(training_dataset.df_lengths)):
        pole_lengths[i] = np.random.uniform(-1, 1)

    for epoch in range(epochs):
        # validation_loss.append(validation_step(net, validation_dataset, a))
        print("\nStart of epoch %d" % (epoch,))
        for batch in tf.range(len(training_dataset)):   # Iterate over the batches of the dataset.

            batch_loss = []
            x_batch, y_batch = training_dataset[batch]
            current_batch_size = np.shape(x_batch)[0]
            training_ids = training_dataset.indexes[
                           batch*training_dataset.batch_size:batch*training_dataset.batch_size+current_batch_size]
            exp_ids = giveme_idx(training_ids, training_dataset)
            net_input = x_batch[:, :, 1:]
            temp = np.copy(x_batch[:, :, 0])
            input_pole_length = np.expand_dims(np.ones_like(temp), axis=2)
            with tf.GradientTape() as tape:

                for i in range(current_batch_size):
                    for j in range(net_info.total_washout):
                        input_pole_length[i, j, 0] = pole_lengths[exp_ids[i]]

                net_input = np.concatenate((input_pole_length, net_input), axis=2)
                net_output = net(net_input, training=True)
                PL_predictions = np.array(net_output[:, :, 1])

                # Use the mean across washout+post_washout predictions
                means = np.mean(PL_predictions, axis=1)
                for i in range(current_batch_size):
                    pole_lengths[exp_ids[i]] = means[i]

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch, net_output)
                loss.append(loss_value)

            if a.plot_while_training is True:

                # WEIGHTS
                first = True
                for layer_weight in net.trainable_weights:
                    if first is True:
                        weights = layer_weight.numpy().flatten()
                        first = False
                    else:
                        weights = np.concatenate((weights, layer_weight.numpy().flatten()))
                if batch == 0:
                    prec_weights = weights
                else:
                    weight_loss.append(loss_fn(prec_weights, weights))
                    prec_weights = weights

                # first_time = True
                # last_pole_lengths = pole_lengths.values()
                # last_pole_lengths = np.expand_dims(np.array(last_pole_lengths), axis=1)
                # if first_time is True:
                #     data_to_plot = last_pole_lengths
                #     first_time = False
                # else:
                #     data_to_plot = np.concatenate((data_to_plot, last_pole_lengths), axis=1)


            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, net.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, net.trainable_weights))

            if a.plot_while_training is True:
                # GRADS
                first = True
                for layer_grads in grads:
                    if first is True:
                        total_grads = layer_grads.numpy().flatten()
                        first = False
                    else:
                        total_grads = np.concatenate((total_grads, layer_grads.numpy().flatten()))
                if batch == 0:
                    prec_grads = total_grads
                else:
                    grads_loss.append(loss_fn(prec_grads, total_grads))
                    prec_grads = total_grads

            if batch % 10 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (batch, float(loss_value))
                )
                print("Seen so far: %s samples" % ((batch + 1) * a.batch_size))

            batch_loss = np.mean(batch_loss)
            epoch_loss.append(batch_loss)

        training_loss.append(np.mean(epoch_loss))
        # validation_loss.append(validation_step_1(net, validation_dataset, a))

    if a.plot_while_training is True:
        #plot_data(data_to_plot)
        plot_data(weight_loss, net_info, what='Weight_change')
        plot_data(grads_loss, net_info, what='Grads_change')

    return np.array(loss), validation_loss

# WITHOUT DICTIONARY AND LOOPING ALONG WASHOUT
def fit_autoregressive_2(net, net_info, training_dataset, a, validation_dataset=None):

    epochs = a.num_epochs
    optimizer = keras.optimizers.Adam(a.lr)
    loss_fn = keras.losses.MeanSquaredError()
    weight_loss = weights = prec_weights = []
    grads_loss = total_grads = prec_grads = []
    training_loss = []
    validation_loss = []
    epoch_loss = []
    previous_pole_length = []

    for epoch in range(epochs):
        # validation_loss.append(validation_step(net, validation_dataset, a))
        print("\nStart of epoch %d" % (epoch,))
        for batch in tf.range(len(training_dataset)):  # Iterate over the batches of the dataset.

            batch_loss = []
            # Take new batch
            x_batch, y_batch = training_dataset[batch]
            washout = np.shape(x_batch)[1]
            current_batch_size = np.shape(x_batch)[0]

            # pick states and reset previous pole_length
            cartpole_states = x_batch[:, :, 1:]
            if batch == 0:
                previous_pole_length = np.expand_dims(
                    tf.random.uniform(shape=(current_batch_size, 1), minval=-1, maxval=1).numpy(), axis=2)
            else:
                previous_pole_length = previous_pole_length[:current_batch_size, :, :]

            with tf.GradientTape() as tape:

                for time_step in range(washout):
                    sample_x = np.expand_dims(cartpole_states[:, time_step, :], axis=1)
                    sample_y = np.expand_dims(y_batch[:, time_step, :], axis=1)

                    net_input = np.concatenate((previous_pole_length, sample_x), axis=2)
                    net_output = net(net_input, training=True)

                    previous_pole_length = np.expand_dims(net_output[:, :, 0], axis=2)

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(sample_y, net_output)
                    batch_loss.append(loss_value)

            if a.plot_while_training is True:
                first = True
                for layer_weight in net.trainable_weights:
                    if first is True:
                        weights = layer_weight.numpy().flatten()
                        first = False
                    else:
                        weights = np.concatenate((weights, layer_weight.numpy().flatten()))
                if batch == 0:
                    prec_weights = weights
                else:
                    weight_loss.append(loss_fn(prec_weights, weights))
                    prec_weights = weights

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, net.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, net.trainable_weights))

            if a.plot_while_training is True:
                # GRADS
                first = True
                for layer_grads in grads:
                    if first is True:
                        total_grads = layer_grads.numpy().flatten()
                        first = False
                    else:
                        total_grads = np.concatenate((total_grads, layer_grads.numpy().flatten()))
                if batch == 0:
                    prec_grads = total_grads
                else:
                    grads_loss.append(loss_fn(prec_grads, total_grads))
                    prec_grads = total_grads

            if batch % 10 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (batch, float(loss_value))
                )
                print("Seen so far: %s samples" % ((batch + 1) * a.batch_size))

            batch_loss = np.mean(batch_loss)
            epoch_loss.append(batch_loss)

        training_loss.append(np.mean(epoch_loss))

    validation_loss = 0

    if a.plot_while_training is True:
        plot_data(weight_loss, net_info, what='Weights_change')
        plot_data(grads_loss, net_info, what='Grads_change')


    return np.array(training_loss), validation_loss

# WITH DICTIONARY AND LOOPING ALONG WASHOUT
def fit_autoregressive_3(net, net_info, training_dataset, a, validation_dataset=None):

    epochs = a.num_epochs
    optimizer = keras.optimizers.Adam(a.lr)
    loss_fn = keras.losses.MeanSquaredError()
    weight_loss = weights = prec_weights = []
    grads_loss = total_grads = prec_grads = []
    training_loss = []
    validation_loss = []
    epoch_loss = []

    pole_lengths = {}
    for i in range(len(training_dataset.df_lengths)):
        pole_lengths[i] = np.random.uniform(-1, 1)

    for epoch in range(epochs):
        # validation_loss.append(validation_step(net, validation_dataset, a))
        print("\nStart of epoch %d" % (epoch,))
        for batch in tf.range(len(training_dataset)):  # Iterate over the batches of the dataset.

            batch_loss = []
            # Take new batch
            x_batch, y_batch = training_dataset[batch]
            current_batch_size = np.shape(x_batch)[0]

            # pick states and reset previous pole_length
            cartpole_states = x_batch[:, :, 1:]

            # pick indexes
            training_ids = training_dataset.indexes[
                           batch * training_dataset.batch_size:batch * training_dataset.batch_size + current_batch_size]
            exp_ids = giveme_idx(training_ids, training_dataset)

            previous_pole_length = np.expand_dims(np.expand_dims(np.ones_like(x_batch[:, 0, 0]), axis=1), axis=2)

            with tf.GradientTape() as tape:

                for time_step in range(net_info.total_washout):

                    for i in range(current_batch_size):
                        previous_pole_length[i, 0, 0] = pole_lengths[exp_ids[i]]

                    sample_x = np.expand_dims(cartpole_states[:, time_step, :], axis=1)
                    sample_y = np.expand_dims(y_batch[:, time_step, :], axis=1)

                    net_input = np.concatenate((previous_pole_length, sample_x), axis=2)
                    net_output = net(net_input, training=True)

                    for i in range(current_batch_size):
                        pole_lengths[exp_ids[i]] = net_output[i, 0, 0]

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(sample_y, net_output)
                    batch_loss.append(loss_value)

            if a.plot_while_training is True:
                first = True
                for layer_weight in net.trainable_weights:
                    if first is True:
                        weights = layer_weight.numpy().flatten()
                        first = False
                    else:
                        weights = np.concatenate((weights, layer_weight.numpy().flatten()))
                if batch == 0:
                    prec_weights = weights
                else:
                    weight_loss.append(loss_fn(prec_weights, weights))
                    prec_weights = weights

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, net.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, net.trainable_weights))

            if a.plot_while_training is True:
                # GRADS
                first = True
                for layer_grads in grads:
                    if first is True:
                        total_grads = layer_grads.numpy().flatten()
                        first = False
                    else:
                        total_grads = np.concatenate((total_grads, layer_grads.numpy().flatten()))
                if batch == 0:
                    prec_grads = total_grads
                else:
                    grads_loss.append(loss_fn(prec_grads, total_grads))
                    prec_grads = total_grads

            if batch % 10 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (batch, float(loss_value))
                )
                print("Seen so far: %s samples" % ((batch + 1) * a.batch_size))

            batch_loss = np.mean(batch_loss)
            epoch_loss.append(batch_loss)

        training_loss.append(np.mean(epoch_loss))

    validation_loss = 0

    if a.plot_while_training is True:
        plot_data(weight_loss, net_info, what='Weights_change')
        plot_data(grads_loss, net_info, what='Grads_change')


    return np.array(training_loss), validation_loss