import numpy as np
import tensorflow as tf
from tensorflow import keras

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


def validation_step(net, validation_dataset, a):

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


def fit_autoregressive(net, net_info, training_dataset, validation_dataset, test_dataset, a):

    epochs = a.num_epochs
    optimizer = keras.optimizers.Adam(a.lr)
    loss_fn = keras.losses.MeanSquaredError()
    loss = []
    training_loss = []
    validation_loss = []
    pole_lengths = {}
    for i in range(len(training_dataset.df_lengths)):
        pole_lengths[i] = a.first_guess_pole_length

    for epoch in range(epochs):
        # validation_loss.append(validation_step(net, validation_dataset, a))
        print("\nStart of epoch %d" % (epoch,))
        for batch in tf.range(len(training_dataset)):   # Iterate over the batches of the dataset.

            x_batch, y_batch = training_dataset[batch]
            current_batch_size = np.shape(x_batch)[0]
            training_ids = training_dataset.indexes[
                           batch*training_dataset.batch_size:batch*training_dataset.batch_size+current_batch_size]
            exp_ids = giveme_idx(training_ids, training_dataset)
            net_input = x_batch[:, :, 1:]
            temp = np.copy(x_batch[:, :, 0])
            input_pole_length = np.expand_dims(temp, axis=2)
            with tf.GradientTape() as tape:

                for i in exp_ids:
                    input_pole_length[exp_ids.index(i), :, 0] = pole_lengths[i]

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

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, net.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, net.trainable_weights))

            # Log every 200 batches.
            if batch % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (batch, float(loss_value))
                )
                print("Seen so far: %s samples" % ((batch + 1) * a.batch_size))

        # validation
        training_loss.append(np.mean(loss))
        validation_loss.append(validation_step(net, validation_dataset, a))

    return np.array(loss), validation_loss
