import numpy as np
import tensorflow as tf
from tensorflow import keras

from SI_Toolkit.Functions.TF.Dataset import Dataset

try:
    from SI_Toolkit_ASF.DataSelector import DataSelector
except:
    print('No DataSelector found.')


def loss_fn(y_batch, predictions):

    return


def fit_autoregressive(net, net_info, training_dataset, validation_dataset, test_dataset, a):

    epochs = a.num_epochs
    optimizer = keras.optimizers.Adam(a.lr)
    horizon = a.post_wash_out_len + a.wash_out_len
    first_iteration = True

    #  adapt the code to allow shuffled training dataset
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        for batch in tf.range(len(training_dataset)):   # Iterate over the batches of the dataset.

            #  fix initial guess for each experiment
            x_batch, y_batch = training_dataset[batch]
            Q_estimates = np.ones_like(y_batch[:, -1, 0])
            if first_iteration is False:
                pole_length_estimates = last_pole_length_previous_batch
            else:
                pole_length_estimates = np.ones_like(y_batch[:, -1, 1])
            net_input = np.ones((16,0,2))
            with tf.GradientTape() as tape:
                for j in range(horizon):
                    # Put previous pole_length as input
                    if first_iteration is True:
                        previous_pole_length = a.first_guess_pole_length*np.ones_like(y_batch[:, -1, 1])
                        first_iteration = False
                    else:
                        if j == 0:  #  adapt if pole_length from previous iteration belongs to same experiment
                            previous_pole_length = pole_length_estimates[:, j, 1]
                        else:
                            previous_pole_length = pole_length_estimates[:, j-1, 1]

                    net_input_col = np.copy(x_batch[:, j, :])
                    net_input_col[:, 0] = previous_pole_length
                    net_input = np.concatenate((net_input, net_input_col), axis=1)
                    if j == (horizon-1):
                        last_pole_length_previous_batch = pole_length_estimates[j]

                    # end section
                net_output = net.evaluate(net_input)
                pole_length_estimates[j], Q_estimates[j] = net_output

                predictions = np.column_stack((np.array(Q_estimates), np.array(pole_length_estimates)))
                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch, predictions)

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

    return
