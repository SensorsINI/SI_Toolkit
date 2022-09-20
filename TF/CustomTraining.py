import numpy as np
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from SI_Toolkit.TF.TrainingData import get_training_files
from SI_Toolkit.TF.Autoregressive_Model import FeedBack

"""Implementation of a custom loss function class and custom Training Loops for model Training."""


class CustomMSE(keras.losses.Loss):
    def __init__(self, wash_out_len=500, post_wash_out_len=20, discount_factor=1, name="custom_mse"):
        super().__init__(name=name)
        self.time_steps = None
        self.discount_vector = None
        self.post_wash_out_len = None
        self.wash_out_len = None
        self.discount_factor = discount_factor

        self.change_parameters(post_wash_out_len, wash_out_len)

    def call(self, y_true, y_pred):
        # losses.shape => (batch, time)
        # Discard losses for timesteps ≤ wash_out_len and timesteps > time_steps
        losses = keras.losses.MSE(y_true, y_pred)[:, self.wash_out_len: self.time_steps]
        # matrix multiplication with the discount vector -> discount losses at the end of the prediction horizon
        losses = tf.linalg.matvec(losses, self.discount_vector)
        # calculate the mean loss of the batches divided by number of steps
        loss = tf.reduce_mean(losses) / self.post_wash_out_len
        return loss

    def change_parameters(self, post_wash_out_len, wash_out_len=500):
        # initialize parameters
        self.wash_out_len = wash_out_len
        self.post_wash_out_len = post_wash_out_len
        self.time_steps = wash_out_len + post_wash_out_len
        discount_vector = np.ones(shape=post_wash_out_len)
        for i in range(post_wash_out_len - 1):
            discount_vector[i + 1] = discount_vector[i] * self.discount_factor
        self.discount_vector = tf.convert_to_tensor(discount_vector, dtype=tf.float32)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Run the forward pass of the model.
        logits = model(x, training=True)
        # Compute the loss value for this minibatch.
        loss_value = loss_fn(y, logits)
    # Calculate the gradients
    grads = tape.gradient(loss_value, model.trainable_weights)
    # Run one step of gradient descent
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # Update the loss for this step.
    epoch_loss_avg.update_state(loss_value)
    return loss_value


@tf.function
def val_step(x, y):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    epoch_val_loss_avg.update_state(loss_value)


def training_loop(train_dataset, val_dataset):
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        print("\nStart of epoch %d/%d" % (epoch + 1, epochs))
        start_time = time.time()
        # # change prediction horizon over the epochs
        # if epoch > 0 and loss_fn.post_wash_out_len < 20:
        #     loss_fn.change_parameters(10 + 2 * epoch)

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train)

            # Log every 2000 batches.
            if step % 2000 == 0:
                print("Training loss at step %d: %.3e" % (step, epoch_loss_avg.result()))

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_step(x_batch_val, y_batch_val)

        # Display the losses on epoch end.
        print("End of epoch: loss: %.3e, val_loss: %.3e" % (epoch_loss_avg.result(), epoch_val_loss_avg.result()))
        train_loss.append(epoch_loss_avg.result())
        val_loss.append(epoch_val_loss_avg.result())
        epoch_loss_avg.reset_states()
        epoch_val_loss_avg.reset_states()
        print("Time taken: %.2fs" % (time.time() - start_time))

    return train_loss, val_loss


if __name__ == '__main__':
    # Import the training data
    training_dataset, validation_dataset, test_set = get_training_files()
    # Define the model
    model = FeedBack(32, 6, 500)
    # Instantiate a loss function.
    loss_fn = CustomMSE(500, 20)
    # Prepare the loss metrics.
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_val_loss_avg = tf.keras.metrics.Mean()
    # Instantiate an optimizer to train the model.
    optimizer = Adam(learning_rate=1e-4)

    # Train the model
    epochs = 5
    training_loss, validation_loss = training_loop(training_dataset, validation_dataset)

    # Plot the training losses
    epoch_number = np.arange(1, epochs + 1, step=1)
    plt.figure(dpi=100)
    plt.rc('font', size=10)
    plt.plot(epoch_number, training_loss, label='training')
    plt.plot(epoch_number, validation_loss, label='validation')
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()

    # Save the training logs and model weights
    model_name = 'ARM-6IN-32H1-32H2-6OUT-0'
    ckpt_path = './SI_Toolkit_ApplicationSpecificFiles/Experiments/L395-790-2/Models/' + model_name
    plt.savefig(ckpt_path + '/training_curve.png')
    model.save_weights(ckpt_path + '/ckpt.ckpt')
