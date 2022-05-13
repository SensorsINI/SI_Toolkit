import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
    CONTROL_INPUTS, CONTROL_INDICES
from SI_Toolkit.TF.TF_Functions.Normalising import normalize_tf, denormalize_tf

"""This file implements an autoregressive model with a warmup method that consists of two hidden layers of GRU. 
The prediction_autoregressive_model function is used to generate the predictions of the model for the Brunton test."""


def get_angle(net_output):
    output = net_output
    angle = tf.math.atan2(net_output[..., 2], net_output[..., 1])[:, tf.newaxis]
    output = tf.concat([angle, output], axis=-1)
    return output


def prediction_autoregressive_model(horizon, dataset):
    # prepare the inputs
    norm_info_path = "./SI_Toolkit_ApplicationSpecificFiles/Experiments/Experiment-1/" \
                     "NormalizationInfo/NI_2022-03-26_18-51-12.csv"
    input_names = ['Q', 'angleD', 'angle_cos', 'angle_sin', 'position', 'positionD']
    output_names = ['angleD', 'angle_cos', 'angle_sin', 'position', 'positionD']
    normalization_info = pd.read_csv(norm_info_path, index_col=0, comment='#')
    normalizing_inputs = tf.convert_to_tensor(normalization_info[input_names[1:]], dtype=tf.float32)
    normalizing_outputs = tf.convert_to_tensor(normalization_info[output_names], dtype=tf.float32)

    states = tf.convert_to_tensor(dataset[STATE_VARIABLES].to_numpy())
    normed_states = normalize_tf(states[:, 1:], normalizing_inputs)  # exclude the angle
    Q = tf.convert_to_tensor(dataset[CONTROL_INPUTS].to_numpy())
    inputs = tf.concat([Q, normed_states], axis=1)
    inputs = inputs[tf.newaxis, :]

    # first state
    state_0 = states[:-horizon, tf.newaxis, :]
    # predict for all starting points in the file with warmup until starting point
    outputs = []
    model = FeedBack(32)
    model.load_weights("./SI_Toolkit_ApplicationSpecificFiles/Experiments/Experiment-1/Models/feedback_model/cp.ckpt") \
        .expect_partial()

    for i in tf.range(len(inputs[0]) - horizon):
        input_sequence = inputs[:, :i + horizon, :]
        wash_out = i + 1
        normed_output = model(input_sequence, False, wash_out, horizon)
        normed_output = tf.reshape(normed_output, [-1, 5])
        output = denormalize_tf(normed_output, normalizing_outputs)
        # add the angle
        output = get_angle(output)
        outputs.append(output)
    outputs = tf.stack(outputs)
    # concatenate the first state
    outputs = tf.concat([state_0, outputs], axis=1)
    return outputs.numpy()


class FeedBack(keras.Model):
    def __init__(self, units, warmup_len=10, out_steps=20):
        super().__init__()
        self.out_steps = out_steps
        self.warmup_len = warmup_len
        self.units = units
        self.gru1_cell = layers.GRUCell(units)
        self.gru2_cell = layers.GRUCell(units)
        # Also wrap the Cells in an RNN to simplify the 'warmup' method.
        self.gru1_rnn = layers.RNN(self.gru1_cell, return_state=True, return_sequences=True)
        self.gru2_rnn = layers.RNN(self.gru2_cell, return_state=True, return_sequences=True)
        self.dense = layers.Dense(5)

    def warmup(self, inputs, warmup_len):
        # inputs.shape => (batch, time, INPUTS=6) -> only use the warmup length
        # x.shape => (batch, warmup_len, gru_units)
        # h.shape => (batch, gru_units)
        inputs = inputs[:, :warmup_len]
        x1, h1 = self.gru1_rnn(inputs)
        x2, h2 = self.gru2_rnn(x1)

        # predictions.shape => (batch, time, OUTPUTS=5)
        predictions = self.dense(x2)
        return predictions, h1, h2

    def call(self, inputs, warmup_prediction=True, warmup_len=None, out_steps=None, training=None):
        if out_steps is None:
            out_steps = self.out_steps
        if warmup_len is None:
            warmup_len = self.warmup_len
        predictions = []
        # Initialize the GRU state with the warmup method.
        warmup_predictions, h1, h2 = self.warmup(inputs, warmup_len)
        # Only keep the latest prediction
        prediction = warmup_predictions[:, -1]
        # Insert the prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, out_steps):
            # Use the last prediction as input but concatenate the control inputs.
            x = prediction
            Q = tf.reshape(inputs[:, warmup_len + n - 1, 0], shape=(-1, 1))
            x = tf.concat([Q, x], axis=1)
            # Execute one gru step.
            x1, h1 = self.gru1_cell(x, states=h1, training=training)
            x2, h2 = self.gru2_cell(h1, states=h2, training=training)
            # Convert the gru output to a prediction.
            prediction = self.dense(x2)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, OUTPUTS)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, OUTPUTS)
        predictions = tf.transpose(predictions, [1, 0, 2])
        # concatenate the warmup predictions
        if warmup_prediction:
            predictions = tf.concat([warmup_predictions, predictions], axis=1)
        return predictions
