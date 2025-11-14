import tensorflow.keras as keras
import tensorflow as tf

# region Define loss and optimizer for training
import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


@tf.keras.utils.register_keras_serializable()
class LossMeanResidual(tf.keras.losses.Loss):
    def __init__(self, loss_mode, wash_out_len, post_wash_out_len, discount_factor=0.9, **kwargs):
        super().__init__(**kwargs)

        self.loss_mode = loss_mode
        self.wash_out_len = wash_out_len
        self.post_wash_out_len = post_wash_out_len
        self.discount_factor = discount_factor

        # Calculate discount vector
        discount_vector = np.ones(shape=post_wash_out_len)
        for i in range(post_wash_out_len - 1):
            discount_vector[i + 1] = discount_vector[i] * discount_factor
        self.discount_vector = tf.convert_to_tensor(discount_vector, dtype=tf.float32)

    def call(self, y_true, y_predicted):

        if 'clipped' in self.loss_mode:
            # Not for model learning!
            condition = tf.abs(y_true) >= 1.0
            y_predicted = tf.where(condition, tf.clip_by_value(y_predicted, -1.0, 1.0), y_predicted)

        # Handle shape mismatch: if y_predicted has one more timestep than y_true, slice it
        pred_time_steps = tf.shape(y_predicted)[1]
        true_time_steps = tf.shape(y_true)[1]
        
        # Use tf.slice to ensure proper gradient computation
        _y_predicted = tf.cond(
            tf.equal(pred_time_steps, true_time_steps + 1),
            lambda: tf.slice(y_predicted, [0, 1, 0], [-1, -1, -1]),  # Slice from index 1 to end
            lambda: y_predicted
        )

        if 'squared' in self.loss_mode:
            losses = keras.losses.MSE(y_true, _y_predicted)  # MSE returns per-sample loss values
        elif 'absolute' in self.loss_mode:
            losses = tf.reduce_mean(tf.abs(y_true - _y_predicted), axis=-1)  # Reduce to match MSE shape
        else:
            raise ValueError(f"Unknown loss mode: {self.loss_mode}")

        # losses has shape [batch_size, time steps] -> this is the loss for every time step
        losses = losses[:, self.wash_out_len:] # This discards losses for timesteps ≤ wash_out_len

        # Get discounted some of losses for a time series
        # Axis (2,1) results in the natural operation of losses * discount_vector
        # loss = keras.layers.Dot(axes=(1, 0))([losses, discount_vector])
        loss = tf.linalg.matvec(losses, self.discount_vector)/self.post_wash_out_len

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "wash_out_len": self.wash_out_len,
            "post_wash_out_len": self.post_wash_out_len,
            "discount_factor": self.discount_factor
        })
        return config


@tf.keras.utils.register_keras_serializable()
class LossMSRSequenceCustomizableRelative(tf.keras.losses.Loss):
    def __init__(self, wash_out_len, post_wash_out_len, discount_factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.wash_out_len = wash_out_len
        self.post_wash_out_len = post_wash_out_len
        self.discount_factor = discount_factor

        # Calculate discount vector
        discount_vector = np.ones(shape=post_wash_out_len)
        for i in range(post_wash_out_len - 1):
            discount_vector[i + 1] = discount_vector[i] * discount_factor
        self.discount_vector = tf.convert_to_tensor(discount_vector, dtype=tf.float32)

    def call(self, y_true, y_predicted):
        y_predicted = ops.convert_to_tensor_v2_with_dispatch(y_predicted)
        y_true = math_ops.cast(y_true, y_predicted.dtype)
        losses = K.mean(math_ops.squared_difference(y_predicted, y_true) / (math_ops.square(y_true) + 0.01), axis=-1)

        # This discards losses for timesteps ≤ wash_out_len
        losses = losses[:, self.wash_out_len:]

        # Get discounted sum of losses for a time series
        loss = tf.linalg.matvec(losses, self.discount_vector)#/self.post_wash_out_len

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "wash_out_len": self.wash_out_len,
            "post_wash_out_len": self.post_wash_out_len,
            "discount_factor": self.discount_factor
        })
        return config