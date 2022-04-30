from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES

from SI_Toolkit_ApplicationSpecificFiles.predictors_customization_tf import next_state_predictor_ODE_tf
from SI_Toolkit.TF.TF_Functions.Compile import Compile

import tensorflow as tf
import numpy as np


def check_dimensions(s, Q):
    # Make sure the input is at least 2d
    if s is None:
        pass
    elif s.ndim == 1:
        s = s[np.newaxis, :]

    if Q is None:
        pass
    elif Q.ndim == 3:  # Q.shape = [batch_size, timesteps, features]
        pass
    elif Q.ndim == 2:  # Q.shape = [timesteps, features]
        Q = Q[np.newaxis, :, :]
    else:  # Q.shape = [features;  tf.rank(Q) == 1
        Q = Q[np.newaxis, np.newaxis, :]

    return s, Q


def convert_to_tensors(s, Q):
    return tf.convert_to_tensor(s, dtype=tf.float32), tf.convert_to_tensor(Q, dtype=tf.float32)


class predictor_ODE_tf:
    def __init__(self, horizon=None, dt=0.02, intermediate_steps=10):

        self.horizon = tf.convert_to_tensor(horizon)
        self.batch_size = None  # Will be adjusted the control input size

        self.initial_state = tf.zeros(shape=(1, len(STATE_VARIABLES)))
        self.output = None

        self.dt = dt
        self.intermediate_steps = intermediate_steps

        self.next_step_predictor = next_state_predictor_ODE_tf(dt, intermediate_steps)

    def predict(self, initial_state, Q):

        initial_state, Q = check_dimensions(initial_state, Q)

        initial_state, Q = convert_to_tensors(initial_state, Q)

        self.batch_size = tf.shape(Q)[0]
        self.initial_state = initial_state

        # I hope this commented fragment can be converted to one graph with Compile, merging with predict_tf.
        # But it throws an error.
        # self.initial_state = tf.cond(
        #     tf.math.logical_and(tf.equal(tf.shape(self.initial_state)[0], 1), tf.not_equal(tf.shape(Q)[0], 1)),
        #     lambda: tf.tile(self.initial_state, (self.batch_size, 1)),
        #     lambda: self.initial_state
        # )

        if tf.shape(self.initial_state)[0] == 1 and tf.shape(Q)[0] != 1:  # Predicting multiple control scenarios for the same initial state
            output = self.predict_tf_tile(self.initial_state, Q, self.batch_size)
        else:  # tf.shape(self.initial_state)[0] == tf.shape(Q)[0]:  # For each control scenario there is separate initial state provided
            output = self.predict_tf(self.initial_state, Q)

        return output.numpy()


    @Compile
    def predict_tf_tile(self, initial_state, Q, batch_size, params=None): # Predicting multiple control scenarios for the same initial state
        initial_state = tf.tile(initial_state, (batch_size, 1))
        return self.predict_tf(initial_state, Q, params=params)


    # Predict (Euler: 6.8ms)
    @Compile
    def predict_tf(self, initial_state, Q, params=None):

        self.output = tf.TensorArray(tf.float32, size=self.horizon + 1, dynamic_size=False)
        self.output = self.output.write(0, initial_state)

        next_state = initial_state

        for k in tf.range(self.horizon):
            next_state = self.next_step_predictor.step(next_state, Q[:, k, :], params)
            self.output = self.output.write(k + 1, next_state)

        self.output = tf.transpose(self.output.stack(), perm=[1, 0, 2])

        return self.output

    def update_internal_state(self, Q, s=None):
        pass





if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
predictor = predictor_ODE_tf(horizon, 0.02, 10)
'''

    timer_predictor(initialisation)
