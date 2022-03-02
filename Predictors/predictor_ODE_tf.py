# "Command line" parameters
from SI_Toolkit.TF.TF_Functions.Initialization import get_net, get_norm_info_for_net
# from SI_Toolkit.load_and_normalize import
from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, CONTROL_INPUTS, \
    augment_predictor_output
from types import SimpleNamespace
import yaml
import os

#  FIXME: YOU SHOULD IMPORT IT THROUGH SI_TOOLKIT_APPLICATION...
from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, POSITION_IDX, POSITIOND_IDX
from SI_Toolkit_ApplicationSpecificFiles.predictors_customization_tf import next_state_predictor_ODE_tf

import tensorflow as tf
import numpy as np

from others.p_globals import (
    k, M, m, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength
)


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

        if Q.ndim == 3:  # Q.shape = [batch_size, timesteps, features]
            pass
        elif Q.ndim == 2:  # Q.shape = [timesteps, features]
            Q = Q[np.newaxis, :, :]
        else:  # Q.shape = [features;  tf.rank(Q) == 1
            Q = Q[np.newaxis, np.newaxis, :]

        # Make sure the input is at least 2d
        if initial_state.ndim == 1:
            initial_state = initial_state[np.newaxis, :]

        Q = tf.convert_to_tensor(Q, dtype=tf.float32)
        self.batch_size = tf.shape(Q)[0]
        initial_state = tf.convert_to_tensor(initial_state, dtype=tf.float32)

        self.initial_state = initial_state

        # I hope this commented fragment can be converted to one graph with tf.function, merging with predict_tf.
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

        if self.batch_size > 1:
            return output.numpy()
        else:
            return tf.squeeze(output).numpy()

    @tf.function(experimental_compile=True)
    def predict_tf_tile(self, initial_state, Q, batch_size, params=None): # Predicting multiple control scenarios for the same initial state
        initial_state = tf.tile(initial_state, (batch_size, 1))
        return self.predict_tf(initial_state, Q, params=params)


    # Predict (Euler: 6.8ms)
    @tf.function(experimental_compile=True)
    def predict_tf(self, initial_state, Q, params=None):

        self.output = tf.TensorArray(tf.float32, size=self.horizon + 1, dynamic_size=False)
        self.output = self.output.write(0, initial_state)

        next_state = initial_state

        for k in tf.range(self.horizon):
            next_state = self.next_step_predictor.step(next_state, Q[:, k, :], params)
            self.output = self.output.write(k + 1, next_state)

        self.output = tf.transpose(self.output.stack(), perm=[1, 0, 2])

        return self.output

    def update_internal_state(self, s, Q):
        pass


if __name__ == '__main__':
    import timeit

    initialisation = '''
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import CONTROL_INPUTS
import numpy as np
batch_size = 2000
horizon = 50
predictor = predictor_ODE_tf(horizon, 0.02, 10)
initial_state = np.random.random(size=(batch_size, 6))
# initial_state = np.random.random(size=(1, 6))
Q = np.float32(np.random.random(size=(batch_size, horizon, len(CONTROL_INPUTS))))
predictor.predict(initial_state, Q)
'''

    code = '''\
predictor.predict(initial_state, Q)'''

    print(timeit.timeit(code, number=100, setup=initialisation) / 100.0)
