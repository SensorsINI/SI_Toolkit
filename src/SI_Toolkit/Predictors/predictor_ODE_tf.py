from SI_Toolkit_ASF_global.predictors_customization import STATE_VARIABLES
from SI_Toolkit.TF.TF_Functions.Compile import Compile
from SI_Toolkit_ASF_global.predictors_customization_tf import next_state_predictor_ODE_tf

import tensorflow as tf
import numpy as np


def check_dimensions(s, Q):
    # Make sure the input is at least 2d
    if tf.rank(s) == 1:
        s = s[tf.newaxis, :]

    if tf.rank(Q) == 3:  # Q.shape = [batch_size, timesteps, features]
        pass
    elif tf.rank(Q) == 2:  # Q.shape = [timesteps, features]
        Q = Q[tf.newaxis, :, :]
    else:  # Q.shape = [features;  tf.rank(Q) == 1
        Q = Q[tf.newaxis, tf.newaxis, :]

    return s, Q


def convert_to_tensors(s, Q):
    return tf.convert_to_tensor(s, dtype=tf.float32), tf.convert_to_tensor(Q, dtype=tf.float32)


class predictor_ODE_tf:
    def __init__(self, horizon=None, dt=0.02, intermediate_steps=10, disable_individual_compilation=False):

        self.disable_individual_compilation = disable_individual_compilation

        self.horizon = tf.convert_to_tensor(horizon)
        self.batch_size = None  # Will be adjusted the control input size

        self.initial_state = tf.zeros(shape=(1, len(STATE_VARIABLES)))
        self.output = None

        self.dt = dt
        self.intermediate_steps = intermediate_steps

        self.next_step_predictor = next_state_predictor_ODE_tf(dt, intermediate_steps, disable_individual_compilation=True)

        if disable_individual_compilation:
            self.predict_tf = self._predict_tf
        else:
            self.predict_tf = Compile(self._predict_tf)


    def predict(self, initial_state, Q):
        initial_state, Q = check_dimensions(initial_state, Q)
        initial_state, Q = convert_to_tensors(initial_state, Q)

        self.batch_size = tf.shape(Q)[0]
        self.initial_state = initial_state

        if tf.shape(self.initial_state)[0] == 1 and tf.shape(Q)[
            0] != 1:  # Predicting multiple control scenarios for the same initial state
            output = self.predict_tf_tile(self.initial_state, Q, self.batch_size)
        else:  # tf.shape(self.initial_state)[0] == tf.shape(Q)[0]:  # For each control scenario there is separate initial state provided
            output = self.predict_tf(self.initial_state, Q)

        if self.batch_size > 1:
            return output
        else:
            return tf.squeeze(output)

    # @tf.function(jit_compile=True)
    def predict_tf_tile(self, initial_state, Q, batch_size):  # Predicting multiple control scenarios for the same initial state
        initial_state = tf.tile(initial_state, (batch_size, 1))
        return self.predict_tf(initial_state, Q)

    def _predict_tf(self, initial_state, Q, params = None):

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
