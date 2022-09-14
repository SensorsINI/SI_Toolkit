from SI_Toolkit_ASF.predictors_customization import STATE_VARIABLES

from SI_Toolkit_ASF.predictors_customization_tf import next_state_predictor_ODE_tf
from SI_Toolkit.Functions.TF.Compile import Compile
from SI_Toolkit.Predictors import predictor

import tensorflow as tf


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


class predictor_ODE_tf(predictor):
    def __init__(self, horizon=None, dt=0.02, intermediate_steps=10, disable_individual_compilation=False, batch_size=1, planning_environment=None, **kwargs):
        self.disable_individual_compilation = disable_individual_compilation

        super().__init__(horizon=tf.convert_to_tensor(horizon), batch_size=batch_size)

        self.initial_state = tf.zeros(shape=(1, len(STATE_VARIABLES)))
        self.output = None

        self.dt = dt
        self.intermediate_steps = intermediate_steps

        if planning_environment is None:
            self.next_step_predictor = next_state_predictor_ODE_tf(dt, intermediate_steps, self.batch_size,
                                                                   disable_individual_compilation=True,)
        else:
            self.next_step_predictor = next_state_predictor_ODE_tf(dt, intermediate_steps, self.batch_size,
                                                                   disable_individual_compilation=True,
                                                                   planning_environment=planning_environment)
        if disable_individual_compilation:
            self.predict_tf = self._predict_tf
        else:
            self.predict_tf = Compile(self._predict_tf)


    def predict(self, initial_state, Q):
        initial_state, Q = convert_to_tensors(initial_state, Q)
        initial_state, Q = check_dimensions(initial_state, Q)

        self.batch_size = tf.shape(Q)[0]
        self.initial_state = initial_state

        output = self.predict_tf(self.initial_state, Q)

        return output.numpy()


    def _predict_tf(self, initial_state, Q, params=None):

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
