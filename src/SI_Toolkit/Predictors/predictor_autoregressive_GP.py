from SI_Toolkit.GP.Models import load_model
from SI_Toolkit.TF.TF_Functions.Normalising import get_normalization_function_tf, get_denormalization_function_tf
from SI_Toolkit.TF.TF_Functions.Compile import Compile

import numpy as np

from types import SimpleNamespace
import os
import yaml

import tensorflow as tf

try:
    from SI_Toolkit_ASF_global.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, augment_predictor_output
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF

config = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'),
                   Loader=yaml.FullLoader)

PATH_TO_MODEL = config["testing"]["PATH_TO_NN"]


class predictor_autoregressive_GP:
    def __init__(self, model_name, horizon, num_rollouts=1, **kwargs):
        # tf.config.run_functions_eagerly(True)
        a = SimpleNamespace()

        if '/' in model_name:
            a.path_to_models = os.path.join(*model_name.split("/")[:-1])+'/'
            a.net_name = model_name.split("/")[-1]
        else:
            a.path_to_models = PATH_TO_MODEL
            a.net_name = model_name

        self.horizon = horizon
        self.num_rollouts = num_rollouts
        self.model = load_model(PATH_TO_MODEL+model_name)
        self.inputs = self.model.state_inputs + self.model.control_inputs

        self.normalize_tf = get_normalization_function_tf(self.model.norm_info,
                                                          self.model.state_inputs)
        self.denormalize_tf = get_denormalization_function_tf(self.model.norm_info,
                                                              self.model.outputs)
        self.indices = [STATE_INDICES.get(key) for key in self.model.outputs]

        self.initial_state = tf.random.uniform(shape=[self.num_rollouts, 6], dtype=tf.float32)
        Q = tf.random.uniform(shape=[self.num_rollouts, self.horizon, 1], dtype=tf.float32)

        self.outputs = None

        self.predict_tf(self.initial_state, Q)  # CHANGE TO PREDICT FOR NON TF MPPI

    def predict(self, initial_state, Q_seq):
        # outputs = self.predict_tf(initial_state, Q_seq)
        outputs = self.predict_tf(initial_state, Q_seq)
        return outputs.numpy()

    @Compile
    def step(self, s, Q):
        x = tf.concat([s, Q], axis=1)
        s, _ = self.model.predict_f(x)
        # if self.num_rollouts == 1:
        #     s = tf.expand_dims(s, axis=0)
        return s

    """
    @Compile
    def step_mean(self, s, Q):
        x = tf.concat([s, Q], axis=1)
        s = self.model.predict_mean(x)
        # if self.num_rollouts == 1:
        #     s = tf.expand_dims(s, axis=0)
        return s
    """

    @tf.function
    def predict_tf(self, initial_state, Q_seq):
        # initial_state = tf.expand_dims(initial_state, axis=0)  # COMMENT OUT FOR TF MPPI
        self.outputs = tf.TensorArray(tf.float64, size=self.horizon+1, dynamic_size=False)

        #self.initial_state = tf.cast(initial_state, dtype=tf.float64)
        self.initial_state = initial_state
        Q_seq = tf.cast(Q_seq, dtype=tf.float64)

        s = tf.gather(self.initial_state, self.indices, axis=1)

        s = self.normalize_tf(s)
        s = tf.cast(s, tf.float64)

        # s = tf.repeat(s, repeats=self.num_rollouts, axis=0)  # COMMENT OUT FOR TF MPPI
        self.outputs = self.outputs.write(0, s)

        s = self.step(s, Q_seq[:, 0, :])

        self.outputs = self.outputs.write(1, s)
        for i in tf.range(1, self.horizon):
            # x = tf.concat([s, Q_seq[:, i, :]], axis=1)
            s = self.step(s, Q_seq[:, i, :])
            # s = tf.transpose(tf.squeeze(s, axis=2))

            self.outputs = self.outputs.write(i+1, s)

        self.outputs = tf.transpose(self.outputs.stack(), perm=[1, 0, 2])

        self.outputs = tf.cast(self.outputs, tf.float32)

        self.outputs = self.denormalize_tf(self.outputs)

        # outputs = tf.stack([outputs[..., 0], outputs[..., 1], tf.math.cos(outputs[..., 0]),
        #                     tf.math.sin(outputs[..., 0]), outputs[..., 2], outputs[..., 3]], axis=2)

        self.outputs = tf.stack([tf.math.atan2(self.outputs[..., 2], self.outputs[..., 1]), self.outputs[..., 0], self.outputs[..., 1],
                            self.outputs[..., 2], self.outputs[..., 3], self.outputs[..., 4]], axis=2)

        return self.outputs

    """
    @Compile
    def predict_tf_mean(self, initial_state, Q_seq):
        # initial_state = tf.expand_dims(initial_state, axis=0)  # COMMENT OUT FOR TF MPPI
        self.outputs = tf.TensorArray(tf.float64, size=self.horizon+1, dynamic_size=False)

        # self.initial_state = tf.cast(initial_state, dtype=tf.float64)
        self.initial_state = initial_state
        Q_seq = tf.cast(Q_seq, dtype=tf.float64)

        s = tf.gather(self.initial_state, self.indices, axis=1)

        s = self.normalize_tf(s)
        s = tf.cast(s, tf.float64)

        # s = tf.repeat(s, repeats=self.num_rollouts, axis=0)  # COMMENT OUT FOR TF MPPI
        self.outputs = self.outputs.write(0, s)

        s = self.step_mean(s, Q_seq[:, 0, :])

        self.outputs = self.outputs.write(1, s)
        for i in tf.range(1, self.horizon):
            # x = tf.concat([s, Q_seq[:, i, :]], axis=1)
            s = self.step_mean(s, Q_seq[:, i, :])
            # s = tf.transpose(tf.squeeze(s, axis=2))

            self.outputs = self.outputs.write(i+1, s)

        self.outputs = tf.transpose(self.outputs.stack(), perm=[1, 0, 2])

        self.outputs = tf.cast(self.outputs, tf.float32)

        self.outputs = self.denormalize_tf(self.outputs)

        # outputs = tf.stack([outputs[..., 0], outputs[..., 1], tf.math.cos(outputs[..., 0]),
        #                     tf.math.sin(outputs[..., 0]), outputs[..., 2], outputs[..., 3]], axis=2)

        self.outputs = tf.stack([tf.math.atan2(self.outputs[..., 2], self.outputs[..., 1]), self.outputs[..., 0], self.outputs[..., 1],
                            self.outputs[..., 2], self.outputs[..., 3], self.outputs[..., 4]], axis=2)

        return self.outputs
    """

    def update_internal_state(self, *args):  # this is here to make the get_prediction function happy
        pass


if __name__ == '__main__':
    import timeit

    initialization = '''
from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
import numpy as np
import tensorflow as tf

horizon = 10
num_rollouts = 1000
predictor = predictor_autoregressive_GP(horizon=horizon, num_rollouts=num_rollouts)

initial_state = tf.random.uniform(shape=[num_rollouts, 6], dtype=tf.float32)
Q = tf.random.uniform(shape=[num_rollouts, horizon, 1], dtype=tf.float32)
'''

    code = '''\
predictor.predict_tf(initial_state, Q)
'''

    print(timeit.timeit(code, number=100, setup=initialization) / 100.0)
