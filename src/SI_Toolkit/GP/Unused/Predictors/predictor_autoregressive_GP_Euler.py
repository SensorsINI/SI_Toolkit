from SI_Toolkit.GP.Models import load_model
from SI_Toolkit.TF.TF_Functions.Normalising import normalize_tf, denormalize_tf
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

# TODO load from config
PATH_TO_MODEL = "./SI_Toolkit_ASF/Experiments/GP-experiment/Models/SVGP_model"


class predictor_autoregressive_GP:
    def __init__(self, horizon, num_rollouts=1):
        # tf.config.run_functions_eagerly(True)

        self.horizon = horizon
        self.num_rollouts = num_rollouts
        self.model = load_model(PATH_TO_MODEL)
        self.inputs = self.model.state_inputs + self.model.control_inputs
        self.normalizing_inputs = tf.convert_to_tensor(self.model.norm_info[['angleD', 'angle_cos', 'angle_sin', 'positionD']], dtype=tf.float64)
        self.normalizing_position = tf.expand_dims(tf.convert_to_tensor(self.model.norm_info['position'], dtype=tf.float64), axis=1)
        self.normalizing_outputs = tf.convert_to_tensor(self.model.norm_info[['angleD', 'angle_cos', 'angle_sin', 'positionD']], dtype=tf.float64)
        self.normalizing_outputs_full = tf.convert_to_tensor(self.model.norm_info[STATE_VARIABLES], dtype=tf.float64)

        self.indices = [STATE_INDICES.get(key) for key in self.model.outputs]

        self.initial_state = tf.random.uniform(shape=[self.num_rollouts, 6], dtype=tf.float32)
        Q = tf.random.uniform(shape=[self.num_rollouts, self.horizon, 1], dtype=tf.float32)

        self.outputs = None

        self.predict_tf(self.initial_state, Q)  # CHANGE TO PREDICT FOR NON TF MPPI

    def predict(self, initial_state, Q_seq):
        outputs = self.predict_tf(initial_state, Q_seq)
        return outputs.numpy()

    # @Compile
    def step(self, s, p, Q):
        s = normalize_tf(s, self.normalizing_inputs)
        # p = normalize_tf(p, self.normalizing_position)

        x = tf.concat([s, Q], axis=1)
        s = self.model.predict_f(x)

        s = denormalize_tf(s, self.normalizing_inputs)
        # p = denormalize_tf(p, self.normalizing_position)

        p = p + 0.02*s[..., 3]

        # p_denorm = ((p + 1.0) / 2.0) * (0.198 + 0.198) - 0.198
        # dp_denorm = ((s[..., 3] + 1.0) / 2.0)

        # p = p_denorm + dp_denorm*0.02
        # p = -1.0 + 2.0 * (p + 0.198) / (0.198 + 0.198)
        # if self.num_rollouts == 1:
        #    s = tf.expand_dims(s, axis=0)
        return s, p

    # @tf.function
    def predict_tf(self, initial_state, Q_seq):
        # initial_state = tf.expand_dims(initial_state, axis=0)  # COMMENT OUT FOR TF MPPI
        self.outputs = tf.TensorArray(tf.float64, size=self.horizon+1, dynamic_size=False)

        self.initial_state = tf.cast(initial_state, dtype=tf.float64)
        Q_seq = tf.cast(Q_seq, dtype=tf.float64)

        s = tf.gather(self.initial_state, [1, 2, 3, 5], axis=1)
        p = tf.gather(self.initial_state, [4], axis=1)

        # s = tf.repeat(s, repeats=self.num_rollouts, axis=0)  # COMMENT OUT FOR TF MPPI
        self.outputs = self.outputs.write(0, tf.concat([s[..., :-1], p, s[..., -1, tf.newaxis]], axis=1))

        s, p = self.step(s, p, Q_seq[:, 0, :])

        self.outputs = self.outputs.write(1, tf.concat([s[..., :-1], p, s[..., -1, tf.newaxis]], axis=1))
        for i in tf.range(1, self.horizon):
            # x = tf.concat([s, Q_seq[:, i, :]], axis=1)
            s, p = self.step(s, p, Q_seq[:, i, :])
            # s = tf.transpose(tf.squeeze(s, axis=2))

            self.outputs = self.outputs.write(i+1, tf.concat([s[..., :-1], p, s[..., -1, tf.newaxis]], axis=1))

        self.outputs = tf.transpose(self.outputs.stack(), perm=[1, 0, 2])

        # outputs = tf.stack([outputs[..., 0], outputs[..., 1], tf.math.cos(outputs[..., 0]),
        #                     tf.math.sin(outputs[..., 0]), outputs[..., 2], outputs[..., 3]], axis=2)

        self.outputs = tf.stack([tf.math.atan2(self.outputs[..., 2], self.outputs[..., 1]), self.outputs[..., 0], self.outputs[..., 1],
                            self.outputs[..., 2], self.outputs[..., 3],  self.outputs[..., 4]], axis=2)

        self.outputs = tf.cast(self.outputs, tf.float32)
        return self.outputs

    def update_internal_state(self, Q):  # this is here to make the get_prediction function happy
        pass


if __name__ == '__main__':
    import timeit

    initialization = '''
from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
import numpy as np
import tensorflow as tf

horizon = 35
num_rollouts = 2000
predictor = predictor_autoregressive_GP(horizon=horizon, num_rollouts=num_rollouts)

initial_state = tf.random.uniform(shape=[num_rollouts, 5], dtype=tf.float32)
Q = tf.random.uniform(shape=[num_rollouts, horizon, 1], dtype=tf.float32)
'''

    code = '''\
predictor.predict_tf(initial_state, Q)
'''

    print(timeit.timeit(code, number=10, setup=initialization) / 10.0)
