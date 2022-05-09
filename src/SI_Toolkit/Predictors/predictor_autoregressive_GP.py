from SI_Toolkit.GP.Models import load_model
from SI_Toolkit.TF.TF_Functions.Normalising import normalize_tf, denormalize_tf

import numpy as np

from types import SimpleNamespace
import os
import yaml

import tensorflow as tf

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, augment_predictor_output
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF

config = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'),
                   Loader=yaml.FullLoader)

# TODO load from config
PATH_TO_MODEL = "./SI_Toolkit_ASF/Experiments/PhysicalData-1/Models/SGPR_model"


class predictor_autoregressive_GP:
    def __init__(self, horizon, num_rollouts=1):
        # tf.config.run_functions_eagerly(True)

        self.horizon = horizon
        self.num_rollouts = num_rollouts
        self.model = load_model(PATH_TO_MODEL)
        self.inputs = self.model.state_inputs + self.model.control_inputs
        self.normalizing_inputs = tf.convert_to_tensor(self.model.norm_info[self.model.control_inputs], dtype=tf.float64)
        self.normalizing_outputs = tf.convert_to_tensor(self.model.norm_info[self.model.outputs], dtype=tf.float64)
        self.normalizing_outputs_full = tf.convert_to_tensor(self.model.norm_info[STATE_VARIABLES], dtype=tf.float64)

        # TODO use Diego's TF mppi

        initial_state = np.random.random(size=[6, ])
        Q = np.random.random(size=[num_rollouts, horizon, 1])

        self.predict(initial_state, Q)

    def predict(self, initial_state, Q_seq):
        outputs = self.predict_tf(initial_state, Q_seq)
        return outputs.numpy()

    @tf.function(input_signature=[tf.TensorSpec(shape=[6, ], dtype=tf.float64),
                                  tf.TensorSpec(shape=[1000, 25, 1], dtype=tf.float64)],
                 jit_compile=True)
    def predict_tf(self, initial_state, Q_seq):
        initial_state = tf.expand_dims(initial_state, axis=0)
        s = tf.cast(initial_state, dtype=tf.float64)
        Q_seq = tf.cast(Q_seq, dtype=tf.float64)

        outputs = tf.TensorArray(tf.float64, size=self.horizon+1, dynamic_size=False)
        s = tf.gather(s, self.model.global_indices, axis=1)

        s = normalize_tf(s, self.normalizing_outputs)

        s = tf.repeat(s, repeats=self.num_rollouts, axis=0)
        outputs = outputs.write(0, s)

        for i in range(self.horizon):
            x = tf.concat([s, Q_seq[:, i, :]], axis=1)

            s, _ = self.model.predict_f(x)
            s = tf.transpose(tf.squeeze(s, axis=2))

            outputs = outputs.write(i+1, s)

        outputs = tf.transpose(outputs.stack(), perm=[1, 0, 2])

        outputs = denormalize_tf(outputs, self.normalizing_outputs)

        # outputs = tf.stack([outputs[..., 0], outputs[..., 1], tf.math.cos(outputs[..., 0]),
        #                     tf.math.sin(outputs[..., 0]), outputs[..., 2], outputs[..., 3]], axis=2)

        outputs = tf.stack([tf.math.atan2(outputs[..., 2], outputs[..., 1]), outputs[..., 0], outputs[..., 1],
                            outputs[..., 2], outputs[..., 3], outputs[..., 4]], axis=2)

        return outputs

    def update_internal_state(self, Q):  # this is here to make the get_prediction function happy
        pass


if __name__ == '__main__':
    import timeit

    initialization = '''
from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
import numpy as np
import tensorflow as tf

horizon = 25
num_rollouts = 2000
predictor = predictor_autoregressive_GP(horizon=horizon, num_rollouts=num_rollouts)

initial_state = np.random.random(size=[6, ])
Q = np.random.random(size=[num_rollouts, horizon, 1])
'''

    code = '''\
predictor.predict(initial_state, Q)
'''

    print(timeit.timeit(code, number=10, setup=initialization) / 10.0)
