from SI_Toolkit.GP.Train import load_model
from SI_Toolkit.TF.TF_Functions.Normalising import normalize_tf, denormalize_tf

import numpy as np

from types import SimpleNamespace
import os
import yaml

import tensorflow as tf

try:
    from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, augment_predictor_output
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF

config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config_testing.yml'), 'r'),
                   Loader=yaml.FullLoader)

# TODO load from config
PATH_TO_MODEL = "./SI_Toolkit_ApplicationSpecificFiles/Experiments/GP-experiment/Models/GP_model"


class predictor_autoregressive_GP:
    def __init__(self, horizon):
        self.horizon = horizon
        self.model = load_model(PATH_TO_MODEL)
        self.inputs = self.model.state_inputs + self.model.control_inputs
        self.normalizing_inputs = tf.convert_to_tensor(self.model.norm_info[self.model.control_inputs], dtype=tf.float64)
        self.normalizing_outputs = tf.convert_to_tensor(self.model.norm_info[self.model.outputs], dtype=tf.float64)
        self.normalizing_outputs_full = tf.convert_to_tensor(self.model.norm_info[STATE_VARIABLES], dtype=tf.float64)

    def predict(self, s, Q_seq):
        Q_seq = tf.convert_to_tensor(Q_seq, dtype=tf.float64)
        Q_seq = normalize_tf(Q_seq, self.normalizing_inputs)

        outputs = tf.TensorArray(tf.float64, size=self.horizon+1)

        if s.shape[1] > len(self.model.outputs):  # parsing full state (for mppi inputs)
            full = True
            s = tf.convert_to_tensor(s, tf.float64)
            outputs = outputs.write(0, s)
            s = tf.gather(s, self.model.global_indices, axis=1)
        else:
            full = False
            s = tf.convert_to_tensor(s, tf.float64)
            outputs = outputs.write(0, s)

        s = normalize_tf(s, self.normalizing_outputs)

        for i in range(self.horizon):
            Q_current = Q_seq[:, i, :]

            x = tf.reshape(
                tf.concat([s, Q_current], axis=1),
                shape=[-1, len(self.inputs)])

            s, _ = self.model.predict_f(x)
            s = tf.reshape(s, shape=[-1, len(self.model.outputs)])

            if full:  # parsing full state (for mppi outputs)
                l = tf.unstack(s, axis=1)
                l = l[:2]+tf.unstack(tf.zeros(shape=[3, s.shape[0]], dtype=tf.float64))+l[2:]
                l = tf.transpose(tf.stack(l))
                outputs = outputs.write(i+1, l)
            else:
                outputs = outputs.write(i+1, s)

        outputs = tf.transpose(outputs.stack(), perm=[1, 0, 2])

        if full:  # parsing full state (for mppi outputs)
            outputs = denormalize_tf(outputs, self.normalizing_outputs_full)
        else:
            outputs = denormalize_tf(outputs, self.normalizing_outputs)

        return outputs.numpy()

    def update_internal_state(self, s, Q):  # this is here to make the get_prediction function happy
        pass


if __name__ == '__main__':
    import timeit

    initialisation = '''
from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
import numpy as np

horizon = 10
num_rollouts = 100
predictor = predictor_autoregressive_GP(horizon=horizon)
initial_state = np.random.random(size=(num_rollouts, 6))
Q = np.float64(np.random.random(size=(num_rollouts, horizon, 1)))

predictor.predict(initial_state, Q)
predictor.update_internal_state(initial_state, Q)
'''

    code = '''\
predictor.predict(initial_state, Q)
predictor.update_internal_state(initial_state, Q)'''

    print(timeit.timeit(code, number=10, setup=initialisation) / 10.0)


