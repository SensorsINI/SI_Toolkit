from SI_Toolkit.GP.Models import load_model
from SI_Toolkit.Functions.TF.Normalising import normalize_tf, denormalize_tf
from SI_Toolkit.Functions.TF.Compile import CompileTF

import os
import yaml

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
        self.normalizing_inputs = tf.convert_to_tensor(self.model.norm_info[self.model.state_inputs], dtype=tf.float64)
        self.normalizing_outputs = tf.convert_to_tensor(self.model.norm_info[self.model.outputs], dtype=tf.float64)
        self.normalizing_outputs_full = tf.convert_to_tensor(self.model.norm_info[STATE_VARIABLES], dtype=tf.float64)

        self.indices = [STATE_INDICES.get(key) for key in self.model.outputs]

        initial_state = np.random.random(size=[6, ])  # CHANGE SHAPE FOR TF MPPI
        Q = np.random.random(size=[self.num_rollouts, self.horizon, 1])

        self.predict(initial_state, Q)  # CHANGE TO PREDICT FOR NON TF MPPI

    def predict(self, initial_state, Q_seq):
        outputs = self.predict_tf(initial_state, Q_seq)
        return outputs.numpy()

    def step(self, s, Q):
        x = tf.concat([s, Q], axis=1)
        s = self.model.predict_f(x)
        # if self.num_rollouts == 1:
        #    s = tf.expand_dims(s, axis=0)
        return s

    @CompileTF
    def predict_tf(self, initial_state, Q_seq):
        initial_state = tf.expand_dims(initial_state, axis=0)
        s = tf.cast(initial_state, dtype=tf.float64)
        Q_seq = tf.cast(Q_seq, dtype=tf.float64)

        outputs = tf.TensorArray(tf.float64, size=self.horizon+1, dynamic_size=False)
        s = tf.gather(s, self.indices, axis=1)

        s = normalize_tf(s, self.normalizing_inputs)

        s = tf.repeat(s, repeats=self.num_rollouts, axis=0)
        outputs = outputs.write(0, s)
        s = self.step(s, Q_seq[:, 0, :])
        outputs = outputs.write(1, s)
        for i in tf.range(1, self.horizon):
            # x = tf.concat([s, Q_seq[:, i, :]], axis=1)
            s = self.step(s, Q_seq[:, i, :])
            # s = tf.transpose(tf.squeeze(s, axis=2))

            outputs = outputs.write(i+1, s)

        outputs = tf.transpose(outputs.stack(), perm=[1, 0, 2])

        outputs = denormalize_tf(outputs, self.normalizing_outputs)

        # outputs = tf.stack([outputs[..., 0], outputs[..., 1], tf.math.cos(outputs[..., 0]),
        #                     tf.math.sin(outputs[..., 0]), outputs[..., 2], outputs[..., 3]], axis=2)

        outputs = tf.stack([tf.math.atan2(outputs[..., 2], outputs[..., 1]), outputs[..., 0], outputs[..., 1],
                            outputs[..., 2], outputs[..., 3], outputs[..., 4]], axis=2)

        # outputs = tf.cast(outputs, tf.float32)
        return outputs

    def update_internal_state(self, Q):  # this is here to make the get_prediction function happy
        pass


if __name__ == '__main__':

    initialization = '''
from SI_Toolkit.Predictors.predictor_autoregressive_GP_old import predictor_autoregressive_GP
import numpy as np
import tensorflow as tf

horizon = 35
num_rollouts = 1000
predictor = predictor_autoregressive_GP(horizon=horizon, num_rollouts=num_rollouts)

initial_state = np.random.random(size=[6, ])
Q = np.random.random(size=[num_rollouts, horizon, 1])
'''

    code = '''\
predictor.predict(initial_state, Q)
'''

    # print(timeit.timeit(code, number=10, setup=initialization) / 10.0)

    import matplotlib.pyplot as plt
    from SI_Toolkit.Predictors.predictor_autoregressive_GP_old import predictor_autoregressive_GP
    import numpy as np
    import tensorflow as tf

    horizon = 35
    num_rollouts = 10
    predictor = predictor_autoregressive_GP(horizon=horizon, num_rollouts=num_rollouts)

    initial_state = np.array([np.pi, 0, -1, 0, 0], dtype=np.float32)
    Q = np.random.uniform(low=-1.0, high=1.0, size=[num_rollouts, horizon, 1])
    for i in range(num_rollouts):
        plt.plot(range(horizon), Q[i, :, 0])
    plt.show()

    s = predictor.predict(initial_state, Q)
    for i in range(num_rollouts):
        plt.plot(range(horizon+1), s[i, :, 0])
    plt.show()