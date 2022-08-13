from SI_Toolkit.GP.Models import load_model
from SI_Toolkit.Functions.TF.Normalising import normalize_tf, denormalize_tf
from SI_Toolkit.Functions.TF.Compile import Compile

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
PATH_TO_MODEL_POSITION = "./SI_Toolkit_ASF/Experiments/PhysicalData-1/Models/GPR_model"
PATH_TO_MODEL_REST = "./SI_Toolkit_ASF/Experiments/PhysicalData-1/Models/SVGP_model"

class predictor_autoregressive_GP:
    def __init__(self, horizon, num_rollouts=1):
        # tf.config.run_functions_eagerly(True)

        self.horizon = horizon
        self.num_rollouts = num_rollouts
        self.model_p = load_model(PATH_TO_MODEL_POSITION)
        self.model_r = load_model(PATH_TO_MODEL_REST)
        self.inputs = ['position', 'positionD', 'angle_sin', 'angle_cos', 'angleD', 'Q']
        self.normalizing_inputs = tf.convert_to_tensor(self.model_p.norm_info[['position', 'positionD', 'angle_sin', 'angle_cos', 'angleD']], dtype=tf.float64)
        self.normalizing_outputs = tf.convert_to_tensor(self.model_p.norm_info[['position', 'positionD', 'angle_sin', 'angle_cos', 'angleD']], dtype=tf.float64)
        self.normalizing_outputs_full = tf.convert_to_tensor(self.model_p.norm_info[STATE_VARIABLES], dtype=tf.float64)

        self.indices_p = [STATE_INDICES.get(key) for key in ['positionD', 'position']]
        self.indices_r = [STATE_INDICES.get(key) for key in ['positionD', 'angle_sin', 'angle_cos', 'angleD']]

        self.initial_state = tf.random.uniform(shape=[self.num_rollouts, 5], dtype=tf.float32)
        Q = tf.random.uniform(shape=[self.num_rollouts, self.horizon, 1], dtype=tf.float32)

        self.outputs = None

        self.predict_tf(self.initial_state, Q)  # CHANGE TO PREDICT FOR NON TF MPPI

    def predict(self, initial_state, Q_seq):
        outputs = self.predict_tf(initial_state, Q_seq)
        return outputs.numpy()

    # @Compile
    def step(self, s, Q):
        p = tf.gather(s, self.indices_p, axis=1)
        r = tf.gather(s, self.indices_r, axis=1)
        p_x = tf.concat([p, Q], axis=1)
        r_x = tf.concat([r, Q], axis=1)
        p = self.model_p.predict_f(p_x)
        r = self.model_r.predict_f(r_x)
        s = tf.concat([r[:, :-1], p[:, :], r[:, -1, tf.newaxis]], axis=1)
        # if self.num_rollouts == 1:
        #     s = tf.expand_dims(s, axis=0)
        print(s)
        return s

    # @Compile
    def predict_tf(self, initial_state, Q_seq):
        # initial_state = tf.expand_dims(initial_state, axis=0)  # COMMENT OUT FOR TF MPPI
        self.outputs = tf.TensorArray(tf.float64, size=self.horizon+1, dynamic_size=False)

        self.initial_state = tf.cast(initial_state, dtype=tf.float64)
        Q_seq = tf.cast(Q_seq, dtype=tf.float64)

        s = tf.gather(self.initial_state, self.model_p.global_indices, axis=1)

        s = normalize_tf(s, self.normalizing_inputs)

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

        self.outputs = denormalize_tf(self.outputs, self.normalizing_outputs)

        # outputs = tf.stack([outputs[..., 0], outputs[..., 1], tf.math.cos(outputs[..., 0]),
        #                     tf.math.sin(outputs[..., 0]), outputs[..., 2], outputs[..., 3]], axis=2)

        self.outputs = tf.stack([tf.math.atan2(self.outputs[..., 2], self.outputs[..., 1]), self.outputs[..., 0], self.outputs[..., 1],
                            self.outputs[..., 2], self.outputs[..., 3], self.outputs[..., 4]], axis=2)

        self.outputs = tf.cast(self.outputs, tf.float32)
        return self.outputs

    def update_internal_state(self, Q):  # this is here to make the get_prediction function happy
        pass


if __name__ == '__main__':
    import timeit

    initialization = '''
from SI_Toolkit.Predictors.predictor_autoregressive_GP_mixed import predictor_autoregressive_GP
import numpy as np
import tensorflow as tf

horizon = 50
num_rollouts = 2000
predictor = predictor_autoregressive_GP(horizon=horizon, num_rollouts=num_rollouts)

initial_state = tf.random.uniform(shape=[num_rollouts, 5], dtype=tf.float32)
Q = tf.random.uniform(shape=[num_rollouts, horizon, 1], dtype=tf.float32)
'''

    code = '''\
predictor.predict_tf(initial_state, Q)
'''

    print(timeit.timeit(code, number=10, setup=initialization) / 10.0)
