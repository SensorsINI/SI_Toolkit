import os
from types import SimpleNamespace

import tensorflow as tf
import yaml
from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.Functions.General.Normalising import get_denormalization_function, get_normalization_function
from SI_Toolkit.Functions.TF.Compile import CompileTF
from SI_Toolkit.GP.Functions.save_and_load import load_model
from SI_Toolkit.Predictors import template_predictor

try:
    from SI_Toolkit_ASF.predictors_customization import (
        CONTROL_INPUTS, STATE_INDICES, STATE_VARIABLES,
        augment_predictor_output)
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF


class predictor_autoregressive_GP(template_predictor):
    supported_computation_libraries = {TensorFlowLibrary}  # Overwrites default from parent
    
    def __init__(self,
                 model_name=None,
                 path_to_model=None,
                 horizon=None,
                 batch_size=1,
                 **kwargs):

        a = SimpleNamespace()
        
        # Convert to platform-independent paths
        model_name: str = os.path.normpath(model_name)
        if path_to_model is not None:
            path_to_model: str = os.path.normpath(path_to_model)

        if len(model_name.split(os.sep)) > 1:
            model_name_contains_path_to_model = True
        else:
            model_name_contains_path_to_model = False

        if model_name_contains_path_to_model:
            a.path_to_models = os.path.join(*model_name.split(os.sep)[:-1]) + os.sep
            a.model_name = model_name.split(os.sep)[-1]
        else:
            a.path_to_models = path_to_model + os.sep
            a.model_name = model_name

        super().__init__(horizon=horizon, batch_size=batch_size)
        self.lib = TensorFlowLibrary
        self.batch_size = self.batch_size
        self.model = load_model(a.path_to_models + a.model_name)
        self.inputs = self.model.state_inputs + self.model.control_inputs

        self.normalize_tf = get_normalization_function(self.model.norm_info,
                                                       self.model.state_inputs,
                                                       self.lib
                                                       )
        self.denormalize_tf = get_denormalization_function(self.model.norm_info,
                                                           self.model.outputs,
                                                           self.lib
                                                           )
        self.indices = [STATE_INDICES.get(key) for key in self.model.outputs]

        self.initial_state = tf.random.uniform(shape=[self.batch_size, 6], dtype=tf.float32)
        Q = tf.random.uniform(shape=[self.batch_size, self.horizon, 1], dtype=tf.float32)

        self.predict_tf(self.initial_state, Q)  # CHANGE TO PREDICT FOR NON TF MPPI

    def predict(self, initial_state, Q_seq):
        outputs = self.predict_tf(initial_state, Q_seq)
        return outputs.numpy()

    @CompileTF
    def step(self, s, Q):
        x = tf.concat([s, Q], axis=1)
        s, _ = self.model.predict_f(x)
        return s


    @CompileTF
    def predict_tf(self, initial_state, Q_seq):

        outputs = tf.TensorArray(tf.float64, size=self.horizon+1, dynamic_size=False)

        self.initial_state = initial_state
        Q_seq = tf.cast(Q_seq, dtype=tf.float64)

        s = tf.gather(self.initial_state, self.indices, axis=1)

        s = self.normalize_tf(s)
        s = tf.cast(s, tf.float64)

        outputs = outputs.write(0, s)

        s = self.step(s, Q_seq[:, 0, :])

        outputs = outputs.write(1, s)
        for i in tf.range(1, self.horizon):
            s = self.step(s, Q_seq[:, i, :])

            outputs = outputs.write(i+1, s)

        outputs = tf.transpose(outputs.stack(), perm=[1, 0, 2])

        outputs = tf.cast(outputs, tf.float32)

        outputs = self.denormalize_tf(outputs)

        outputs = tf.stack([tf.math.atan2(outputs[..., 2], outputs[..., 1]), outputs[..., 0], outputs[..., 1],
                            outputs[..., 2], outputs[..., 3], outputs[..., 4]], axis=2)

        return outputs

    def update_internal_state(self, *args):
        pass


if __name__ == '__main__':
    import timeit

    initialization = '''
from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
import numpy as np
import tensorflow as tf

horizon = 10
batch_size = 1000
predictor = predictor_autoregressive_GP(horizon=horizon, batch_size=batch_size)

initial_state = tf.random.uniform(shape=[batch_size, 6], dtype=tf.float32)
Q = tf.random.uniform(shape=[batch_size, horizon, 1], dtype=tf.float32)
'''

    code = '''\
predictor.predict_tf(initial_state, Q)
'''

    print(timeit.timeit(code, number=100, setup=initialization) / 100.0)
