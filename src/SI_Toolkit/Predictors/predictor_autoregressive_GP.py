import os
from types import SimpleNamespace

import tensorflow as tf

from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.Functions.General.Normalising import get_denormalization_function, get_normalization_function
from SI_Toolkit.Functions.TF.Compile import CompileTF
from SI_Toolkit.GP.Functions.save_and_load import load_model
from SI_Toolkit.Predictors import template_predictor

try:
    from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import (
        CONTROL_INPUTS, STATE_INDICES, STATE_VARIABLES,
        )
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

from SI_Toolkit.Predictors.autoregression import autoregression_loop

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF


class predictor_autoregressive_GP(template_predictor):
    supported_computation_libraries = {TensorFlowLibrary}  # Overwrites default from parent
    
    def __init__(self,
                 model_name=None,
                 path_to_model=None,
                 horizon=None,
                 batch_size=1,
                 variable_parameters=None,
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

        self.denormalize_tf = get_denormalization_function(self.model.norm_info,
                                                           self.model.outputs,
                                                           self.lib
                                                           )
        self.normalize_state = get_normalization_function(self.model.norm_info, STATE_VARIABLES, self.lib)

        self.indices_inputs_reg = [STATE_INDICES.get(key) for key in self.model.outputs]

        self.initial_state = tf.random.uniform(shape=[self.batch_size, 6], dtype=tf.float32)

        self.model_input_reg_initial_normed = self.lib.zeros([self.batch_size, len(self.indices_inputs_reg)], dtype=self.lib.float32)
        self.last_initial_state = self.lib.zeros([self.batch_size, len(STATE_VARIABLES)], dtype=self.lib.float32)

        # Next conversion only relevant for TF
        self.model_input_reg_initial_normed = self.lib.to_variable(self.model_input_reg_initial_normed, self.lib.float32)
        self.last_initial_state = self.lib.to_variable(self.last_initial_state, self.lib.float32)


        self.AL: autoregression_loop = autoregression_loop(
            model_inputs_len=len(self.inputs),
            model_outputs_len=len(self.model.outputs),
            batch_size=self.batch_size,
            lib=self.lib,
            differential_model_autoregression_helper_instance=None,
        )

    def predict(self, initial_state, Q_seq):
        outputs = self.predict_core(initial_state, Q_seq)
        return outputs.numpy()

    @CompileTF
    def step(self, s, Q):
        x = tf.concat([s, Q], axis=1)
        s, _ = self.model.predict_f(x)
        return s

    def model_for_AL(self, x):
        x = tf.cast(x[:, 0, :], tf.float64)
        s, _ = self.model.predict_f(x)
        s = tf.cast(s, tf.float32)
        return s

    @CompileTF
    def predict_core(self, initial_state, Q):

        self.lib.assign(self.last_initial_state, initial_state)

        initial_state_normed = self.normalize_state(initial_state)

        self.lib.assign(self.model_input_reg_initial_normed, self.lib.gather_last(initial_state_normed, self.indices_inputs_reg))

        outputs = self.AL.run(
            model=self.model_for_AL,
            horizon=self.horizon,
            external_input_right=Q,
            initial_input=self.model_input_reg_initial_normed,
            predictor='gp'
        )

        outputs = self.denormalize_tf(outputs)

        outputs_augmented = tf.stack([tf.math.atan2(outputs[..., 2], outputs[..., 1]), outputs[..., 0], outputs[..., 1],
                            outputs[..., 2], outputs[..., 3], outputs[..., 4]], axis=2)

        outputs_augmented = self.lib.concat((initial_state[:, self.lib.newaxis, :], outputs_augmented), axis=1)

        return outputs_augmented

    def update_internal_state(self, *args):
        pass


if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
predictor = predictor_autoregressive_GP(horizon=horizon, batch_size=batch_size, model_name=GP_name, path_to_model=path_to_model, update_before_predicting=False)
        '''

    timer_predictor(initialisation)
