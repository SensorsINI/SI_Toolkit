"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""

"""
Control inputs should be first (regarding vector indices) inputs of the vector.
all other net inputs in the same order as net outputs
"""

# TODO: Make horizon updatable in runtime

import os
from types import SimpleNamespace
from SI_Toolkit.Predictors import template_predictor
from SI_Toolkit.computation_library import TensorFlowLibrary, PyTorchLibrary, NumpyLibrary

import numpy as np
from typing import Optional

from SI_Toolkit.Functions.General.Initialization import (get_net,
                                                         get_norm_info_for_net)
from SI_Toolkit.Functions.General.Normalising import (
    get_denormalization_function, get_normalization_function,
    )
from SI_Toolkit.Functions.General.value_precision import set_value_precision
from SI_Toolkit.Functions.TF.Compile import CompileAdaptive
from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import predictor_output_augmentation

from SI_Toolkit.Predictors.autoregression import autoregression_loop, differential_model_autoregression_helper, check_dimensions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF


class predictor_autoregressive_neural(template_predictor):
    supported_computation_libraries = {TensorFlowLibrary, PyTorchLibrary, NumpyLibrary}  # Overwrites default from parent
    # Numpy library is supported only when it comes to evaluation of hls4ml models

    def __init__(
        self,
        model_name=None,
        path_to_model=None,
        horizon=None,
        dt=None,
        batch_size=None,
        variable_parameters=None,
        disable_individual_compilation=False,
        update_before_predicting=True,
        mode=None,
        hls=False,
        input_quantization='float',
        **kwargs
    ):
        super().__init__(horizon=horizon, batch_size=batch_size)
        self.dt = dt

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
            a.net_name = model_name.split(os.sep)[-1]
        else:
            a.path_to_models = path_to_model + os.sep
            a.net_name = model_name

        # Create a copy of the network suitable for inference (stateful and with sequence length one)

        if hls:
            remove_redundant_dimensions = True
        else:
            remove_redundant_dimensions = False
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True,
                    remove_redundant_dimensions=remove_redundant_dimensions,)

        # Allows to use predictor for simple network evaluation
        self.mode = mode
        if mode == 'simple evaluation':
            self.predictor_initial_input_features = np.array([], dtype='<U9')
            self.predictor_external_input_features = np.array(self.net_info.inputs)
            self.predictor_output_features = np.array(self.net_info.outputs)
            self.horizon = 1

        if hasattr(self.net_info, 'dt'):
            if self.net_info.dt != self.dt:
                print(f'\n dt of the network {self.net_info.dt} s is different from dt requested of the predictor {self.dt}.\n'
                      f'Using dt of the network {self.net_info.dt} s.\n'
                      f'If it is what you intended (e.g. in Brunton test), ignore this message.\n')
            self.dt = self.net_info.dt


        if self.net_info.library == 'TF':
            net, _ = \
                get_net(a, time_series_length=1,
                        batch_size=self.batch_size, stateful=True)
            self.memory_states_ref = net.layers
        elif self.net_info.library == 'Pytorch':
            self.net.reset_internal_states()
            self.memory_states_ref = self.net.return_internal_states()

        if self.net_info.library == 'TF':
            from SI_Toolkit.computation_library import TensorFlowLibrary
            self.lib = TensorFlowLibrary
            from SI_Toolkit.Functions.TF.Network import (
                _copy_internal_states_from_ref, _copy_internal_states_to_ref)
        elif self.net_info.library == 'Pytorch':
            from SI_Toolkit.computation_library import PyTorchLibrary
            self.lib = PyTorchLibrary
            from SI_Toolkit.Functions.Pytorch.Network import (
                _copy_internal_states_from_ref, _copy_internal_states_to_ref)
        else:
            raise NotImplementedError('predictor_autoregressive_neural defined only for TF and Pytorch')

        if hls:
            self.lib = NumpyLibrary

        self.copy_internal_states_from_ref = _copy_internal_states_from_ref
        self.copy_internal_states_to_ref = _copy_internal_states_to_ref

        if np.any(['D_' in output_name for output_name in self.net_info.outputs]):
            self.differential_network = True
            if self.dt is None:
                raise ValueError('Differential network was loaded but timestep dt was not provided to the predictor')
        else:
            self.differential_network = False

        self.update_before_predicting = update_before_predicting
        self.last_net_input_reg_initial = None
        self.last_optimal_control_input = None

        self.predictor_initial_input_indices = {x: np.where(self.predictor_initial_input_features == x)[0][0] for x in self.predictor_initial_input_features}
        self.predictor_external_input_indices = {x: np.where(self.predictor_external_input_features == x)[0][0] for x in self.predictor_external_input_features}
        self.predictor_output_indices = {x: np.where(self.predictor_output_features == x)[0][0] for x in self.predictor_output_features}

        self.model_input_features = self.net_info.inputs
        self.model_output_features = self.net_info.outputs

        self.model_external_input_features = [feature for feature in self.model_input_features if feature in self.predictor_external_input_features]
        self.model_initial_input_features = [feature for feature in self.model_input_features if feature in self.predictor_initial_input_features] 
        
        if self.net_info.normalize:
            self.normalization_info = get_norm_info_for_net(self.net_info)
            self.normalize_state = get_normalization_function(self.normalization_info, self.predictor_initial_input_features, self.lib)
            self.normalize_inputs = get_normalization_function(self.normalization_info, self.model_initial_input_features, self.lib)
            self.normalize_control_inputs = get_normalization_function(self.normalization_info, self.predictor_external_input_features, self.lib)
        else:
            self.normalization_info = None

        self.model_external_input_indices = self.lib.to_tensor(
            [self.predictor_external_input_indices.get(key) for key in self.model_external_input_features], dtype=self.lib.int64)
        self.model_initial_input_indices = self.lib.to_tensor(
            [self.predictor_initial_input_indices.get(key) for key in self.model_initial_input_features], dtype=self.lib.int64)

        if self.differential_network:
            self.dmah: Optional[differential_model_autoregression_helper] = \
                differential_model_autoregression_helper(
                    inputs=self.net_info.inputs,
                    outputs=self.net_info.outputs,
                    normalization_info=self.normalization_info,
                    dt=self.dt,
                    batch_size=self.batch_size,
                    lib=self.lib,
                )
            outputs_names = [(x[2:] if x[:2] == 'D_' else x) for x in self.net_info.outputs]
        else:
            self.dmah: Optional[differential_model_autoregression_helper] = None
            outputs_names = self.net_info.outputs

        if self.net_info.normalize:
            self.denormalize_outputs = get_denormalization_function(self.normalization_info, outputs_names, self.lib)

        self.augmentation = predictor_output_augmentation(self.net_info, self.lib, differential_network=self.differential_network)

        indices_outputs_rev = [self.predictor_output_indices.get(key, np.inf) for key in outputs_names+self.augmentation.features_augmentation]
        missing_indices_output = [i for i in range(len(self.predictor_output_indices)) if i not in indices_outputs_rev]

        self.indices_outputs = self.lib.to_tensor(np.argsort(indices_outputs_rev+missing_indices_output), dtype=self.lib.int64)
        self.indices_outputs = self.indices_outputs[:len(self.indices_outputs)-indices_outputs_rev.count(np.inf)]

        self.missing_outputs = self.lib.zeros((self.batch_size, self.horizon, len(missing_indices_output)))

        self.model_initial_input_normed = self.lib.zeros([self.batch_size, len(self.model_initial_input_indices)], dtype=self.lib.float32)
        self.last_initial_state = self.lib.zeros([self.batch_size, len(self.predictor_initial_input_features)], dtype=self.lib.float32)

        # Next conversion only relevant for TF
        self.model_initial_input_normed = self.lib.to_variable(self.model_initial_input_normed, self.lib.float32)
        self.last_initial_state = self.lib.to_variable(self.last_initial_state, self.lib.float32)

        self.output = np.zeros([self.batch_size, self.horizon + 1, len(self.predictor_output_features)],
                               dtype=np.float32)

        self.input_quantization = input_quantization

        self.AL: autoregression_loop = autoregression_loop(
            model_inputs_len=len(self.net_info.inputs),
            model_outputs_len=len(self.net_info.outputs),
            batch_size=self.batch_size,
            lib=self.lib,
            differential_model_autoregression_helper_instance=self.dmah,
        )

        if not hls:
            self.predict_with_update_tf = CompileAdaptive(self._predict_with_update_tf)  # This was compiled per default before I implemented hls. As for hls one need not compiled version.

            if disable_individual_compilation:
                self.predict_core = self._predict_core
                self.update_internal_state_tf = self._update_internal_state_tf
            else:
                self.predict_core = CompileAdaptive(self._predict_core)
                self.update_internal_state_tf = CompileAdaptive(self._update_internal_state_tf)
        else:
            # Manipulating of internal state currently not implemented.
            # Not clear if supported at all in HLS model
            self.copy_internal_states_from_ref = lambda *args: None
            self.copy_internal_states_to_ref = lambda *args: None
            # Convert network to HLS form
            from SI_Toolkit.HLS4ML.hls4ml_functions import convert_model_with_hls4ml
            self.net, _ = convert_model_with_hls4ml(self.net)
            self.net.compile()
            # Not compilation supported for HLS models
            self.predict_with_update_tf = self._predict_with_update_tf
            self.predict_core = self._predict_core
            self.update_internal_state_tf = self._update_internal_state_tf

    def predict(self, initial_state, Q, last_optimal_control_input=None) -> np.array:

        initial_state = self.lib.to_tensor(initial_state, dtype=self.lib.float32)
        Q = self.lib.to_tensor(Q, dtype=self.lib.float32)

        if last_optimal_control_input is not None:
            last_optimal_control_input = self.lib.to_tensor(last_optimal_control_input, dtype=self.lib.float32)

        initial_state, Q = check_dimensions(initial_state, Q, self.lib)

        if self.update_before_predicting and self.last_initial_state is not None and (
                last_optimal_control_input is not None or self.last_optimal_control_input is not None):
            if last_optimal_control_input is None:
                last_optimal_control_input = self.last_optimal_control_input
            output = self.predict_with_update_tf(initial_state, Q, self.last_initial_state,
                                                     last_optimal_control_input)
        else:
            output = self.predict_core(initial_state, Q)

        self.output = self.lib.to_numpy(output)
        return self.output

    def _predict_with_update_tf(self, initial_state, Q, last_initial_state, last_optimal_control_input):
        self._update_internal_state_tf(last_optimal_control_input, last_initial_state)
        return self._predict_core(initial_state, Q)

    def _predict_core(self, initial_state, Q):

        self.lib.assign(self.last_initial_state, initial_state)

        if self.net_info.normalize:
            initial_state_normed = self.normalize_state(initial_state)
            Q = self.normalize_control_inputs(Q)
        else:
            initial_state_normed = initial_state

        if self.dmah:
            self.dmah.set_starting_point(initial_state_normed)

        self.lib.assign(self.model_initial_input_normed, self.lib.gather_last(initial_state_normed, self.model_initial_input_indices))

        model_initial_input_normed = self.lib.gather_last(initial_state_normed, self.model_initial_input_indices)
        model_external_input_normed = self.lib.gather_last(Q, self.model_external_input_indices)

        if self.input_quantization != 'float':
            model_initial_input_normed = set_value_precision(model_initial_input_normed, self.input_quantization, lib=self.lib)
            model_external_input_normed = set_value_precision(model_external_input_normed, self.input_quantization, lib=self.lib)

        self.lib.assign(self.model_initial_input_normed, model_initial_input_normed)

        # load internal RNN state if applies
        self.copy_internal_states_from_ref(self.net, self.memory_states_ref)

        outputs = self.AL.run(
            model=self.net,
            horizon=self.horizon,
            initial_input=self.model_initial_input_normed,
            external_input_left=model_external_input_normed,
        )

        if self.net_info.normalize:
            outputs = self.denormalize_outputs(outputs)

        # Augment
        outputs_augmented = self.augmentation.augment(outputs)

        outputs_augmented = self.lib.concat([outputs_augmented, self.missing_outputs], axis=-1)

        outputs_augmented = self.lib.gather_last(outputs_augmented, self.indices_outputs)

        if not self.mode == "simple evaluation":
            outputs_augmented = self.lib.concat((initial_state[:, self.lib.newaxis, :], outputs_augmented), axis=1)

        return outputs_augmented

    def update_internal_state(self, Q0=None, s=None):

        s, Q0 = check_dimensions(s, Q0, self.lib)
        if Q0 is not None:
            Q0 = self.lib.to_tensor(Q0, dtype=self.lib.float32)

        if s is None:
            s = self.last_initial_state

        if self.update_before_predicting:
            self.last_optimal_control_input = Q0

            self.lib.assign(self.last_initial_state, s)

        else:
            self.update_internal_state_tf(Q0, s)

    def _update_internal_state_tf(self, Q0, s):

        if self.net_info.net_type == 'Dense':
            pass
        else:

            net_input_reg = self.lib.gather_last(s, self.model_initial_input_indices)  # [batch_size, features]

            if self.net_info.normalize:
                net_input_reg = self.normalize_inputs(net_input_reg)
                Q0 = self.normalize_control_inputs(Q0)
            Q0 = self.lib.gather_last(Q0, self.model_external_input_indices)

            self.copy_internal_states_from_ref(self.net, self.memory_states_ref)

            net_input = self.lib.reshape(self.lib.concat([Q0[:, 0, :], net_input_reg], axis=1),
                                   [-1, 1, len(self.net_info.inputs)])

            self.net(net_input)  # Using net directly

            self.copy_internal_states_to_ref(self.net, self.memory_states_ref)


    def reset(self):
        self.last_optimal_control_input = None
        self.last_optimal_control_input = None


if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_autoregressive_neural import predictor_autoregressive_neural
predictor = predictor_autoregressive_neural(horizon, batch_size=batch_size, model_name=model_name, update_before_predicting=True, dt=0.01)
'''

    timer_predictor(initialisation)
