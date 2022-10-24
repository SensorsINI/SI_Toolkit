"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""

"""
This is a predictor for autoregressive neural network constructed in tensorflow
Control inputs should be first (regarding vector indices) inputs of the vector.
all other net inputs in the same order as net outputs
"""

"""
Using predictor:
1. Initialize while initializing controller
    This step load the RNN - it may take quite a bit of time
    During initialization you need to provide RNN which should be loaded
2. Call iterativelly three functions
    a) setup(initial_state, horizon, etc.)
    b) predict(Q)
    c) update_net
    
    ad a) at this stage you can change the parameters for prediction like e.g. horizon, dt
            It also prepares 0 state of the prediction, and tensors for saving the results,
            to make b) max performance. This function should be called BEFORE starting solving an optimisation problem
    ad b) predict is optimized to get the prediction of future states of the system as fast as possible.
        It accepts control input (vector) as its only input and is intended to be used at every evaluation of the cost functiomn
    ad c) this method updates the internal state of RNN. It accepts control input for current time step (scalar) as its only input
            it should be called only after the optimization problem is solved with the control input used in simulation
            
"""

# TODO: Make horizon updatable in runtime

import os
from types import SimpleNamespace
from SI_Toolkit.Predictors import template_predictor

import numpy as np
# "Command line" parameters
from SI_Toolkit.Functions.General.Initialization import (get_net,
                                                         get_norm_info_for_net)
from SI_Toolkit.Functions.General.Normalising import (
    get_denormalization_function, get_normalization_function,
    get_scaling_function_for_output_of_differential_network)
from SI_Toolkit.Functions.TF.Compile import CompileAdaptive
from SI_Toolkit_ASF.predictors_customization import (CONTROL_INPUTS,
                                                     STATE_INDICES,
                                                     STATE_VARIABLES)
from SI_Toolkit_ASF.predictors_customization_tf import \
    predictor_output_augmentation_tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF

def check_dimensions(s, Q, lib):
    # Make sure the input is at least 2d
    if s is not None:
        if lib.ndim(s) == 1:
            s = s[lib.newaxis, :]

    if lib.ndim(Q) == 3:  # Q.shape = [batch_size, timesteps, features]
        pass
    elif lib.ndim(Q) == 2:  # Q.shape = [timesteps, features]
        Q = Q[lib.newaxis, :, :]
    else:  # Q.shape = [features;  rank(Q) == 1
        Q = Q[lib.newaxis, lib.newaxis, :]

    return s, Q


class predictor_autoregressive_tf(template_predictor):
    def __init__(
        self,
        model_name=None,
        path_to_model=None,
        horizon=None,
        dt=None,
        batch_size=None,
        disable_individual_compilation=False,
        update_before_predicting=True,
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
            a.path_to_models = os.path.join(model_name.split(os.sep)[:-1]) + os.sep
            a.net_name = model_name.split(os.sep)[-1]
        else:
            a.path_to_models = path_to_model
            a.net_name = model_name

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        net, _ = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        if self.net_info.library == 'TF':
            from SI_Toolkit.computation_library import TensorFlowLibrary
            self.lib = TensorFlowLibrary
            from tensorflow import TensorArray
            self.TensorArray = TensorArray
            from SI_Toolkit.Functions.TF.Network import (
                _copy_internal_states_from_ref, _copy_internal_states_to_ref)
        elif self.net_info.library == 'Pytorch':
            from SI_Toolkit.computation_library import PyTorchLibrary
            self.lib = PyTorchLibrary
            from SI_Toolkit.Functions.Pytorch.Network import (
                _copy_internal_states_from_ref, _copy_internal_states_to_ref)
        else:
            raise NotImplementedError('predictor_autoregressive_neural defined only for TF and Pytorch')

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

        self.layers_ref = net.layers

        self.normalization_info = get_norm_info_for_net(self.net_info)

        self.normalize_state = get_normalization_function(self.normalization_info, STATE_VARIABLES, self.lib)
        self.normalize_inputs = get_normalization_function(self.normalization_info, self.net_info.inputs[len(CONTROL_INPUTS):], self.lib)
        self.normalize_control_inputs = get_normalization_function(self.normalization_info, self.net_info.inputs[:len(CONTROL_INPUTS)], self.lib)

        self.indices_inputs_reg = self.lib.to_tensor(
            [STATE_INDICES.get(key) for key in self.net_info.inputs[len(CONTROL_INPUTS):]], dtype=self.lib.int64)

        if self.differential_network:

            self.rescale_output_diff_net = get_scaling_function_for_output_of_differential_network(
                self.normalization_info,
                self.net_info.outputs,
                self.dt,
                self.lib
            )

            outputs_names = np.array([x[2:] for x in self.net_info.outputs])

            self.indices_state_to_output = self.lib.to_tensor([STATE_INDICES.get(key) for key in outputs_names], dtype=self.lib.int64)
            output_indices = {x: np.where(outputs_names == x)[0][0] for x in outputs_names}
            self.indices_output_to_input = self.lib.to_tensor([output_indices.get(key) for key in self.net_info.inputs[len(CONTROL_INPUTS):]], dtype=self.lib.int64)

        else:
            outputs_names = self.net_info.outputs

        self.denormalize_outputs = get_denormalization_function(self.normalization_info, outputs_names, self.lib)
        self.indices_outputs = [STATE_INDICES.get(key) for key in outputs_names]
        self.augmentation = predictor_output_augmentation_tf(self.net_info, self.lib, differential_network=self.differential_network)
        self.indices_augmentation = self.augmentation.indices_augmentation
        self.indices_outputs = self.lib.to_tensor(np.argsort(self.indices_outputs + self.indices_augmentation), dtype=self.lib.int64)

        self.net_input_reg_initial_normed = self.lib.zeros([self.batch_size, len(self.indices_inputs_reg)], dtype=self.lib.float32)
        self.last_initial_state = self.lib.zeros([self.batch_size, len(STATE_VARIABLES)], dtype=self.lib.float32)

        # Next conversion only relevant for TF
        self.net_input_reg_initial_normed = self.lib.to_variable(self.net_input_reg_initial_normed, self.lib.float32)
        self.last_initial_state = self.lib.to_variable(self.last_initial_state, self.lib.float32)

        self.output = np.zeros([self.batch_size, self.horizon + 1, len(STATE_VARIABLES)],
                               dtype=np.float32)

        self.predict_with_update_tf = CompileAdaptive(self._predict_with_update_tf)

        if disable_individual_compilation:
            self.predict_tf = self._predict_tf
            self.update_internal_state_tf = self._update_internal_state_tf
        else:
            self.predict_tf = CompileAdaptive(self._predict_tf)
            self.update_internal_state_tf = CompileAdaptive(self._update_internal_state_tf)

        print('Init done')

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
            output = self.predict_tf(initial_state, Q)

        self.output = self.lib.to_numpy(output)
        return self.output

    def _predict_with_update_tf(self, initial_state, Q, last_initial_state, last_optimal_control_input):
        self._update_internal_state_tf(last_optimal_control_input, last_initial_state)
        return self._predict_tf(initial_state, Q)

    def _predict_tf(self, initial_state, Q):

        self.lib.assign(self.last_initial_state, initial_state)

        net_input_reg_initial = self.lib.gather_last(initial_state, self.indices_inputs_reg)  # [batch_size, features]

        self.lib.assign(self.net_input_reg_initial_normed, self.normalize_inputs(net_input_reg_initial))

        next_net_input = self.net_input_reg_initial_normed

        Q_normed = self.normalize_control_inputs(Q)

        # load internal RNN state if applies
        self.copy_internal_states_from_ref(self.net, self.layers_ref)

        if self.lib.lib == 'TF':
            outputs = self.TensorArray(self.lib.float32, size=self.horizon)
        else:
            outputs = self.lib.zeros([self.batch_size, self.horizon, len(self.net_info.outputs)])

        if self.differential_network:
            initial_state_normed = self.normalize_state(initial_state)
            output = self.lib.gather_last(initial_state_normed, self.indices_state_to_output)

        for i in self.lib.arange(self.horizon):

            Q_current = Q_normed[:, i, :]

            net_input = self.lib.reshape(
                self.lib.concat([Q_current, next_net_input], axis=1),
                shape=[-1, 1, len(self.net_info.inputs)])

            net_output = self.net(net_input)

            net_output = self.lib.reshape(net_output, [-1, len(self.net_info.outputs)])

            if self.differential_network:
                output = output + self.rescale_output_diff_net(net_output)
                next_net_input = self.lib.gather_last(output, self.indices_output_to_input)
            else:
                output = net_output
                next_net_input = net_output

            if self.lib.lib == 'TF':
                outputs = outputs.write(i, output)
            else:
                outputs[:, i, :] = output

        if self.lib.lib == 'TF':
            outputs = self.lib.permute(outputs.stack(), [1, 0, 2])


        outputs = self.denormalize_outputs(outputs)

        # Augment
        outputs_augmented = self.augmentation.augment(outputs)

        outputs_augmented = self.lib.gather_last(outputs_augmented, self.indices_outputs)

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

            net_input_reg = self.lib.gather_last(s, self.indices_inputs_reg)  # [batch_size, features]

            net_input_reg_normed = self.normalize_inputs(net_input_reg)

            Q0_normed = self.normalize_control_inputs(Q0)

            self.copy_internal_states_from_ref(self.net, self.layers_ref)

            net_input = self.lib.reshape(self.lib.concat([Q0_normed[:, 0, :], net_input_reg_normed], axis=1),
                                   [-1, 1, len(self.net_info.inputs)])

            self.net(net_input)  # Using net directly

            self.copy_internal_states_to_ref(self.net, self.layers_ref)


    def reset(self):
        self.last_optimal_control_input = None
        self.last_optimal_control_input = None


if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
predictor = predictor_autoregressive_tf(horizon, batch_size=batch_size, model_name=model_name, update_before_predicting=True, dt=0.01)
'''

    timer_predictor(initialisation)
