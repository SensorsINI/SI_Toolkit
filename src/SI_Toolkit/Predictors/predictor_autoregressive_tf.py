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

# "Command line" parameters
from SI_Toolkit.TF.TF_Functions.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.TF.TF_Functions.Normalising import get_normalization_function_tf, get_denormalization_function_tf, \
    get_scaling_function_for_output_of_differential_network
from SI_Toolkit.TF.TF_Functions.Network import _copy_internal_states_from_ref, _copy_internal_states_to_ref
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from SI_Toolkit_ASF_global.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
    CONTROL_INPUTS
from SI_Toolkit_ASF_global.predictors_customization_tf import predictor_output_augmentation_tf
from SI_Toolkit.Predictors import predictor

import numpy as np

from types import SimpleNamespace
import os
import yaml

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF

config = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'),
                   Loader=yaml.FullLoader)

PATH_TO_NN = config['testing']['PATH_TO_NN']


def check_dimensions(s, Q):
    # Make sure the input is at least 2d
    if s is not None:
        if tf.rank(s) == 1:
            s = s[tf.newaxis, :]

    if tf.rank(Q) == 3:  # Q.shape = [batch_size, timesteps, features]
        pass
    elif tf.rank(Q) == 2:  # Q.shape = [timesteps, features]
        Q = Q[tf.newaxis, :, :]
    else:  # Q.shape = [features;  tf.rank(Q) == 1
        Q = Q[tf.newaxis, tf.newaxis, :]

    return s, Q


def convert_to_tensors(s, Q):
    return tf.convert_to_tensor(s, dtype=tf.float32), tf.convert_to_tensor(Q, dtype=tf.float32)


class predictor_autoregressive_tf(predictor):
    def __init__(self, horizon=None, batch_size=None, net_name=None, update_before_predicting=True, disable_individual_compilation=False, dt=None, **kwargs):

        self.batch_size = batch_size
        self.horizon = horizon

        self.dt = dt

        a = SimpleNamespace()

        if '/' in net_name:
            a.path_to_models = os.path.join(*net_name.split("/")[:-1])+'/'
            a.net_name = net_name.split("/")[-1]
        else:
            a.path_to_models = PATH_TO_NN
            a.net_name = net_name

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        net, _ = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

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

        self.normalize_state_tf = get_normalization_function_tf(self.normalization_info, STATE_VARIABLES)
        self.normalize_inputs_tf = get_normalization_function_tf(self.normalization_info, self.net_info.inputs[len(CONTROL_INPUTS):])
        self.normalize_control_inputs_tf = get_normalization_function_tf(self.normalization_info, self.net_info.inputs[:len(CONTROL_INPUTS)])

        self.indices_inputs_reg = tf.convert_to_tensor(
            [STATE_INDICES.get(key) for key in self.net_info.inputs[len(CONTROL_INPUTS):]])

        if self.differential_network:

            self.rescale_output_diff_net = get_scaling_function_for_output_of_differential_network(
                self.normalization_info,
                self.net_info.outputs,
                self.dt
            )

            outputs_names = np.array([x[2:] for x in self.net_info.outputs])

            self.indices_state_to_output = tf.convert_to_tensor([STATE_INDICES.get(key) for key in outputs_names])
            output_indices = {x: np.where(outputs_names == x)[0][0] for x in outputs_names}
            self.indices_output_to_input = tf.convert_to_tensor([output_indices.get(key) for key in self.net_info.inputs[len(CONTROL_INPUTS):]])

        else:
            outputs_names = self.net_info.outputs

        self.denormalize_outputs_tf = get_denormalization_function_tf(self.normalization_info, outputs_names)
        self.indices_outputs = [STATE_INDICES.get(key) for key in outputs_names]
        self.augmentation = predictor_output_augmentation_tf(self.net_info, differential_network=self.differential_network)
        self.indices_augmentation = self.augmentation.indices_augmentation
        self.indices_outputs = tf.convert_to_tensor(np.argsort(self.indices_outputs + self.indices_augmentation))

        self.net_input_reg_initial_normed = tf.Variable(
            tf.zeros([self.batch_size, len(self.indices_inputs_reg)], dtype=tf.float32))
        self.last_initial_state = tf.Variable(tf.zeros([self.batch_size, len(STATE_VARIABLES)], dtype=tf.float32))

        self.output = np.zeros([self.batch_size, self.horizon + 1, len(STATE_VARIABLES)],
                               dtype=np.float32)

        if disable_individual_compilation:
            self.predict_tf = self._predict_tf
            self.update_internal_state_tf = self._update_internal_state_tf
        else:
            self.predict_tf = Compile(self._predict_tf)
            self.update_internal_state_tf = Compile(self._update_internal_state_tf)

        print('Init done')

    def predict(self, initial_state, Q, last_optimal_control_input=None) -> np.array:

        initial_state, Q = convert_to_tensors(initial_state, Q)
        if last_optimal_control_input is not None:
            last_optimal_control_input = tf.convert_to_tensor(last_optimal_control_input, dtype=tf.float32)

        initial_state, Q = check_dimensions(initial_state, Q)

        if self.update_before_predicting and self.last_initial_state is not None and (
                last_optimal_control_input is not None or self.last_optimal_control_input is not None):
            if last_optimal_control_input is None:
                last_optimal_control_input = self.last_optimal_control_input
            output = self.predict_with_update_tf(initial_state, Q, self.last_initial_state,
                                                     last_optimal_control_input)
        else:
            output = self.predict_tf(initial_state, Q)

        self.output = output.numpy()
        return self.output

    @Compile
    def predict_with_update_tf(self, initial_state, Q, last_initial_state, last_optimal_control_input):
        self._update_internal_state_tf(last_optimal_control_input, last_initial_state)
        return self._predict_tf(initial_state, Q)

    def _predict_tf(self, initial_state, Q):

        self.last_initial_state.assign(initial_state)

        net_input_reg_initial = tf.gather(initial_state, self.indices_inputs_reg, axis=-1)  # [batch_size, features]

        self.net_input_reg_initial_normed.assign(
            self.normalize_inputs_tf(net_input_reg_initial)
        )

        next_net_input = self.net_input_reg_initial_normed

        Q_normed = self.normalize_control_inputs_tf(Q)

        # load internal RNN state if applies
        _copy_internal_states_from_ref(self.net, self.layers_ref)

        outputs = tf.TensorArray(tf.float32, size=self.horizon)

        if self.differential_network:
            initial_state_normed = self.normalize_state_tf(initial_state)
            output = tf.gather(initial_state_normed, self.indices_state_to_output, axis=-1)

        for i in tf.range(self.horizon):

            Q_current = Q_normed[:, i, :]

            net_input = tf.reshape(
                tf.concat([Q_current, next_net_input], axis=1),
                shape=[-1, 1, len(self.net_info.inputs)])

            net_output = self.net(net_input)

            net_output = tf.reshape(net_output, [-1, len(self.net_info.outputs)])

            if self.differential_network:
                output = output + self.rescale_output_diff_net(net_output)
                next_net_input = tf.gather(output, self.indices_output_to_input, axis=-1)
            else:
                output = net_output
                next_net_input = net_output
            outputs = outputs.write(i, output)

        outputs = tf.transpose(outputs.stack(), perm=[1, 0, 2])

        outputs = self.denormalize_outputs_tf(outputs)

        # Augment
        outputs_augmented = self.augmentation.augment(outputs)

        outputs_augmented = tf.gather(outputs_augmented, self.indices_outputs, axis=-1)

        outputs_augmented = tf.concat((initial_state[:, tf.newaxis, :], outputs_augmented), axis=1)

        return outputs_augmented

    def update_internal_state(self, Q0=None, s=None):

        s, Q0 = check_dimensions(s, Q0)
        if Q0 is not None:
            Q0 = tf.convert_to_tensor(Q0, dtype=tf.float32)

        if s is None:
            s = self.last_initial_state

        if self.update_before_predicting:
            self.last_optimal_control_input = Q0
            self.last_initial_state.assign(s)
        else:
            self.update_internal_state_tf(Q0, s)

    def _update_internal_state_tf(self, Q0, s):

        if self.net_info.net_type == 'Dense':
            pass
        else:

            net_input_reg = tf.gather(s, self.indices_inputs_reg, axis=-1)  # [batch_size, features]

            net_input_reg_normed = self.normalize_inputs_tf(net_input_reg)

            Q0_normed = self.normalize_control_inputs_tf(Q0)

            _copy_internal_states_from_ref(self.net, self.layers_ref)

            net_input = tf.reshape(tf.concat([Q0_normed[:, 0, :], net_input_reg_normed], axis=1),
                                   [-1, 1, len(self.net_info.inputs)])

            self.net(net_input)  # Using net directly

            _copy_internal_states_to_ref(self.net, self.layers_ref)


    def reset(self):
        self.last_optimal_control_input = None
        self.last_optimal_control_input = None


if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
predictor = predictor_autoregressive_tf(horizon, batch_size=batch_size, net_name=net_name, update_before_predicting=True, dt=0.01)
'''

    timer_predictor(initialisation)
