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
from SI_Toolkit.TF.TF_Functions.Normalising import normalize_tf, denormalize_tf
from SI_Toolkit.TF.TF_Functions.Network import copy_internal_states_from_ref, copy_internal_states_to_ref
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from SI_Toolkit_ASF_global.predictors_customization_tf import STATE_VARIABLES, STATE_INDICES, \
    CONTROL_INPUTS, predictor_output_augmentation_tf

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
    if s is None:
        pass
    elif s.ndim == 1:
        s = s[np.newaxis, :]

    if Q is None:
        pass
    elif Q.ndim == 3:  # Q.shape = [batch_size, timesteps, features]
        pass
    elif Q.ndim == 2:  # Q.shape = [timesteps, features]
        Q = Q[np.newaxis, :, :]
    else:  # Q.shape = [features;  tf.rank(Q) == 1
        Q = Q[np.newaxis, np.newaxis, :]

    return s, Q


def check_batch_size(x, batch_size, argument_type):
    if tf.shape(x)[0] != batch_size:
        if tf.shape(x)[0] == 1:
            if argument_type == 's':
                return tf.tile(x, (batch_size, 1))
            elif argument_type == 'Q':
                return tf.tile(x, (batch_size, 1, 1))
        else:
            raise ValueError("Tensor has neither dimension 1 nor the one of the batch size")
    else:
        return x


def convert_to_tensors(s, Q):
    return tf.convert_to_tensor(s, dtype=tf.float32), tf.convert_to_tensor(Q, dtype=tf.float32)


class predictor_autoregressive_tf:
    def __init__(self, horizon=None, batch_size=None, net_name=None, update_before_predicting=True):

        self.batch_size = batch_size
        self.horizon = horizon

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

        self.update_before_predicting = update_before_predicting
        self.last_net_input_reg_initial = None
        self.last_optimal_control_input = None

        self.layers_ref = net.layers

        self.normalization_info = get_norm_info_for_net(self.net_info)

        self.normalizing_inputs = tf.convert_to_tensor(
            self.normalization_info[self.net_info.inputs[len(CONTROL_INPUTS):]], dtype=tf.float32)
        self.normalizing_outputs = tf.convert_to_tensor(self.normalization_info[self.net_info.outputs],
                                                        dtype=tf.float32)

        self.indices_inputs_reg = tf.convert_to_tensor(
            [STATE_INDICES.get(key) for key in self.net_info.inputs[len(CONTROL_INPUTS):]])
        self.indices_net_output = [STATE_INDICES.get(key) for key in self.net_info.outputs]
        self.augmentation = predictor_output_augmentation_tf(self.net_info)
        self.indices_augmentation = self.augmentation.indices_augmentation
        self.indices_outputs = tf.convert_to_tensor(np.argsort(self.indices_net_output + self.indices_augmentation))

        self.net_input_reg_initial = None
        self.net_input_reg_initial_normed = tf.Variable(
            tf.zeros([self.batch_size, len(self.indices_inputs_reg)], dtype=tf.float32))

        self.output = np.zeros([self.batch_size, self.horizon + 1, len(STATE_VARIABLES)],
                               dtype=np.float32)

        print('Init done')

    def predict(self, initial_state, Q, last_optimal_control_input=None) -> np.array:

        initial_state, Q = check_dimensions(initial_state, Q)
        self.output[:, 0, :] = initial_state
        self.batch_size = tf.shape(Q)[0]

        initial_state, Q = convert_to_tensors(initial_state, Q)

        initial_state = check_batch_size(initial_state, self.batch_size, 's')
        Q = check_batch_size(Q, self.batch_size, 'Q')

        if self.update_before_predicting and self.last_net_input_reg_initial is not None and (
                last_optimal_control_input is not None or self.last_optimal_control_input is not None):
            if last_optimal_control_input is None:
                last_optimal_control_input = self.last_optimal_control_input
            net_output = self.predict_with_update_tf(initial_state, Q, self.last_net_input_reg_initial,
                                                     last_optimal_control_input)
        else:
            net_output = self.predict_tf(initial_state, Q)

        self.output[:, 1:, :] = net_output.numpy()

        return self.output

    @Compile
    def predict_with_update_tf(self, initial_state, Q, last_net_input_reg_initial, last_optimal_control_input):
        self.update_internal_state_tf(last_optimal_control_input, last_net_input_reg_initial)
        return self.predict_tf(initial_state, Q)

    @Compile
    def predict_tf(self, initial_state, Q):

        net_input_reg_initial = tf.gather(initial_state, self.indices_inputs_reg, axis=-1)  # [batch_size, features]

        self.net_input_reg_initial_normed.assign(normalize_tf(
            net_input_reg_initial, self.normalizing_inputs
        ))

        # load internal RNN state if applies
        copy_internal_states_from_ref(self.net, self.layers_ref)

        net_outputs = tf.TensorArray(tf.float32, size=self.horizon)
        net_output = tf.zeros(shape=(self.batch_size, len(self.net_info.outputs)), dtype=tf.float32)

        for i in tf.range(self.horizon):

            Q_current = Q[:, i, :]

            if i == 0:
                net_input = tf.reshape(
                    tf.concat([Q_current, self.net_input_reg_initial_normed], axis=1),
                    shape=[-1, 1, len(self.net_info.inputs)])
            else:
                net_input = tf.reshape(
                    tf.concat([Q_current, net_output], axis=1),
                    shape=[-1, 1, len(self.net_info.inputs)])

            net_output = self.net(net_input)

            net_output = tf.reshape(net_output, [-1, len(self.net_info.outputs)])

            net_outputs = net_outputs.write(i, net_output)

        net_outputs = tf.transpose(net_outputs.stack(), perm=[1, 0, 2])

        net_outputs = denormalize_tf(net_outputs, self.normalizing_outputs)

        # Augment
        output = self.augmentation.augment(net_outputs)

        output = tf.gather(output, self.indices_outputs, axis=-1)

        return output

    def update_internal_state(self, Q0=None, s=None):

        s, Q0 = check_dimensions(s, Q0)
        if Q0 is not None:
            Q0 = tf.convert_to_tensor(Q0, dtype=tf.float32)

        if s is not None:
            net_input_reg_initial = tf.gather(tf.convert_to_tensor(s, dtype=tf.float32), self.indices_inputs_reg, axis=-1)
            net_input_reg_initial_normed = normalize_tf(net_input_reg_initial, self.normalizing_inputs)
            net_input_reg_initial_normed = check_batch_size(net_input_reg_initial_normed, self.batch_size, 's')
        else:
            net_input_reg_initial_normed = self.net_input_reg_initial_normed

        Q0 = check_batch_size(Q0, self.batch_size, 'Q')

        if self.update_before_predicting:
            self.last_optimal_control_input = Q0
            self.last_net_input_reg_initial = net_input_reg_initial_normed
        else:
            self.update_internal_state_tf(Q0, net_input_reg_initial_normed)

    @Compile
    def update_internal_state_tf(self, Q0, s):

        if self.net_info.net_type == 'Dense':
            pass
        else:
            copy_internal_states_from_ref(self.net, self.layers_ref)

            net_input = tf.reshape(tf.concat([Q0[:, 0, :], s], axis=1),
                                   [-1, 1, len(self.net_info.inputs)])

            self.net(net_input)  # Using net directly

            copy_internal_states_to_ref(self.net, self.layers_ref)


    def reset(self):
        self.last_optimal_control_input = None
        self.last_optimal_control_input = None


if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
predictor = predictor_autoregressive_tf(horizon, batch_size=batch_size, net_name=net_name)
'''

    timer_predictor(initialisation)
