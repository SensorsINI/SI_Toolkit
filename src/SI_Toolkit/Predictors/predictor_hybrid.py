"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""

"""
This predictor combines a autoregressive neural network constructed in tensorflow with a noisy ode to make predictions.
Control inputs should be first (regarding vector indices) inputs of the vector.
Network inputs are predictions of the ODE and network outputs are states 
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

try:
    from SI_Toolkit_ASF_global.predictors_customization_tf import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, predictor_output_augmentation_tf
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

#try:
#    from SI_Toolkit.Predictors.predictor_noisy import predictor_noisy
#except ModuleNotFoundError:
#    print('Noisy predictor not available')

try:
    from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
except ModuleNotFoundError:
    print('Noisy predictor not available')

import numpy as np

from types import SimpleNamespace
import os
import yaml

import tensorflow as tf

import timeit

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


class predictor_hybrid:
    def __init__(self, horizon=None, dt=0.02, intermediate_steps=10, batch_size=None, net_name=None):

        self.batch_size = batch_size
        self.horizon = horizon
        self.predictor = predictor_ODE_tf(horizon=1, dt=dt, intermediate_steps=intermediate_steps)  # choose predictor

        a = SimpleNamespace()

        if '/' in net_name:
            a.path_to_models = os.path.join(*net_name.split("/")[:-1]) + '/'
            a.net_name = net_name.split("/")[-1]
        else:
            a.path_to_models = PATH_TO_NN
            a.net_name = net_name

        a.net_name = net_name

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        net, _ = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        self.layers_ref = net.layers

        self.normalization_info = get_norm_info_for_net(self.net_info)

        self.normalizing_inputs = tf.convert_to_tensor(
            self.normalization_info[self.net_info.inputs[len(CONTROL_INPUTS):]], dtype=tf.float32)
        self.normalizing_outputs = tf.convert_to_tensor(self.normalization_info[self.net_info.outputs],
                                                        dtype=tf.float32)

        self.indices_inputs_reg = [STATE_INDICES.get(key[:-5]) for key in self.net_info.inputs[len(CONTROL_INPUTS):]]
        self.indices_inputs_reg = [STATE_INDICES.get(key[:-5]) for key in self.net_info.inputs[len(CONTROL_INPUTS):]]
        self.indices_net_output = [STATE_INDICES.get(key) for key in self.net_info.outputs]
        self.augmentation = predictor_output_augmentation_tf(self.net_info)
        self.indices_augmentation = self.augmentation.indices_augmentation
        self.indices_outputs = tf.convert_to_tensor(np.argsort(self.indices_net_output + self.indices_augmentation))

        self.net_input_reg_initial_normed = tf.Variable(
            tf.zeros([self.batch_size, len(self.indices_inputs_reg)], dtype=tf.float32))
        self.net_input_reg_normed = tf.Variable(
            tf.zeros([self.batch_size, len(self.indices_inputs_reg)], dtype=tf.float32))

        self.output = np.zeros([self.batch_size, self.horizon + 1, len(STATE_VARIABLES)],
                               dtype=np.float32)

        print('Init done')

    def predict(self, initial_state, Q) -> np.array:
        start_time = timeit.default_timer()
        initial_state, Q = check_dimensions(initial_state, Q)

        self.output[:, 0, :] = initial_state
        prediction = self.predictor.predict(initial_state, Q)
        prediction = prediction[:, -1, :]  # only take the latest prediction
        prediction, Q = convert_to_tensors(prediction, Q)
        prediction = check_batch_size(prediction, self.batch_size, 's')
        Q = check_batch_size(Q, self.batch_size, 'Q')

        net_output = self.predict_tf(prediction, Q)
        self.output[..., 1:, :] = net_output.numpy()
        print('predictor_hybrid prediction took ' + str(timeit.default_timer() - start_time) + ' seconds')

        return self.output

    @tf.function(experimental_compile=True)
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
                state = denormalize_tf(net_output, self.normalizing_outputs)
                state = self.augmentation.augment_single_state(state)
                state = tf.gather(state, self.indices_outputs, axis=-1)  # rearrange
                prediction = self.predictor.predict_tf(state, Q_current[:,tf.newaxis,:])
                prediction = prediction[:, -1, :] # only take the latest prediction

                net_input_reg = tf.gather(prediction, self.indices_inputs_reg,
                                                  axis=-1)  # [batch_size, features]
                self.net_input_reg_normed.assign(normalize_tf(
                    net_input_reg, self.normalizing_inputs
                ))

                net_input = tf.reshape(
                    tf.concat([Q_current, self.net_input_reg_normed], axis=1),
                    shape=[-1, 1, len(self.net_info.inputs)])

            net_output = self.net(net_input)  # run Network

            net_output = tf.reshape(net_output, [-1, len(self.net_info.outputs)])
            net_outputs = net_outputs.write(i, net_output)

        net_outputs = tf.transpose(net_outputs.stack(), perm=[1, 0, 2])
        net_outputs = denormalize_tf(net_outputs, self.normalizing_outputs)
        # Augment
        output = self.augmentation.augment(net_outputs)
        output = tf.gather(output, self.indices_outputs, axis=-1)  # rearrange
        return output

    def update_internal_state(self, Q0=None, s=None):
        s, Q0 = check_dimensions(s, Q0)

        if s is not None:
            prediction = self.predictor.predict(s, Q0)
            prediction = prediction[..., -1, :]
            net_input_reg_initial = tf.gather(prediction, self.indices_inputs_reg, axis=-1)
            net_input_reg_initial_normed = normalize_tf(net_input_reg_initial, self.normalizing_inputs)
            net_input_reg_initial_normed = check_batch_size(net_input_reg_initial_normed, self.batch_size, 's')
        else:
            net_input_reg_initial_normed = self.net_input_reg_initial_normed

        if Q0 is not None:
            Q0 = tf.convert_to_tensor(Q0, dtype=tf.float32)

        Q0 = check_batch_size(Q0, self.batch_size, 'Q')

        self.update_internal_state_tf(Q0, net_input_reg_initial_normed)

    @tf.function(experimental_compile=True)
    def update_internal_state_tf(self, Q0, s):

        if self.net_info.net_type == 'Dense':
            pass
        else:
            copy_internal_states_from_ref(self.net, self.layers_ref)

            net_input = tf.reshape(tf.concat([Q0[:, 0, :], s], axis=1),
                                   [-1, 1, len(self.net_info.inputs)])

            self.net(net_input)  # Using net directly

            copy_internal_states_to_ref(self.net, self.layers_ref)


if __name__ == '__main__':
    from SI_Toolkit.Predictors.timer_predictor import timer_predictor

    initialisation = '''
from SI_Toolkit.Predictors.predictor_hybrid import predictor_hybrid
predictor = predictor_hybrid(horizon, 0.02, 10, batch_size=batch_size, net_name=net_name)
    '''

    timer_predictor(initialisation)
