"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""

"""
This is a predictor for autoregressive RNNs constructed in tensorflowrol
This predictor is good only for one control input being first net input, all other net inputs in the same order
as net outputs, and all net outputs being closed loop, no dt, no target position
horizon cannot be changed in runtime
"""


"""
Using predictor:
1. Initialize while initializing controller
    This step load the RNN - it make take quite a bit of time
    During initialization you only need to provide RNN which should be loaded
2. Call iterativelly three functions
    a) setup(initial_state, horizon, etc.)
    b) predict(Q)
    c) update_net
    
    ad a) at this stage you can change the parameters for prediction like e.g. horizon, dt
            It also prepares 0 state of the prediction, and tensors for saving the results,
            to make b) max performance. This function should be called BEFORE starting solving an optim
    ad b) predict is optimized to get the prediction of future states of the system as fast as possible.
        It accepts control input (vector) as its only input and is intended to be used at every evaluation of the cost functiomn
    ad c) this method updates the internal state of RNN. It accepts control input for current time step (scalar) as its only input
            it should be called only after the optimization problem is solved with the control input used in simulation
            
"""

#TODO: for the moment it is not possible to update RNN more often than mpc dt
#   Updating it more often will lead to false results.

# "Command line" parameters
from SI_Toolkit.TF.TF_Functions.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.TF.TF_Functions.Network import get_internal_states, load_internal_states
from SI_Toolkit.load_and_normalize import *

try:
    from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, CONTROL_INPUTS, augment_predictor_output
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

from types import SimpleNamespace
import yaml

import os
import tensorflow as tf
import logging
#logging.getLogger('tensorflow').setLevel(logging.ERROR)
#tf.get_logger().setLevel(logging.ERROR)

from globals import *
import time as global_time


class predictor_autoregressive_tf:
    def __init__(self, horizon=None, batch_size=None, net_name=None):
        # Neural Network
        config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'), Loader=yaml.FullLoader)
        a = SimpleNamespace()
        a.path_to_models = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config['paths']['path_to_experiment'] + "Models/"
        a.net_name = net_name
        self.net, self.net_info = get_net(a, time_series_length=1, batch_size=batch_size, stateful=True, unroll=False)
        self.normalization_info = get_norm_info_for_net(self.net_info)[self.net_info.outputs]

        # Network sizes
        self.batch_size = batch_size
        self.horizon = horizon
        self.net_input_length = len(self.net_info.inputs)
        self.control_length = len(CONTROL_INPUTS)
        self.state_length = self.net_input_length - self.control_length

        # Helpers
        self.default_internal_states = get_internal_states(self.net)
        self.output_array = np.zeros([self.batch_size, self.horizon+1, len(STATE_VARIABLES)+self.control_length], dtype=np.float32)
        self.state_indices_list = [STATE_INDICES.get(key) for key in self.net_info.outputs]

        # Denormalization
        # normalized = 2 * min
        # denormalized = (normalized + 1) / 2 *  (max-min) + min = normalized * (max-min) / 2 + (max-min) / 2 + min = normalized * (max-min) / 2 + (max+min) / 2
        min = tf.convert_to_tensor(self.normalization_info.loc['min'].to_numpy(), dtype=tf.float32)
        max = tf.convert_to_tensor(self.normalization_info.loc['max'].to_numpy(), dtype=tf.float32)
        #self.normalization_offset = tf.ones(shape=(self.batch_size, self.horizon, self.state_length)) * (max+min) / 2
        #self.normalization_scale = tf.ones(shape=(self.batch_size, self.horizon, self.state_length)) * (max-min) / 2
        self.denormalization_offset = tf.ones(shape=(self.batch_size, self.horizon, self.state_length)) * (max+min) / 2
        self.denormalization_scale = tf.ones(shape=(self.batch_size, self.horizon, self.state_length)) * (max-min) / 2


        #elif normalization_type == 'minmax_sym':
        #    normalized_array[..., feature_idx] = -1.0 + 2.0 * (
        #            (denormalized_array[..., feature_idx]-normalization_info.at['min', features[feature_idx]])
        #            /
        #            (normalization_info.at['max', features[feature_idx]] - normalization_info.at['min', features[feature_idx]])
        #    )


    def setup(self, initial_state: np.array, prediction_denorm=True):
        self.initial_state = initial_state

    def predict(self, Q, single_step=False) -> np.array:
        return self.predict_tf(tf.convert_to_tensor(self.initial_state), tf.convert_to_tensor(Q))

    # Predict (14.1ms)
    def predict_tf(self, initial_state, Q):
        initial_state = initial_state.numpy()
        self.output_array[:, 0, :-self.control_length] = initial_state

        initial_input_net_without_Q = initial_state[..., [STATE_INDICES.get(key) for key in self.net_info.inputs[len(CONTROL_INPUTS):]]]
        self.net_initial_input_without_Q = normalize_numpy_array(initial_input_net_without_Q, self.net_info.inputs[len(CONTROL_INPUTS):], self.normalization_info)

        # [1:] excludes Q which is not included in initial_state_normed
        # As the only feature written with big Q it should be first on each list.
        self.net_initial_input_without_Q_TF = tf.convert_to_tensor(self.net_initial_input_without_Q, tf.float32)
        self.net_initial_input_without_Q_TF = tf.reshape(self.net_initial_input_without_Q_TF, [-1, len(self.net_info.inputs[1:])])

        output_array = self.output_array

        # Assignment (0.29ms)
        output_array[..., :-1, -1] = Q.numpy()

        # load internal RNN state (0.85ms)
        load_internal_states(self.net, self.default_internal_states)

        # Convert (0.06ms)

        # Run NN (2.9ms)
        start = global_time.time()
        net_outputs = self.iterate_net(Q=Q, initial_input=self.net_initial_input_without_Q_TF)
        performance_measurement[3] = global_time.time() - start

        # Update Output (5.2ms)
        start = global_time.time()
        output_array[..., 1:, self.state_indices_list] = net_outputs.numpy()
        performance_measurement[4] = global_time.time() - start

        # Augment (2.0ms)
        start = global_time.time()
        augment_predictor_output(output_array, self.net_info)
        performance_measurement[5] = global_time.time() - start

        return output_array

    def update_internal_state(self, Q0):
        # load internal RNN state
        load_internal_states(self.net, self.default_internal_states)

        # Run current input through network
        Q0 = tf.squeeze(tf.convert_to_tensor(Q0, dtype=tf.float32))
        Q0 = tf.reshape(Q0, [-1, 1])
        if self.net_info.net_type == 'Dense':
            net_input = tf.concat([Q0, self.net_initial_input_without_Q_TF], axis=1)
        else:
            net_input = (tf.reshape(tf.concat([Q0, self.net_initial_input_without_Q_TF], axis=1), [-1, 1, len(self.net_info.inputs)]))
        # self.evaluate_net(self.net_current_input) # Using tf.function to compile net
        self.net(net_input)  # Using net directly

        self.default_internal_states = get_internal_states(self.net)

    @tf.function(experimental_compile=True)
    def iterate_net(self, Q, initial_input):
        Q_current = tf.zeros(shape=(self.batch_size, self.control_length), dtype=tf.float32)
        net_input = tf.zeros(shape=(self.batch_size, self.state_length+self.control_length), dtype=tf.float32)
        net_output = tf.zeros(shape=(self.batch_size, self.state_length), dtype=tf.float32)
        net_outputs = tf.TensorArray(tf.float32, size=self.horizon, dynamic_size=False)

        Q = tf.transpose(Q)

        for i in tf.range(50):
            Q_current = tf.expand_dims(Q[i], axis=1)

            if i == 0:
                net_input = tf.expand_dims(tf.concat([Q_current, initial_input], axis=1), axis=1)
            else:
                net_input = tf.expand_dims(tf.concat([Q_current, net_output], axis=1), axis=1)

            net_output = self.evaluate_net(net_input)
            net_output = tf.squeeze(net_output, axis=1)
            net_outputs = net_outputs.write(i, net_output)

        # Stacking
        output = tf.transpose(net_outputs.stack(), perm=[1, 0, 2])

        # Denormalization
        output = self.denormalization_offset + self.denormalization_scale * output

        return output

    @tf.function(experimental_compile=True)
    def evaluate_net(self, net_input):
        return self.net(net_input)