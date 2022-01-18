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
import time

try:
    from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, CONTROL_INPUTS, augment_predictor_output
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

import numpy as np

from types import SimpleNamespace
import yaml, os

import tensorflow as tf

from globals import *
import time as global_time

config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'), Loader=yaml.FullLoader)

NET_NAME = config['modeling']['NET_NAME']
PATH_TO_MODELS = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config['paths']['path_to_experiment'] + "Models/"

class predictor_autoregressive_tf:
    def __init__(self, horizon=None, batch_size=None, net_name=None):

        a = SimpleNamespace()
        self.batch_size = batch_size
        self._horizon = None  # Helper variable for horizon settoer
        self.horizon = horizon
        a.path_to_models = PATH_TO_MODELS

        a.net_name = net_name

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = get_net(a, time_series_length=1, batch_size=self.batch_size, stateful=True, unroll=False)

        #iterate_net = tf.keras.Sequential()

        #for i in range(self.horizon):
        #    iterate_net.add(tf.keras.layers.GRU(units=32, batch_input_shape=(self.batch_size, 1, 6), return_sequences=True, stateful=True))
        #    iterate_net.add(tf.keras.layers.GRU(units=32, return_sequences=True, stateful=True))
        #    iterate_net.add(tf.keras.layers.Dense(units=5))

        #for i, layer in enumerate(iterate_net.layers):
        #    layer.set_weights(self.net.layers[i % 3].get_weights())

        #iterate_net.summary()
        #self.evaluate_net = evaluate_net

        # def iterate_model(batch_size, horizon, weights):
        #     input = tf.keras.layers.Input(shape=(self.batch_size, 1, 6))
        #
        #     gru_0 = tf.keras.layers.GRU(units=32, batch_input_shape=(self.batch_size, 1, 6), return_sequences=True, stateful=True)
        #     gru_1 = tf.keras.layers.GRU(units=32, return_sequences=True, stateful=True)
        #     dense = tf.keras.layers.Dense(units=5)
        #
        #     for i in self.horizon:
        #
        #
        #     similarity_ab = similarity_layer([taste_a, taste_b])
        #     similarity_ac = similarity_layer([taste_a, taste_c])
        #
        #     output = tf.keras.activations.sigmoid(similarity_ab - similarity_ac)
        #
        #     return tf.keras.Model(inputs=input, outputs=[output])
        #
        # iterate_net = iterate_model(batch_size=self.batch_size, horizon=self.horizon, weights=[l.get_weights() for l in self.net.layers])

        self.normalization_info = get_norm_info_for_net(self.net_info)

        # Make a prediction

        self.rnn_internal_states = get_internal_states(self.net)

        self.net_initial_input_without_Q = np.zeros([len(self.net_info.inputs) - len(CONTROL_INPUTS)], dtype=np.float32)

        self.prediction_denorm = None # Set to True or False in setup, determines if output should be denormalized

        self.output_array = np.zeros([self.batch_size, self.horizon+1, len(STATE_VARIABLES)+len(CONTROL_INPUTS)], dtype=np.float32)

        self.output_array_single_step = np.zeros([self.batch_size, 2, len(STATE_VARIABLES)+1], dtype=np.float32)

        # Retracing tensorflow functions
        try:
            self.evaluate_net = self.evaluate_net_f.get_concrete_function(
                net_input=tf.TensorSpec((self.batch_size, 1, len(self.net_info.inputs)), tf.float32)
            )
        except Exception as e:
            print(e)
            self.evaluate_net = self.evaluate_net_f

        try:
            self.iterate_net = self.iterate_net_f.get_concrete_function(
                Q=tf.TensorSpec((self.batch_size, self.horizon), tf.float32),
                initial_input=tf.TensorSpec((self.batch_size, len(self.net_info.inputs)-1), tf.float32)
            )
        except Exception as e:
            print(e)
            self.iterate_net = self.iterate_net_f

        #self.net.compile()
        #self.net.save('model_single_layer.hdf5')

        print('Init done')

    def setup(self, initial_state: np.array, prediction_denorm=True):

        self.output_array[..., 0, :-1] = initial_state

        initial_input_net_without_Q = initial_state[..., [STATE_INDICES.get(key) for key in self.net_info.inputs[len(CONTROL_INPUTS):]]]
        self.net_initial_input_without_Q = normalize_numpy_array(initial_input_net_without_Q, self.net_info.inputs[len(CONTROL_INPUTS):], self.normalization_info)

        # [1:] excludes Q which is not included in initial_state_normed
        # As the only feature written with big Q it should be first on each list.
        self.net_initial_input_without_Q_TF = tf.convert_to_tensor(self.net_initial_input_without_Q, tf.float32)
        self.net_initial_input_without_Q_TF = tf.reshape(self.net_initial_input_without_Q_TF, [-1, len(self.net_info.inputs[1:])])
        if prediction_denorm:
            self.prediction_denorm = True
        else:
            self.prediction_denorm = False

        # print('Setup done')

    def predict(self, Q, single_step=False) -> np.array:

        if single_step:
            output_array = self.output_array_single_step
        else:
            output_array = self.output_array

        output_array[..., :-1, -1] = Q

        # load internal RNN state if applies
        start = global_time.time()
        load_internal_states(self.net, self.rnn_internal_states)
        performance_measurement[4] = global_time.time() - start

        Q = tf.convert_to_tensor(Q)

        start = global_time.time()
        tf.profiler.experimental.start('tf_logs')
        net_outputs = self.iterate_net(Q=Q, initial_input=self.net_initial_input_without_Q_TF)
        tf.profiler.experimental.stop()
        performance_measurement[3] = global_time.time() - start

        start = global_time.time()
        # Denormalize
        output_array[..., 1:, [STATE_INDICES.get(key) for key in self.net_info.outputs]] = denormalize_numpy_array(net_outputs.numpy(), self.net_info.outputs, self.normalization_info)

        performance_measurement[5] = global_time.time() - start
        # Augment
        augment_predictor_output(output_array, self.net_info)

        return output_array


    # @tf.function
    def update_internal_state(self, Q0):
        # load internal RNN state
        load_internal_states(self.net, self.rnn_internal_states)

        # Run current input through network
        Q0 = tf.squeeze(tf.convert_to_tensor(Q0, dtype=tf.float32))
        Q0 = tf.reshape(Q0, [-1, 1])
        if self.net_info.net_type == 'Dense':
            net_input = tf.concat([Q0, self.net_initial_input_without_Q_TF], axis=1)
        else:
            net_input = (tf.reshape(tf.concat([Q0, self.net_initial_input_without_Q_TF], axis=1), [-1, 1, len(self.net_info.inputs)]))
        # self.evaluate_net(self.net_current_input) # Using tf.function to compile net
        self.net(net_input)  # Using net directly

        self.rnn_internal_states = get_internal_states(self.net)

    @tf.function(experimental_compile=False)
    def iterate_net_f_new(self, Q, initial_input):
        batch_size = Q.shape[0]
        horizon = Q.shape[1]
        states = initial_input.shape[1]

        net_outputs = tf.zeros(shape=(batch_size, horizon, states), dtype=tf.float32)
        net_output = tf.zeros(shape=(batch_size, states), dtype=tf.float32)

        for i in tf.range(0, horizon):
            Q_current = tf.reshape(Q[:, i], [-1, 1])

            if i == 0:
                net_input = (
                    tf.reshape(tf.concat([Q_current, initial_input], axis=1), [-1, 1, states+1]))
            else:
                net_input = tf.reshape(tf.concat([Q_current, net_output], axis=1), [-1, 1, states+1])

            net_output = self.evaluate_net(net_input)
            net_output = tf.reshape(net_output, [-1, states])
            net_outputs[:, i, :] = net_output

        net_outputs = tf.transpose(net_outputs, perm=[1, 0, 2])
        return net_outputs

    @tf.function(experimental_compile=True)
    def iterate_net_f(self, Q, initial_input):
        batch_size = Q.shape[0]
        horizon = Q.shape[1]
        states = initial_input.shape[1]

        net_input = tf.zeros(shape=(batch_size, states+1), dtype=tf.float32)
        net_outputs = tf.TensorArray(tf.float32, size=horizon, dynamic_size=False)
        net_output = tf.zeros(shape=(batch_size, states), dtype=tf.float32)
        Q_current = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)
        Q = tf.transpose(Q)


        for i in tf.range(50):
            Q_current = tf.expand_dims(Q[i], axis=1)

            if i == 0:
                net_input = tf.expand_dims(tf.concat([Q_current, initial_input], axis=1), axis=1)
            else:
                net_input = tf.expand_dims(tf.concat([Q_current, net_output], axis=1), axis=1)

            net_output = self.evaluate_net(net_input)
            net_output = tf.reshape(net_output, [-1, len(self.net_info.outputs)])
            net_outputs = net_outputs.write(i, net_output)

        net_outputs = tf.transpose(net_outputs.stack(), perm=[1, 0, 2])
        return tf.zeros(shape=(batch_size, horizon, states), dtype=tf.float32)

    @tf.function(experimental_compile=False)
    def evaluate_net_f(self, net_input):
        net_output = self.net(net_input)
        return net_output

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, value):
        if self._horizon is None:
            # print('I used initialization setter!')
            self._horizon = value
        else:
            # print('I used normal setter!')
            self._horizon = value
            # self.output_array = np.zeros([self.horizon + 1, self.batch_size, len(STATE_VARIABLES) + 1],
            #                              dtype=np.float32)
            #
            # Q_type = tf.TensorSpec((self.horizon,), tf.float32)
            #
            # initial_input_type = tf.TensorSpec((len(self.net_info.inputs) - 1,), tf.float32)
            #
            # net_input_type = tf.TensorSpec((self.batch_size, len(self.net_info.inputs)), tf.float32)
            #
            # # Retracing tensorflow functions
            # try:
            #     self.evaluate_net = self.evaluate_net_f.get_concrete_function(net_input=net_input_type)
            # except:
            #     self.evaluate_net = self.evaluate_net_f
            #
            # try:
            #     self.iterate_net = self.iterate_net_f.get_concrete_function(Q=Q_type,
            #                                                                 initial_input=initial_input_type)
            #     print(self.iterate_net)
            # except:
            #     self.iterate_net = self.iterate_net_f