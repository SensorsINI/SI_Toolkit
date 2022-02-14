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

# "Command line" parameters
from SI_Toolkit.TF.TF_Functions.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.load_and_normalize import *
from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, CONTROL_INPUTS, augment_predictor_output
from types import SimpleNamespace
import yaml
import os
import tensorflow as tf
from globals import *
import time as global_time
from others.p_globals import (
    k, M, m, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength
)

class predictor_autoregressive_tf:
    def __init__(self, horizon=None, batch_size=None, net_name=None, dt=0.02, intermediate_steps=10, use_runge_kutta=False, normalization=True):
        self.net_name = net_name
        self.net_type = net_name
        self.batch_size = batch_size
        self.horizon = horizon
        self.dt = dt
        self.intermediate_steps = intermediate_steps
        self.use_runge_kutta = use_runge_kutta
        self.normalization = normalization

        if net_name == 'EulerTF':
            # Network sizes
            self.control_length = len(CONTROL_INPUTS)
            self.state_length =len(STATE_VARIABLES)
            self.state_indices_list = [STATE_INDICES.get(key) for key in STATE_VARIABLES]
        else:
            # Neural Network
            config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'), Loader=yaml.FullLoader)
            if '/' in net_name:
                self.model_path = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + net_name.rsplit('/', 1)[0] + '/Models/'
                self.net_name = self.net_name.split("/")[-1]
                self.net_type = self.net_name.split("-")[0]
            else:
                self.model_path = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config['paths']['path_to_experiment'] + "Models/"
                self.net_name = self.net_name
                self.net_type = self.net_name.split("-")[0]

            print(self.model_path, self.net_name, self.net_type)

            a = SimpleNamespace()
            #a.path_to_models = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config['paths']['path_to_experiment'] + "Models/"
            #a.net_name = net_name
            a.path_to_models = self.model_path
            a.net_name = self.net_name
            self.net, self.net_info = get_net(a, time_series_length=1, batch_size=batch_size, stateful=True, unroll=False)
            self.normalization_info = get_norm_info_for_net(self.net_info)[self.net_info.outputs]

            # Network sizes
            self.net_input_length = len(self.net_info.inputs)
            self.control_length = len(CONTROL_INPUTS)
            self.state_length = self.net_input_length - self.control_length

            # Helpers
            self.state_indices_list = [STATE_INDICES.get(key) for key in self.net_info.outputs]

            # De/Normalization:
            if self.normalization:
                # normalized = 2 * (denormalize - min) / (max-min) - 1 = denormalized * 2 / (max-min) - 2 * min / (max-min) - 1 = denormalized * 2 / (max-min) -
                # denormalized = (normalized + 1) / 2 *  (max-min) + min = normalized * (max-min) / 2 + (max-min) / 2 + min = normalized * (max-min) / 2 + (max+min) / 2
                min = tf.convert_to_tensor(self.normalization_info.loc['min'].to_numpy(), dtype=tf.float32)
                max = tf.convert_to_tensor(self.normalization_info.loc['max'].to_numpy(), dtype=tf.float32)
                self.normalization_offset = -2 * tf.ones(shape=(self.batch_size, self.state_length)) * tf.math.divide(min, max-min) - tf.ones(shape=(self.batch_size, self.state_length))
                self.normalization_scale = 2 * tf.ones(shape=(self.batch_size, self.state_length)) * tf.math.reciprocal(max-min)
                self.denormalization_offset = tf.ones(shape=(self.batch_size, self.horizon, self.state_length)) * (max+min) / 2
                self.denormalization_scale = tf.ones(shape=(self.batch_size, self.horizon, self.state_length)) * (max-min) / 2
            else:
                self.normalization_offset = tf.zeros(shape=(self.batch_size, self.state_length))
                self.normalization_scale = tf.ones(shape=(self.batch_size, self.state_length))
                self.denormalization_offset = tf.zeros(shape=(self.batch_size, self.horizon, self.state_length))
                self.denormalization_scale = tf.ones(shape=(self.batch_size, self.horizon, self.state_length))

            #print(self.normalization_offset, self.normalization_scale)
            #print(self.denormalization_offset, self.denormalization_scale)

            # RNN States
            if self.net_type == 'GRU':
                self.gru_layers = [layer for layer in self.net.layers if (('gru' in layer.name) or ('lstm' in layer.name) or ('rnn' in layer.name))]
                self.num_gru_layers = len(self.gru_layers)
                self.rnn_internal_states = [tf.zeros(shape=(self.batch_size, layer.units)) for layer in self.gru_layers]
            self.Q_prev = tf.convert_to_tensor(0, dtype=tf.float32)

    # TODO: replace everywhere with predict_tf
    # DEPRECATED: This version is in-efficient since it copies all batches to GPU
    def setup(self, initial_state, prediction_denorm=True):
        self.initial_input = initial_state

    # TODO: replace everywhere with predict_tf
    # DEPRECATED: This version is in-efficient since it copies all batches to GPU
    def predict(self, Q, single_step=False):
        # Predict TF
        net_output = self.predict_tf(tf.convert_to_tensor(self.initial_input[0,...], dtype=tf.float32), tf.convert_to_tensor(Q, dtype=tf.float32))

        # Prepare Deprecated Output
        output_array = np.zeros([self.batch_size, self.horizon + 1, len(STATE_VARIABLES) + self.control_length], dtype=np.float32)
        output_array[:, 0, :-self.control_length] = self.initial_input
        output_array[..., :-1, -1] = Q
        output_array[:, 1:, :6] = net_output.numpy()

        return output_array

    # Predict (Euler: 6.8ms, RNN:10.5ms)
    @tf.function(experimental_compile=True)
    def predict_tf(self, initial_state, Q):
        # Select States
        if self.net_name != 'EulerTF':
            initial_state = tf.gather(initial_state, self.state_indices_list, axis=0)

        self.initial_state_tf = tf.tile(tf.expand_dims(initial_state, axis=0), [self.batch_size, 1])

        # Normalization
        if self.net_name != 'EulerTF':
            self.initial_state_tf = self.normalization_offset + tf.math.multiply(self.normalization_scale, self.initial_state_tf)

        # Run Last Input for RNN States
        if self.net_type == 'GRU':
            # Set RNN States
            for j in range(self.num_gru_layers):
                self.gru_layers[j].states[0].assign(self.rnn_internal_states[j])

            # Apply Last Input
            Q0 = tf.tile(tf.expand_dims(tf.expand_dims(self.Q_prev, axis=0), axis=0), [self.batch_size, 1])
            net_input = tf.reshape(tf.concat([Q0, self.initial_state_tf], axis=1), [-1, 1, len(self.net_info.inputs)])
            self.net(net_input)

            # Get RNN States
            for j in range(self.num_gru_layers):
                self.rnn_internal_states[j] = tf.identity(self.gru_layers[j].states[0])

        # Run Iterations (Euler:ms, RNN:ms)
        start = global_time.time()
        net_outputs = self.iterate_net(initial_state=self.initial_state_tf, Q=Q)
        performance_measurement[1] = global_time.time() - start

        # Denormalization
        if self.net_name != 'EulerTF':
            net_outputs = self.denormalization_offset + tf.math.multiply(self.denormalization_scale, net_outputs)

        # Augmentation
        if self.net_name != 'EulerTF':
            # Network Output: angleD, angle_cos, angle_sin, position, positionD
            # Return: angle, angleD, angle_cos, angle_sin, position, positionD
            net_outputs = tf.stack([tf.math.atan2(net_outputs[...,2], net_outputs[...,1]), net_outputs[...,0], net_outputs[...,1], net_outputs[...,2], net_outputs[...,3], net_outputs[...,4]], axis=2)

        return net_outputs

    def update_internal_state_tf(self, Q0):
        self.Q_prev = Q0

    # TODO: replace everywhere with update_internal_state_tf
    # DEPRECATED: This version is in-efficient since it copies all batches to GPU
    def update_internal_state(self, Q):
        if self.net_name != 'EulerTF':
            if tf.is_tensor(Q):
                self.update_internal_state_tf(tf.convert_to_tensor(Q[0,...], dtype=tf.float32))
            else:
                self.update_internal_state_tf(tf.convert_to_tensor(Q, dtype=tf.float32))

    @tf.function(experimental_compile=True)
    def iterate_net(self, Q, initial_state):
        Q_current = tf.zeros(shape=(self.batch_size, self.control_length), dtype=tf.float32)
        net_input = tf.zeros(shape=(self.batch_size, self.state_length+self.control_length), dtype=tf.float32)
        net_output = tf.zeros(shape=(self.batch_size, self.state_length), dtype=tf.float32)
        net_outputs = tf.TensorArray(tf.float32, size=self.horizon, dynamic_size=False)
        Q = tf.transpose(Q)

        for i in tf.range(self.horizon):
            Q_current = tf.expand_dims(Q[i], axis=1)

            if i == 0:
                net_input = tf.concat([Q_current, initial_state], axis=1)
            else:
                net_input = tf.concat([Q_current, net_output], axis=1)

            if self.net_name != 'EulerTF':
                net_output = self.net(tf.expand_dims(net_input, axis=1))
                net_output = tf.squeeze(net_output, axis=1)
            else:
                net_output = self.euler_net(net_input)

            net_outputs = net_outputs.write(i, net_output)

        # Stacking
        output = tf.transpose(net_outputs.stack(), perm=[1, 0, 2])

        return output

    @tf.function(experimental_compile=True)
    def cartpole_ode(self, x, Q):
        # Coordinates Change (Angles are flipped in respect to https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html)
        angle       = -x[:, 0]
        angleD      = -x[:, 1]
        position    = x[:, 2]
        positionD   = x[:, 3]

        ca = tf.math.cos(angle)
        sa = tf.math.sin(angle)
        f = u_max * Q

        # Cart Friction
        # Equation 34  (https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html)
        # Alternatives: Equation 31, 32, 33
        F_f = - M_fric * positionD

        # Joint Friction
        # Equation 40 (https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html)
        # Alternatives: Equation 35, 39, 38 & 41
        M_f = J_fric * angleD

        # Equation 28-F (https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html)
        positionDD = (
                (
                        m * g * sa * ca  # Gravity
                        - (1 + k) * (
                                f  # Motor Force
                                + (m * L * angleD**2 * sa)  # Movement from Pole
                                + F_f  # Cart Friction
                        )
                        - (M_f * ca / L)  # Joint Friction
                ) / (m * ca**2 - (1 + k) * (M + m))
        )
        #positionDD = tf.zeros_like(sa)

        # Equation 27-F (https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html)
        angleDD = (
            (
                g * sa                  # Gravity
                - positionDD * ca       # Movement from Cart
                - M_f / (m * L)         # Joint Friction
            ) / ((1 + k) * L)
        )

        return tf.stack([-angleD, -angleDD, positionD, positionDD], axis=1)

    @tf.function(experimental_compile=True)
    def euler(self, x, Q, h):
        derivative = self.cartpole_ode(x, Q)
        return x + derivative*h

    @tf.function(experimental_compile=True)
    def runge_kutta(self, x, Q, h):
        k1 = self.cartpole_ode(x, Q)
        k2 = self.cartpole_ode(x + 0.5*k1*h, Q)
        k3 = self.cartpole_ode(x + 0.5*k2*h, Q)
        k4 = self.cartpole_ode(x + k3*h, Q)

        return x + k1*h/6 + k2*h/3 + k3*h/3 + k4*h/6

    @tf.function(experimental_compile=True)
    def angle_wrapping(self, x):
        return tf.math.atan2(tf.math.sin(x), tf.math.cos(x))

    @tf.function(experimental_compile=True)
    def euler_net(self, inputs):
        # Input Order [Q, angle, angleD, angle_cos, angle_sin, position, positionD]
        Q = inputs[:,0]
        x = tf.gather(inputs, [1,2,5,6], axis=1)

        self.t_step = tf.constant(self.dt / self.intermediate_steps, dtype=tf.float32)

        for i in range(self.intermediate_steps):
            if self.use_runge_kutta:
                x = self.runge_kutta(x, Q, self.t_step)
            else:
                x = self.euler(x, Q, self.t_step)

        # Output Order [angle, angleD, angle_cos, angle_sin, position, positionD]
        return tf.stack([self.angle_wrapping(x[:, 0]), x[:, 1], tf.math.cos(x[:,0]), tf.math.sin(x[:,0]), x[:, 2], x[:, 3]], axis=1)