
# "Command line" parameters
from SI_Toolkit.TF.TF_Functions.Initialization import get_net, get_norm_info_for_net
# from SI_Toolkit.load_and_normalize import
from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, CONTROL_INPUTS, augment_predictor_output
from types import SimpleNamespace
import yaml
import os
import tensorflow as tf
import numpy as np

from others.p_globals import (
    k, M, m, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength
)

class predictor_ODE_tf:
    def __init__(self, horizon=None, batch_size=None, net_name=None, dt=0.02, intermediate_steps=10, normalization=True):
        self.net_name = net_name
        self.net_type = net_name
        self.batch_size = batch_size
        self.horizon = horizon
        self.dt = dt
        self.intermediate_steps = intermediate_steps
        self.normalization = normalization

        self.initial_state_tf = None
        self.prev_initial_state_tf = None

        self.control_length = len(CONTROL_INPUTS)
        self.state_length =len(STATE_VARIABLES)
        self.state_indices_list = [STATE_INDICES.get(key) for key in STATE_VARIABLES]

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
        output_array[..., :-1, -len(CONTROL_INPUTS):] = Q
        output_array[:, 1:, :len(STATE_VARIABLES)] = net_output.numpy()

        return output_array

    # Predict (Euler: 6.8ms, RNN:10.5ms)
    @tf.function(experimental_compile=True)
    def predict_tf(self, initial_state, Q):

        self.initial_state_tf = tf.tile(tf.expand_dims(initial_state, axis=0), [self.batch_size, 1])

        # Run Iterations
        net_outputs = self.iterate_net(initial_state=self.initial_state_tf, Q=Q)

        return net_outputs

    def update_internal_state(self, Q):
        pass

    @tf.function(experimental_compile=True)
    def iterate_net(self, Q, initial_state):

        net_output = tf.zeros(shape=(self.batch_size, self.state_length), dtype=tf.float32)
        net_outputs = tf.TensorArray(tf.float32, size=self.horizon, dynamic_size=False)

        for i in tf.range(self.horizon):
            Q_current = Q[..., i, :]

            if i == 0:
                net_input = tf.concat([Q_current, initial_state], axis=1)
            else:
                net_input = tf.concat([Q_current, net_output], axis=1)

            net_output = self.euler_net(net_input)

            net_outputs = net_outputs.write(i, net_output)

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
    def angle_wrapping(self, x):
        return tf.math.atan2(tf.math.sin(x), tf.math.cos(x))

    @tf.function(experimental_compile=True)
    def euler_net(self, inputs):
        # Input Order [Q, angle, angleD, angle_cos, angle_sin, position, positionD]
        Q = inputs[:,0]
        x = tf.gather(inputs, [1,2,5,6], axis=1)

        self.t_step = tf.constant(self.dt / self.intermediate_steps, dtype=tf.float32)

        for i in range(self.intermediate_steps):
            x = self.euler(x, Q, self.t_step)

        # Output Order [angle, angleD, angle_cos, angle_sin, position, positionD]
        return tf.stack([self.angle_wrapping(x[:, 0]), x[:, 1], tf.math.cos(x[:,0]), tf.math.sin(x[:,0]), x[:, 2], x[:, 3]], axis=1)