# NumpyNetworks.py

import numpy as np
import tensorflow as tf


###############################################################################
# 1) Define NumPy-based GRU and Dense layers
###############################################################################

class NumpyGRULayer:
    """
    A single GRU layer in NumPy, matching Keras default gate order: [z, r, h]
    for reset_after=True, implementation=2.

    Attributes:
      - W_xz, W_xr, W_xh: (input_dim, units)
      - W_hz, W_hr, W_hh: (units,     units)
      - b_xz, b_xr, b_xh: (units,)
      - b_hz, b_hr, b_hh: (units,)
      - return_sequences: bool
    """

    def __init__(self, kernel, recurrent_kernel, bias, units, return_sequences=False):
        """
        :param kernel:            (input_dim, 3*units)
        :param recurrent_kernel:  (units, 3*units)
        :param bias:             (2, 3*units) if reset_after=True
        :param units: number of hidden units
        :param return_sequences:  if True, return full sequence; else return last state
        """
        self.units = units
        self.return_sequences = return_sequences

        # Gate order for Keras GRU reset_after=True is [z, r, h]
        #  kernel[:, 0:units] -> x->z
        #  kernel[:, u:2u]    -> x->r
        #  kernel[:, 2u:3u]   -> x->h
        W_xz = kernel[:, 0:units]
        W_xr = kernel[:, units:2 * units]
        W_xh = kernel[:, 2 * units:3 * units]

        W_hz = recurrent_kernel[:, 0:units]
        W_hr = recurrent_kernel[:, units:2 * units]
        W_hh = recurrent_kernel[:, 2 * units:3 * units]

        # bias has shape (2, 3*units): first row is input bias, second row is recurrent bias
        b_xz = bias[0, 0:units]
        b_xr = bias[0, units:2 * units]
        b_xh = bias[0, 2 * units:3 * units]

        b_hz = bias[1, 0:units]
        b_hr = bias[1, units:2 * units]
        b_hh = bias[1, 2 * units:3 * units]

        # Store them as attributes
        self.W_xz, self.W_xr, self.W_xh = W_xz, W_xr, W_xh
        self.W_hz, self.W_hr, self.W_hh = W_hz, W_hr, W_hh

        self.b_xz, self.b_xr, self.b_xh = b_xz, b_xr, b_xh
        self.b_hz, self.b_hr, self.b_hh = b_hz, b_hr, b_hh

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x_seq, h_init=None):
        """
        Forward pass for a sequence:
        :param x_seq: shape = (batch_size, timesteps, input_dim)
        :param h_init: optional initial hidden state, shape=(batch_size, units)
                       default is zeros if None
        :return: either
           - (batch_size, timesteps, units) if return_sequences=True
           - (batch_size, units)            if return_sequences=False
        """
        batch_size, timesteps, input_dim = x_seq.shape
        if h_init is None:
            h_init = np.zeros((batch_size, self.units), dtype=np.float32)

        h = h_init
        outputs = []

        for t in range(timesteps):
            x_t = x_seq[:, t, :]  # shape: (batch_size, input_dim)

            # GRU cell logic
            z = self._sigmoid(x_t @ self.W_xz + h @ self.W_hz + self.b_xz + self.b_hz)
            r = self._sigmoid(x_t @ self.W_xr + h @ self.W_hr + self.b_xr + self.b_hr)
            h_tilde = np.tanh(x_t @ self.W_xh + self.b_xh + r * (h @ self.W_hh + self.b_hh))

            h = (1 - z) * h_tilde + z * h
            outputs.append(h)

        outputs = np.stack(outputs, axis=1)  # shape: (batch_size, timesteps, units)

        if self.return_sequences:
            return outputs
        else:
            # Return only the last hidden state
            return outputs[:, -1, :]


class NumpyDense:
    """
    Simple fully connected layer: y = x * W + b
    """

    def __init__(self, W, b):
        # W.shape = (in_features, out_features)
        # b.shape = (out_features,)
        self.W = W
        self.b = b

    def forward(self, x):
        """
        :param x: shape = (batch_size, in_features)
                  or (batch_size, timesteps, in_features) if we want time-distributed
        :return: shape = (batch_size, out_features)
                  or (batch_size, timesteps, out_features) if time-distributed
        """
        if x.ndim == 2:
            # (batch_size, in_features)
            return x @ self.W + self.b
        elif x.ndim == 3:
            # (batch_size, timesteps, in_features)
            batch_size, timesteps, in_feats = x.shape
            # We'll apply the dense to each time step
            out = []
            for t in range(timesteps):
                out_t = x[:, t, :] @ self.W + self.b
                out.append(out_t)
            # stack
            out = np.stack(out, axis=1)  # (batch_size, timesteps, out_features)
            return out
        else:
            raise ValueError("Dense layer forward() supports only 2D or 3D input.")


###############################################################################
# 2) Wrap multiple GRU layers + final Dense into a single "NumpyGRUNetwork"
###############################################################################

class NumpyGRUNetwork:
    """
    A stack of GRU layers, optionally followed by Dense layers,
    matching the structure of a corresponding TF Keras model.
    """

    def __init__(self, keras_model):
        """
        :param keras_model: a tf.keras.Sequential (or similar) containing:
              - One or more GRU layers (with reset_after=True)
              - Possibly one or more Dense layers
        We parse the layers in order and build equivalent NumPy layers.
        """
        self.layers = []

        for layer in keras_model.layers:
            layer_type = type(layer).__name__

            if layer_type == 'GRU':
                # get GRU parameters
                kernel, recurrent_kernel, bias = layer.get_weights()
                units = layer.units
                return_sequences = layer.return_sequences
                numpy_layer = NumpyGRULayer(
                    kernel, recurrent_kernel, bias,
                    units=units,
                    return_sequences=return_sequences
                )
                self.layers.append(numpy_layer)

            elif layer_type == 'Dense':
                W, b = layer.get_weights()
                numpy_layer = NumpyDense(W, b)
                self.layers.append(numpy_layer)

            else:
                raise NotImplementedError(f"Layer type {layer_type} not supported in NumpyGRUNetwork.")

    def forward(self, x, h_inits=None):
        """
        :param x: shape = (batch_size, timesteps, input_dim)
        :param h_inits: list of initial hidden states, one per GRU layer
                        (optional). If None, zeros are used.
        :return: final output from the last layer
        """
        if h_inits is None:
            h_inits = [None] * len(self.layers)

        out = x
        gru_layer_index = 0

        for layer in self.layers:
            if isinstance(layer, NumpyGRULayer):
                out = layer.forward(out, h_init=h_inits[gru_layer_index])
                gru_layer_index += 1
            elif isinstance(layer, NumpyDense):
                out = layer.forward(out)
            else:
                raise ValueError("Unknown layer in forward pass.")

        return out


###############################################################################
# 3) Testing Script
###############################################################################

def test_gru_network(num_gru_layers=2, units_per_layer=[8, 4], input_dim=3, output_dim=2):
    """
    Build a Keras model with multiple GRU layers + final Dense.
    Compare outputs with the NumpyGRUNetwork for random inputs.

    :param num_gru_layers: integer, how many GRU layers
    :param units_per_layer: list of integers, units for each GRU layer
    :param input_dim: dimension of input
    :param output_dim: dimension of final Dense output
    """
    # 1) Build a Keras Sequential model with `num_gru_layers` GRUs
    model = tf.keras.Sequential()

    # First GRU layer (must define batch_input_shape or input_shape)
    model.add(tf.keras.layers.GRU(units_per_layer[0],
                                  return_sequences=(num_gru_layers > 1),
                                  reset_after=True,
                                  input_shape=(None, input_dim)))

    # Additional GRU layers
    for i in range(1, num_gru_layers):
        return_sequences = (i < num_gru_layers - 1)  # only last GRU returns False
        model.add(tf.keras.layers.GRU(units_per_layer[i],
                                      return_sequences=return_sequences,
                                      reset_after=True))

    # Final Dense layer
    model.add(tf.keras.layers.Dense(output_dim))

    # 2) Build (initialize) the model
    model.build()

    # 3) Create random input
    batch_size = 2
    timesteps = 5
    x_in = np.random.randn(batch_size, timesteps, input_dim).astype(np.float32)

    # 4) Get TF output
    tf_out = model(x_in).numpy()

    # 5) Build equivalent Numpy network
    numpy_net = NumpyGRUNetwork(model)

    # 6) Get NumPy output
    np_out = numpy_net.forward(x_in)

    # 7) Compare
    diff = np.abs(tf_out - np_out).max()
    print(f"TF output = {tf_out}\n")
    print(f"NumPy output = {np_out}\n")
    print(f"Max absolute difference = {diff}")
    if diff < 1e-6:
        print("SUCCESS: outputs match very closely!\n")
    else:
        print("WARNING: mismatch found. Check your gate ordering or biases.\n")


if __name__ == "__main__":
    # Example 1: Single-layer GRU, units=4, final Dense output_dim=2
    print("==== Test single-layer GRU ====")
    test_gru_network(num_gru_layers=1, units_per_layer=[4], input_dim=3, output_dim=2)

    # Example 2: Two-layer GRU, [8, 4], final Dense=2
    print("==== Test two-layer GRU ====")
    test_gru_network(num_gru_layers=2, units_per_layer=[8, 4], input_dim=3, output_dim=2)
