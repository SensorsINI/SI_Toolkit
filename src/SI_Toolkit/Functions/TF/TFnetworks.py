# tf_gru_visible_units_test.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints

from SI_Toolkit.Functions.General.NumpyNetworks import NumpyGRULayer

# -------------------------
# TensorFlow counterpart
# -------------------------
class TFGRULayer(tf.keras.layers.Layer):
    """
    TensorFlow reimplementation with visible_units masking at output only.
    Recurrence uses the full hidden state.
    """

    def __init__(
        self,
        kernel,
        recurrent_kernel,
        bias,
        units,
        return_sequences=False,
        *,
        visible_units=None,
        name=None,
        dtype=tf.float32,
    ):
        super().__init__(name=name, dtype=dtype)
        self.units = units
        self.return_sequences = return_sequences

        # weights
        self.W_xz = tf.constant(kernel[:, 0:units], dtype=dtype)
        self.W_xr = tf.constant(kernel[:, units:2 * units], dtype=dtype)
        self.W_xh = tf.constant(kernel[:, 2 * units:3 * units], dtype=dtype)

        self.W_hz = tf.constant(recurrent_kernel[:, 0:units], dtype=dtype)
        self.W_hr = tf.constant(recurrent_kernel[:, units:2 * units], dtype=dtype)
        self.W_hh = tf.constant(recurrent_kernel[:, 2 * units:3 * units], dtype=dtype)

        self.b_xz = tf.constant(bias[0, 0:units], dtype=dtype)
        self.b_xr = tf.constant(bias[0, units:2 * units], dtype=dtype)
        self.b_xh = tf.constant(bias[0, 2 * units:3 * units], dtype=dtype)

        self.b_hz = tf.constant(bias[1, 0:units], dtype=dtype)
        self.b_hr = tf.constant(bias[1, units:2 * units], dtype=dtype)
        self.b_hh = tf.constant(bias[1, 2 * units:3 * units], dtype=dtype)

        # visible_units mask
        if visible_units is None:
            mask = np.ones((units,), dtype=np.float32)
        else:
            assert isinstance(visible_units, int), "visible_units must be int."
            assert 0 < visible_units < units, "visible_units must satisfy 0 < visible_units < units."
            mask = np.zeros((units,), dtype=np.float32)
            mask[:visible_units] = 1.0
        self._mask = tf.constant(mask, dtype=dtype)

    @tf.function(jit_compile=False)
    def call(self, x_seq, h_init=None):
        """
        x_seq: (B, T, D)
        h_init: (B, U) or None
        Returns masked outputs: (B, T, U) or (B, U).
        """
        x_seq = tf.convert_to_tensor(x_seq, dtype=self.dtype)
        B = tf.shape(x_seq)[0]
        T = tf.shape(x_seq)[1]

        h = tf.zeros((B, self.units), dtype=self.dtype) if h_init is None else tf.convert_to_tensor(h_init, dtype=self.dtype)

        ta = tf.TensorArray(self.dtype, size=T, infer_shape=True)
        for t in tf.range(T):
            x_t = x_seq[:, t, :]

            z = tf.sigmoid(tf.matmul(x_t, self.W_xz) + tf.matmul(h, self.W_hz) + self.b_xz + self.b_hz)
            r = tf.sigmoid(tf.matmul(x_t, self.W_xr) + tf.matmul(h, self.W_hr) + self.b_xr + self.b_hr)
            h_tilde = tf.tanh(tf.matmul(x_t, self.W_xh) + self.b_xh + r * (tf.matmul(h, self.W_hh) + self.b_hh))

            h = (1.0 - z) * h_tilde + z * h
            ta = ta.write(t, h)

        outputs = tf.transpose(ta.stack(), perm=[1, 0, 2])  # (B, T, U)

        if self.return_sequences:
            return outputs * self._mask[tf.newaxis, tf.newaxis, :]
        else:
            return outputs[:, -1, :] * self._mask[tf.newaxis, :]


class CustomGRUCell(tf.keras.layers.Layer):
    """GRU cell (reset_after=True, gate order [z,r,h], implementation=2).
    Recurrence uses the full hidden state; masking is applied only to the output."""
    def __init__(self, units, activation="tanh", recurrent_activation="sigmoid",
                 kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal",
                 bias_initializer="zeros", kernel_regularizer=None, recurrent_regularizer=None,
                 bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                 bias_constraint=None, visible_units=None, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.state_size = self.units
        self.output_size = self.units
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        if visible_units is None:
            self._mask = None
        else:
            if not isinstance(visible_units, int):
                raise TypeError("visible_units must be int.")
            if not (0 < visible_units < self.units):
                raise ValueError("visible_units must satisfy 0 < visible_units < units.")
            mask = tf.concat(
                [tf.ones([visible_units], tf.float32),
                 tf.zeros([self.units - visible_units], tf.float32)], axis=0)
            self._mask = tf.reshape(mask, [1, self.units])

    def build(self, input_shape):
        d = int(input_shape[-1]); u = self.units
        self.kernel = self.add_weight(
            "kernel", shape=(d, 3*u),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint, trainable=True)
        self.recurrent_kernel = self.add_weight(
            "recurrent_kernel", shape=(u, 3*u),
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint, trainable=True)
        # reset_after=True bias: (2, 3U): [input_bias; recurrent_bias]
        self.bias = self.add_weight(
            "bias", shape=(2, 3*u),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint, trainable=True)
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        h = states[0]

        # Gate slices derived dynamically so they reflect latest variable values
        W_xz, W_xr, W_xh = tf.split(self.kernel, 3, axis=1)
        W_hz, W_hr, W_hh = tf.split(self.recurrent_kernel, 3, axis=1)

        b_in, b_rec = tf.unstack(self.bias, axis=0)      # shapes (3U,), (3U,)
        b_xz, b_xr, b_xh = tf.split(b_in, 3, axis=0)     # each (U,)
        b_hz, b_hr, b_hh = tf.split(b_rec, 3, axis=0)    # each (U,)

        z = self.recurrent_activation(
            tf.matmul(inputs, W_xz) + tf.matmul(h, W_hz) + b_xz + b_hz)
        r = self.recurrent_activation(
            tf.matmul(inputs, W_xr) + tf.matmul(h, W_hr) + b_xr + b_hr)
        h_tilde = self.activation(
            tf.matmul(inputs, W_xh) + b_xh + r * (tf.matmul(h, W_hh) + b_hh))

        h_new = (1.0 - z) * h_tilde + z * h
        y = h_new if self._mask is None else (h_new * self._mask)
        return y, [h_new]

    def get_config(self):
        base = super().get_config()
        return {
            **base,
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation),
            "recurrent_activation": tf.keras.activations.serialize(self.recurrent_activation),
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": tf.keras.initializers.serialize(self.recurrent_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": tf.keras.regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": tf.keras.constraints.serialize(self.recurrent_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
            "visible_units": None if self._mask is None else int(tf.reduce_sum(self._mask)),
        }


class CustomGRU(tf.keras.layers.RNN):
    """
    RNN wrapper around CustomGRUCell to match tf.keras.layers.GRU API surface you use.
    """
    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        return_sequences=False,
        stateful=False,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        visible_units=None,
        name=None,
        **kwargs,
    ):
        cell = CustomGRUCell(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            visible_units=visible_units,
        )
        super().__init__(
            cell=cell,
            return_sequences=return_sequences,
            stateful=stateful,
            name=name,
            **kwargs,
        )
        # Keep activity regularizer to mirror Dense/RNN layers behavior
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def get_config(self):
        base = super().get_config()
        # pull visible_units from cell (None or int)
        vis = None
        if hasattr(self.cell, "_mask") and self.cell._mask is not None:
            vis = int(tf.reduce_sum(self.cell._mask))
        base.update({
            "units": self.cell.units,
            "activation": tf.keras.activations.serialize(self.cell.activation),
            "recurrent_activation": tf.keras.activations.serialize(self.cell.recurrent_activation),
            "stateful": self.stateful,
            "return_sequences": self.return_sequences,
            "visible_units": vis,
            "activity_regularizer": tf.keras.regularizers.serialize(self.activity_regularizer),
        })
        return base


# -------------------------
# Optional: Keras GRU (for unmasked parity only)
# -------------------------
def make_keras_gru(units, return_sequences, input_dim):
    gru = tf.keras.layers.GRU(
        units,
        return_sequences=return_sequences,
        reset_after=True,
        implementation=2,
        use_bias=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        dtype=tf.float32,
    )
    gru.build(input_shape=(None, None, input_dim))
    return gru

# -------------------------
# Tests
# -------------------------
def main():
    rng = np.random.default_rng(1234)
    tf.random.set_seed(1234)

    B, T, D, U = 3, 6, 5, 7
    x_seq = rng.standard_normal((B, T, D)).astype(np.float32)
    h0 = rng.standard_normal((B, U)).astype(np.float32)
    kernel = rng.standard_normal((D, 3 * U)).astype(np.float32)
    recurrent_kernel = rng.standard_normal((U, 3 * U)).astype(np.float32)
    bias = rng.standard_normal((2, 3 * U)).astype(np.float32)

    # --- Unmasked: NumPy == TF == Keras ---
    for rs in (False, True):
        np_layer = NumpyGRULayer(kernel, recurrent_kernel, bias, units=U, return_sequences=rs)
        tf_layer = TFGRULayer(kernel, recurrent_kernel, bias, units=U, return_sequences=rs)

        y_np = np_layer.forward(x_seq, h_init=h0)
        y_tf = tf_layer(x_seq, h_init=h0).numpy()
        print(f"[Unmasked | return_sequences={rs}] NumPy vs TF:", np.allclose(y_np, y_tf, atol=1e-6, rtol=1e-6))

        keras_gru = make_keras_gru(U, rs, D)
        keras_gru.set_weights([kernel, recurrent_kernel, bias])
        y_k = keras_gru(tf.convert_to_tensor(x_seq), initial_state=tf.convert_to_tensor(h0)).numpy()
        print(f"[Unmasked | return_sequences={rs}] NumPy/TF vs Keras:", np.allclose(y_np, y_k, atol=1e-6, rtol=1e-6))

    # --- Masked: NumPy == TF (recurrence unmasked; output masked) ---
    visible_units = 3  # asserts 0 < 3 < U
    for rs in (False, True):
        np_layer_m = NumpyGRULayer(kernel, recurrent_kernel, bias, units=U, return_sequences=rs, visible_units=visible_units)
        tf_layer_m = TFGRULayer(kernel, recurrent_kernel, bias, units=U, return_sequences=rs, visible_units=visible_units)

        y_np_m = np_layer_m.forward(x_seq, h_init=h0)
        y_tf_m = tf_layer_m(x_seq, h_init=h0).numpy()
        print(f"[Masked (visible_units={visible_units}) | return_sequences={rs}] NumPy vs TF:",
              np.allclose(y_np_m, y_tf_m, atol=1e-6, rtol=1e-6))

    # Sanity: eager vs graph with mask
    tf_layer_graph = TFGRULayer(kernel, recurrent_kernel, bias, units=U, return_sequences=True, visible_units=visible_units)
    @tf.function
    def run_graph(x, s):
        return tf_layer_graph(x, h_init=s)
    y_g = run_graph(x_seq, h0).numpy()
    y_e = tf_layer_graph(x_seq, h0).numpy()
    print("[TF eager vs graph with mask] equal:", np.allclose(y_g, y_e, atol=1e-6, rtol=1e-6))

    # --- CustomGRU (unmasked): parity with TFGRULayer & NumPy ---
    for rs in (False, True):
        cg = CustomGRU(U, return_sequences=rs, visible_units=None)

        # Build once (random weights), then overwrite with reference weights.
        _ = cg(tf.convert_to_tensor(x_seq), initial_state=tf.convert_to_tensor(h0))

        # Assign directly to the cell variables
        cg.cell.kernel.assign(kernel)
        cg.cell.recurrent_kernel.assign(recurrent_kernel)
        cg.cell.bias.assign(bias)

        y_cg = cg(tf.convert_to_tensor(x_seq), initial_state=tf.convert_to_tensor(h0)).numpy()

        # Fresh TF reference for the same rs
        tf_ref = TFGRULayer(kernel, recurrent_kernel, bias, units=U, return_sequences=rs)
        y_ref = tf_ref(x_seq, h_init=h0).numpy()

        print(f"[CustomGRU unmasked | return_sequences={rs}] vs TF ref:",
              np.allclose(y_cg, y_ref, atol=1e-6, rtol=1e-6))
        print(f"[CustomGRU unmasked | return_sequences={rs}] vs NumPy:",
              np.allclose(y_cg, NumpyGRULayer(kernel, recurrent_kernel, bias, units=U, return_sequences=rs)
                          .forward(x_seq, h_init=h0), atol=1e-6, rtol=1e-6))

    # --- CustomGRU (masked): parity with TFGRULayer (recurrence unmasked; output masked) ---
    visible_units = 3  # 0 < 3 < U
    for rs in (False, True):
        cg_m = CustomGRU(U, return_sequences=rs, visible_units=visible_units)
        _ = cg_m(tf.convert_to_tensor(x_seq), initial_state=tf.convert_to_tensor(h0))

        # Assign directly
        cg_m.cell.kernel.assign(kernel)
        cg_m.cell.recurrent_kernel.assign(recurrent_kernel)
        cg_m.cell.bias.assign(bias)

        y_cg_m = cg_m(tf.convert_to_tensor(x_seq), initial_state=tf.convert_to_tensor(h0)).numpy()

        tf_layer_m = TFGRULayer(kernel, recurrent_kernel, bias, units=U,
                                return_sequences=rs, visible_units=visible_units)
        y_tf_m = tf_layer_m(x_seq, h_init=h0).numpy()

        print(f"[CustomGRU masked (visible_units={visible_units}) | return_sequences={rs}] vs TF ref:",
              np.allclose(y_cg_m, y_tf_m, atol=1e-6, rtol=1e-6))

    # --- CustomGRU: eager vs graph (masked) ---
    cg_graph = CustomGRU(U, return_sequences=True, visible_units=visible_units)
    _ = cg_graph(tf.convert_to_tensor(x_seq), initial_state=tf.convert_to_tensor(h0))

    # Assign directly
    cg_graph.cell.kernel.assign(kernel)
    cg_graph.cell.recurrent_kernel.assign(recurrent_kernel)
    cg_graph.cell.bias.assign(bias)

    @tf.function
    def run_graph_cg(x, s):
        return cg_graph(x, initial_state=s)

    y_cg_g = run_graph_cg(tf.convert_to_tensor(x_seq), tf.convert_to_tensor(h0)).numpy()
    y_cg_e = cg_graph(tf.convert_to_tensor(x_seq), initial_state=tf.convert_to_tensor(h0)).numpy()
    print("[CustomGRU eager vs graph with mask] equal:", np.allclose(y_cg_g, y_cg_e, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    main()
