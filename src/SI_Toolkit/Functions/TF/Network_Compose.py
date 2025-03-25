def create_bayesian_network(net_info,
                            net_type,
                            h_size,
                            time_series_length,
                            batch_size=None,
                            activation='tanh',
                            **kwargs):

    import tensorflow as tf
    import tensorflow_probability as tfp
    import numpy as np

    tfd = tfp.distributions
    inputs_len = net_info.inputs_len

    if net_type != 'Dense':
        raise NotImplementedError('Only dense Bayesian networks are supported')

    # Posterior distribution for the weights
    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])

    # Trainable prior distribution for the weights
    def prior_trainable(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=1),
                reinterpreted_batch_ndims=1)),
        ])

    # Determine the number of outputs
    num_outputs = len(net_info.outputs)

    # Build the input layer
    input_layer = tf.keras.layers.Input(shape=(time_series_length, inputs_len), batch_size=batch_size)

    # Reshape the input to merge time series and batch dimensions
    reshaped_input = tf.keras.layers.Reshape((-1, inputs_len))(input_layer)

    # Build hidden layers
    x = reshaped_input
    for size in h_size:
        x = tfp.layers.DenseVariational(
            units=size,
            make_posterior_fn=posterior_mean_field,
            make_prior_fn=prior_trainable,
            kl_weight=1.0e-3
        )(x)
        x = tf.keras.layers.Activation(activation)(x)

    # Build output layer
    output_layer = tfp.layers.DenseVariational(
        units=num_outputs,
        make_posterior_fn=posterior_mean_field,
        make_prior_fn=prior_trainable,
        kl_weight=1.0e-3
    )(x)
    output_layer = tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t, scale=1)
    )(output_layer)

    # Reshape the output back to [batch_size, time_series_length, num_outputs]
    reshaped_output = tf.keras.layers.Reshape((time_series_length, num_outputs))(output_layer)

    # Construct the model
    model = tf.keras.Model(inputs=input_layer, outputs=reshaped_output)

    return model, net_info
