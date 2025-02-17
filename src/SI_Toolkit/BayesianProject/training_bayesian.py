import tensorflow as tf
import tf_keras
import numpy as np
import tensorflow_probability as tfp
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles

batch_size = 64
num_outputs = 1
h_size = [32]
activation = 'tanh'
epochs = 1000
learning_rate = 0.01

network_inputs = ['angle_sin', 'angle_cos']
network_outputs = ['angle']

tfd = tfp.distributions

# Define negative log-likelihood loss
negloglik = lambda y, rv_y: -rv_y.log_prob(y)

# Load dataset
get_files_from = './Driver/CartPoleSimulation/SI_Toolkit_ASF/Experiments/AGRU_L_m_pole_full/Recordings/Train/'
paths_to_recordings = get_paths_to_datafiles(get_files_from)
dfs = load_data(list_of_paths_to_datafiles=paths_to_recordings, verbose=False)

print(dfs[0].columns)

x = dfs[0].loc[:, network_inputs].to_numpy()
y = dfs[0].loc[:, network_outputs].to_numpy()
# y, x, x_tst = load_dataset()
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

# Define posterior and prior functions
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf_keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf_keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

# Build the model
layers = []
for size in h_size:
    layers.append(tfp.layers.DenseVariational(
        units=size,
        make_posterior_fn=posterior_mean_field,
        make_prior_fn=prior_trainable,
        kl_weight=1/x.shape[0],
    ))
    layers.append(tf_keras.layers.Activation(activation))

layers.append(tfp.layers.DenseVariational(
    units=2 * num_outputs,
    make_posterior_fn=posterior_mean_field,
    make_prior_fn=prior_trainable,
    kl_weight=1/x.shape[0],
))
layers.append(tfp.layers.DistributionLambda(
    lambda t: tfd.Normal(
        loc=t[..., :num_outputs],
        scale=1e-3 + tf.math.softplus(0.01 * t[..., num_outputs:])
    )
))

model = tf_keras.Sequential(layers)

# Compile and train the model
model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)
model.fit(dataset, epochs=epochs, verbose=True)

# Save the trained model
model.save('bayesian_regression_model')
