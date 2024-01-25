import sys

import importlib.util

import tensorflow as tf

from SI_Toolkit.Functions.TF.Compile import CompileTF

try:
    import qkeras
except ModuleNotFoundError:
    print('QKeras not found. Quantization-aware training will not be available.')

def load_pretrained_net_weights(net, ckpt_path, verbose=True):
    """
    A function loading parameters (weights and biases) from a previous training to a net RNN instance
    :param net: An instance of RNN
    :param ckpt_path: path to .ckpt file storing weights and biases
    :return: No return. Modifies net in place.
    """
    if verbose:
        print("Loading Model: ", ckpt_path)
        print('')

    net.load_weights(ckpt_path).expect_partial()


def compose_net_from_module(net_info,
                            time_series_length,
                            batch_size,
                            stateful=False,
                            **kwargs,
                            ):
    net_type, module_name, class_name = net_info.net_name.split('-')
    path = './SI_Toolkit_ASF/Modules/'

    spec = importlib.util.spec_from_file_location(f"{module_name}.{class_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    net = getattr(module, class_name)(time_series_length, batch_size, net_info)
    net.build((batch_size, time_series_length, len(net_info.inputs)))

    print(f'Loaded the model {class_name} from {path}.')

    net_info.net_type = net_type

    return net, net_info


def compose_net_from_net_name(net_info,
                              time_series_length,
                              batch_size=None,
                              stateful=False,
                              remove_redundant_dimensions=False,
                              **kwargs,
                              ):

    net_name = net_info.net_name
    inputs_list = net_info.inputs
    outputs_list = net_info.outputs

    # Get the information about network architecture from the network name
    # Split the names into "LSTM/GRU", "128H1", "64H2" etc.
    names = net_name.split('-')
    layers = ['H1', 'H2', 'H3', 'H4', 'H5']
    h_size = []  # Hidden layers sizes
    for name in names:
        for index, layer in enumerate(layers):
            if layer in name:
                # assign the variable with name obtained from list layers.
                h_size.append(int(name[:-2]))

    if not h_size:
        raise ValueError('You have to provide the size of at least one hidden layer in rnn name')

    h_number = len(h_size)

    if 'GRU' in names:
        net_type = 'GRU'
    elif 'LSTM' in names:
        net_type = 'LSTM'
    elif 'Dense' in names:
        net_type = 'Dense'
    else:
        net_type = 'RNN-Basic'

    if hasattr(net_info, 'quantization') and net_info.quantization['ACTIVATED']:
        if 'qkeras' not in sys.modules:
            raise ModuleNotFoundError('QKeras not found. Quantization-aware training will not be available. Change config_training or install the module')
        if 'GRU' in names:
            layer_type = qkeras.qrecurrent.QGRU
        elif 'LSTM' in names:
            layer_type = qkeras.qrecurrent.QLSTM
        elif 'Dense' in names:
            layer_type = qkeras.qlayers.QDense
        else:
            layer_type = qkeras.qrecurrent.QSimpleRNN
    else:
        if 'GRU' in names:
            layer_type = tf.keras.layers.GRU
        elif 'LSTM' in names:
            layer_type = tf.keras.layers.LSTM
        elif 'Dense' in names:
            layer_type = tf.keras.layers.Dense
        else:
            layer_type = tf.keras.layers.SimpleRNN

    activation = 'tanh'
    activation_last_layer = 'linear'
    quantization_last_layer_args = {}
    quantization_args = {}

    if hasattr(net_info, 'quantization') and net_info.quantization['ACTIVATED']:
        activation = qkeras.quantizers.quantized_tanh(**net_info.quantization['ACTIVATION'])
        quantization_last_layer_args['kernel_quantizer'] = qkeras.quantizers.quantized_bits(**net_info.quantization['KERNEL'])
        quantization_last_layer_args['bias_quantizer'] = qkeras.quantizers.quantized_bits(**net_info.quantization['BIAS'])
        quantization_args = quantization_last_layer_args.copy()
        if net_type in ['GRU', 'LSTM', 'RNN-Basic']:
            quantization_args['recurrent_quantizer'] = qkeras.quantizers.quantized_bits(**net_info.quantization['RECURRENT'])

    if hasattr(net_info, 'regularization') and net_info.regularization['ACTIVATED']:
        regularization_kernel = net_info.regularization['KERNEL']
        regularization_bias = net_info.regularization['BIAS']
        regularization_activity = net_info.regularization['ACTIVITY']
    else:
        regularization_kernel = {'l1': 0.0, 'l2': 0.0}
        regularization_bias = {'l1': 0.0, 'l2': 0.0}
        regularization_activity = {'l1': 0.0, 'l2': 0.0}

    net = tf.keras.Sequential()

    # Construct network
    # Either dense...
    if net_type == 'Dense':

        if remove_redundant_dimensions and time_series_length==1:
            shape_input = (len(inputs_list),)
            batch_size = None
        else:
            shape_input = (time_series_length, len(inputs_list))

        net.add(tf.keras.Input(batch_size=batch_size, shape=shape_input))
        for i in range(h_number):
            net.add(layer_type(
                units=h_size[i], activation=activation, batch_size=batch_size,  name='layers_{}'.format(i),
                kernel_regularizer=tf.keras.regularizers.l1_l2(**regularization_kernel),
                bias_regularizer=tf.keras.regularizers.l1_l2(**regularization_bias),
                activity_regularizer=tf.keras.regularizers.l1_l2(**regularization_activity),
                **quantization_args,
            ))
    else:

        if remove_redundant_dimensions and batch_size==1:
            shape_input = (time_series_length, len(inputs_list))
        else:
            shape_input = (batch_size, time_series_length, len(inputs_list))

        # Or RNN...
        net.add(layer_type(
            units=h_size[0],
            activation=activation,
            batch_input_shape=shape_input,
            return_sequences=True,
            stateful=stateful,
            kernel_regularizer=tf.keras.regularizers.l1_l2(**regularization_kernel),
            bias_regularizer=tf.keras.regularizers.l1_l2(**regularization_bias),
            activity_regularizer=tf.keras.regularizers.l1_l2(**regularization_activity),
            **quantization_args,
        ))
        # Define following layers
        for i in range(1, len(h_size)):
            net.add(layer_type(
                units=h_size[i],
                activation=activation,
                return_sequences=True,
                stateful=stateful,
                kernel_regularizer=tf.keras.regularizers.l1_l2(**regularization_kernel),
                bias_regularizer=tf.keras.regularizers.l1_l2(**regularization_bias),
                activity_regularizer=tf.keras.regularizers.l1_l2(**regularization_activity),
                **quantization_args,
            ))

    if hasattr(net_info, 'regularization') and net_info.regularization['ACTIVATED']:
        net.add(qkeras.qlayers.QDense(units=len(outputs_list), name='layers.{}'.format(h_number),
                                      activation=activation_last_layer,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(**regularization_kernel),
                                      bias_regularizer=tf.keras.regularizers.l1_l2(**regularization_bias),
                                      **quantization_last_layer_args,
                                      ))
    else:
        net.add(tf.keras.layers.Dense(units=len(outputs_list), name='layers.{}'.format(h_number), activation=activation_last_layer,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(**regularization_kernel),
                                      bias_regularizer=tf.keras.regularizers.l1_l2(**regularization_bias),
                                      ))

    print('Constructed a neural network of type {}, with {} hidden layers with sizes {} respectively.'
          .format(net_type, len(h_size), ', '.join(map(str, h_size))))

    net_info.net_type = net_type

    return net, net_info


def _copy_internal_states_to_ref(net, memory_states_ref):
    for layer, layer_ref in zip(net.layers, memory_states_ref):
        if (('gru' in layer.name) or
                ('lstm' in layer.name) or
                ('rnn' in layer.name)):

            for single_state, single_state_ref in zip(layer.states, layer_ref.states):
                single_state_ref.assign(tf.convert_to_tensor(single_state))
        else:
            pass


def _copy_internal_states_from_ref(net, memory_states_ref):
    for layer, layer_ref in zip(net.layers, memory_states_ref):
        if (('gru' in layer.name) or
                ('lstm' in layer.name) or
                ('rnn' in layer.name)):

            for single_state, single_state_ref in zip(layer.states, layer_ref.states):
                single_state.assign(tf.convert_to_tensor(single_state_ref))
        else:
            pass


copy_internal_states_to_ref = CompileTF(_copy_internal_states_to_ref)
copy_internal_states_from_ref = CompileTF(_copy_internal_states_from_ref)

def plot_weights_distribution(model, show=True, path_to_save=None):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    for i in range(len(model.layers)):
        for j in range(len(model.layers[i].weights)):
            w = model.layers[i].weights[j].numpy()
            h, b = np.histogram(w, bins=100)
            plt.figure(figsize=(7, 7))
            plt.bar(b[:-1], h, width=b[1] - b[0])
            plt.yscale('log')
            plt.xlabel("params sizes")
            plt.ylabel("number of params")
            plt.title(f'{model.layers[i].weights[j].name}\n# of params = {np.size(w)}; % of zeros = {np.sum(w == 0) / np.size(w)}')

            mean = np.mean(w)
            min_value = np.min(w)
            max_value = np.max(w)
            plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)
            min_ylim, max_ylim = plt.ylim()
            min_xlim, max_xlim = plt.xlim()
            x_values_range = abs(max_xlim - min_xlim)
            log_max = np.log10(max_ylim)  # Adjust these based on your y-axis range
            plt.text(mean + 0.1*x_values_range, 10**(0.9 * log_max), f'Mean: {mean:.3f}')
            plt.text(mean + 0.1*x_values_range, 10**(0.85 * log_max), f"Range: {min_value:.2f} - {max_value:.2f}")

            if show:
                plt.show()
            if path_to_save is not None:
                name_to_save = model.layers[i].weights[j].name.replace('/', '_').replace(':', '_')
                plt.savefig(os.path.join(path_to_save, f'{name_to_save}.png'))
