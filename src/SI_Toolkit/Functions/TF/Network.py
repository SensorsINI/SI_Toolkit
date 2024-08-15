import sys

import importlib.util

import tensorflow as tf
import numpy as np

from SI_Toolkit.Functions.TF.Compile import CompileTF

try:
    import qkeras
except (ModuleNotFoundError, ImportError, AttributeError) as error:
    print('QKeras not found or not working. \n'
          'Quantization-aware training will not be available.'
          f'Got an error: \n'
          f'{error}. \n'
          )

try:
    from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
    from tensorflow_model_optimization.sparsity.keras import strip_pruning
except ModuleNotFoundError:
    print('tensorflow_model_optimization not found. Pruning will not be available.')

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
    path = './SI_Toolkit_ASF/ToolkitCustomization/Modules/'

    spec = importlib.util.spec_from_file_location(class_name, f"{path}/{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    net = getattr(module, class_name)(time_series_length, batch_size, net_info)

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
            layer_type = qkeras.QGRU
        elif 'LSTM' in names:
            layer_type = qkeras.QLSTM
        elif 'Dense' in names:
            layer_type = qkeras.QDense
        else:
            layer_type = qkeras.QSimpleRNN
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
        activation = qkeras.quantizers.quantized_tanh(**net_info.quantization['ACTIVATION'], use_real_tanh=True, symmetric=True)
        quantization_last_layer_args['activation'] = qkeras.quantizers.quantized_bits(**net_info.quantization['KERNEL'], alpha=1)
        quantization_last_layer_args['kernel_quantizer'] = qkeras.quantizers.quantized_bits(**net_info.quantization['KERNEL'], alpha=1)
        quantization_last_layer_args['bias_quantizer'] = qkeras.quantizers.quantized_bits(**net_info.quantization['BIAS'], alpha=1)
        quantization_args = quantization_last_layer_args.copy()
        if net_type in ['GRU', 'LSTM', 'RNN-Basic']:
            quantization_args['recurrent_quantizer'] = qkeras.quantizers.quantized_bits(**net_info.quantization['RECURRENT'], alpha=1)

    if hasattr(net_info, 'regularization') and net_info.regularization['ACTIVATED']:
        regularization_kernel = tf.keras.regularizers.l1_l2(**net_info.regularization['KERNEL'])
        regularization_bias = tf.keras.regularizers.l1_l2(**net_info.regularization['BIAS'])
        regularization_activity = tf.keras.regularizers.l1_l2(**net_info.regularization['ACTIVITY'])
    else:
        regularization_kernel = None
        regularization_bias = None
        regularization_activity = None

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
            if hasattr(net_info, 'quantization') and net_info.quantization['ACTIVATED']:
                net.add(layer_type(
                    units=h_size[i], batch_size=batch_size, name='layers_{}'.format(i),
                    kernel_regularizer=regularization_kernel,
                    bias_regularizer=regularization_bias,
                    activity_regularizer=regularization_activity,
                    **quantization_args,
                ))
                net.add(qkeras.QActivation(activation=activation))
            else:
                net.add(layer_type(
                    units=h_size[i], batch_size=batch_size, name='layers_{}'.format(i),
                    kernel_regularizer=regularization_kernel,
                    bias_regularizer=regularization_bias,
                    activity_regularizer=regularization_activity,
                    **quantization_args,
                ))
                net.add(tf.keras.layers.Activation(tf.keras.activations.tanh))
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
            kernel_regularizer=regularization_kernel,
            bias_regularizer=regularization_bias,
            activity_regularizer=regularization_activity,
            **quantization_args,
        ))
        # Define following layers
        for i in range(1, len(h_size)):
            net.add(layer_type(
                units=h_size[i],
                activation=activation,
                return_sequences=True,
                stateful=stateful,
                kernel_regularizer=regularization_kernel,
                bias_regularizer=regularization_bias,
                activity_regularizer=regularization_activity,
                **quantization_args,
            ))

    if hasattr(net_info, 'quantization') and net_info.quantization['ACTIVATED']:
        net.add(qkeras.QDense(units=len(outputs_list), name='layers_{}'.format(h_number),
                                      kernel_regularizer=regularization_kernel,
                                      bias_regularizer=regularization_bias,
                                      **quantization_last_layer_args,
                                      ))
    else:
        net.add(tf.keras.layers.Dense(units=len(outputs_list), name='layers_{}'.format(h_number), activation=activation_last_layer,
                                      kernel_regularizer=regularization_kernel,
                                      bias_regularizer=regularization_bias,
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

def plot_params_histograms(params, title, show=True, path_to_save=None):
    import numpy as np
    import matplotlib.pyplot as plt
    h, b = np.histogram(params, bins=100)
    plt.figure(figsize=(7, 7))
    plt.bar(b[:-1], h, width=b[1] - b[0])
    plt.yscale('log')
    plt.xlabel("params sizes")
    plt.ylabel("number of params")
    plt.title(title)

    number_params = np.size(params)
    mean = np.mean(params)
    min_value = np.min(params)
    max_value = np.max(params)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    x_values_range = abs(max_xlim - min_xlim)
    log_max = np.log10(max_ylim)
    log_min = np.log10(min_ylim)

    plt.text(mean + 0.1 * x_values_range, 10 ** (log_min + 0.95 * (log_max-log_min)), f'Mean: {mean:.3f}')
    plt.text(mean + 0.1 * x_values_range, 10 ** (log_min + 0.85 * (log_max-log_min)), f"Range: {min_value:.2f} - {max_value:.2f}")
    max_int = int(np.floor(np.max([abs(min_value), abs(max_value)])))
    num_bits = num_bits_needed_for_integer_part(max_int)
    plt.text(mean + 0.1 * x_values_range, 10 ** (log_min + 0.80 * (log_max-log_min)), f"Bits needed \nfor biggest integer ({max_int}): {num_bits}")
    plt.text(mean + 0.1 * x_values_range, 10 ** (log_min + 0.65 * (log_max-log_min)), f'Number params: {number_params}')
    unique_params = np.unique(params)
    diff = np.diff(np.sort(unique_params))
    if len(diff) == 0:
        minimum_difference = 0
    else:
        minimum_difference = np.min(diff)

    if minimum_difference == 0:
        fractional_bits = np.inf
    else:
        fractional_bits = -np.log2(minimum_difference)

    plt.text(mean + 0.1 * x_values_range, 10 ** (log_min + 0.70 * (log_max-log_min)),
             f"Unique params & -log2(min difference): \n(indicates quantization) \n{len(unique_params)}; {fractional_bits}")



    if show:
        plt.show()
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.close()

def plot_weights_distribution(model, show=True, path_to_save=None):
    import os
    import numpy as np
    for i in range(len(model.layers)):
        for j in range(len(model.layers[i].weights)):
            w = model.layers[i].weights[j].numpy()
            name_to_save = model.layers[i].weights[j].name.replace('/', '_').replace(':', '_')
            if path_to_save is not None:
                full_path_to_save = os.path.join(path_to_save, f'{name_to_save}.png')
            else:
                full_path_to_save = None
            title = f'{model.layers[i].weights[j].name}\n# of params = {np.size(w)}; % of zeros = {np.sum(w == 0) / np.size(w)}'
            plot_params_histograms(w, title, show=show, path_to_save=full_path_to_save)


def get_activation_statistics(model, datasets, path_to_save=None):
    import numpy as np
    import os
    from tqdm import tqdm
    print()
    print('Calculating activations statistics...')
    print('For each - except for last - layer the calculation is done twice: with and without the activation function')
    # Creating a list of intermediate models for each layer's output
    intermediate_models = [tf.keras.Model(inputs=model.input, outputs=layer.output) for layer in model.layers]
    for i in range(len(intermediate_models)):
        layer_model = intermediate_models[i]
        layer_name = layer_model.layers[-1].name
        activations = []

        # Passing the dataset through the intermediate model
        if isinstance(datasets, list):
            for k in range(len(datasets)):
                for batch in tqdm(datasets[k], desc=f'Progress layer or activation {i+1} (out of {len(intermediate_models)}) '
                                                    f'dataset {k+1} (out of {len(datasets)}) - activations statistics', leave=False, position=0):
                    features = batch[0]
                    batch_activations = layer_model(features, training=False)
                    activations.append(batch_activations)
        else:
            for batch in tqdm(datasets, desc=f'Progress layer or activation {i+1} (out of {len(intermediate_models)}) - activations statistics', leave=False, position=0):
                features = batch[0]
                batch_activations = layer_model(features, training=False)
                activations.append(batch_activations)

        # Concatenating activations across all batches
        activations = np.concatenate(activations, axis=0)

        if path_to_save is not None:
            full_path_to_save = os.path.join(path_to_save, f'{layer_name}_activations.png')
        else:
            full_path_to_save = None

        plot_params_histograms(activations, title=layer_name, show=False, path_to_save=full_path_to_save)


def num_bits_needed_for_integer_part(n):
    import math
    if n < 0:
        raise ValueError("Number must be non-negative")
    if n == 0:
        return 1  # At least 1 bit is needed to represent 0
    else:
        return math.floor(math.log2(n)) + 1

def get_pruning_params(net_info, number_of_batches):
    # region Defining pruning
    if hasattr(net_info, 'pruning_activated') and net_info.pruning_activated:
        if 'tensorflow_model_optimization' not in sys.modules:
            raise ModuleNotFoundError('tensorflow_model_optimization not found. Pruning will not be available. Change config_training or install the module')
        if net_info.pruning_schedule == 'CONSTANT_SPARSITY':
            pruning_schedule_params = net_info.pruning_schedules[net_info.pruning_schedule]
            selected_pruning_schedule = pruning_schedule.ConstantSparsity(
                target_sparsity=pruning_schedule_params['target_sparsity'],
                begin_step=int(pruning_schedule_params['begin_step_in_epochs']*number_of_batches),
                end_step=int(pruning_schedule_params['end_step_in_training_fraction']*net_info.num_epochs*number_of_batches),
                frequency=int(np.maximum(1, number_of_batches/pruning_schedule_params['frequency_per_epoch'])))
        elif net_info.pruning_schedule == 'POLYNOMIAL_DECAY':
            pruning_schedule_params = net_info.pruning_schedules[net_info.pruning_schedule]
            selected_pruning_schedule = pruning_schedule.PolynomialDecay(
                initial_sparsity=pruning_schedule_params['initial_sparsity'],
                final_sparsity=pruning_schedule_params['final_sparsity'],
                begin_step=int(pruning_schedule_params['begin_step_in_epochs']*number_of_batches),
                end_step=int(pruning_schedule_params['end_step_in_training_fraction']*net_info.num_epochs*number_of_batches),
                power=pruning_schedule_params['power'],
                frequency=int(np.maximum(1, number_of_batches/pruning_schedule_params['frequency_per_epoch'])))
        else:
            raise NotImplementedError('Pruning schedule {} is not implemented yet.'.format(net_info.pruning_schedule))

        pruning_params = {"pruning_schedule": selected_pruning_schedule}
        return pruning_params

def make_prunable(net, net_info, number_of_batches):
    pruning_params = get_pruning_params(net_info, number_of_batches)

    # Rebuild the model with pruned layers
    pruned_layers = []
    for i, layer in enumerate(net.layers):
        # Adjust pruning parameters for the last layer if needed
        if i == len(net.layers) - 1:
            net_info.pruning_schedules['CONSTANT_SPARSITY']['target_sparsity'] = \
            net_info.pruning_schedules['CONSTANT_SPARSITY']['target_sparsity_last_layer']
            net_info.pruning_schedules['POLYNOMIAL_DECAY']['final_sparsity'] = \
            net_info.pruning_schedules['POLYNOMIAL_DECAY']['final_sparsity_last_layer']
            pruning_params = get_pruning_params(net_info, number_of_batches)
        if not isinstance(layer, tf.keras.layers.InputLayer):  # Skip input layer
            # Wrap layer with pruning
            pruned_layer = prune.prune_low_magnitude(layer, **pruning_params)
        else:
            pruned_layer = layer  # Keep input layer unchanged
        pruned_layers.append(pruned_layer)

    # Reconstruct the model with pruned layers
    prunable_model = tf.keras.Sequential(pruned_layers)

    # Prune whole network
    # prunable_model = prune.prune_low_magnitude(net, **pruning_params)
    return prunable_model