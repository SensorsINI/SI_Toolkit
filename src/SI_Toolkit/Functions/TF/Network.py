import importlib.util

import tensorflow as tf

from SI_Toolkit.Functions.TF.Compile import CompileTF

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
        layer_type = tf.keras.layers.GRU
    elif 'LSTM' in names:
        net_type = 'LSTM'
        layer_type = tf.keras.layers.LSTM
    elif 'Dense' in names:
        net_type = 'Dense'
        layer_type = tf.keras.layers.Dense
    else:
        net_type = 'RNN-Basic'
        layer_type = tf.keras.layers.SimpleRNN

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
                units=h_size[i], activation='tanh', batch_size=batch_size,  name='layers.{}'.format(i)
            ))
    else:

        if remove_redundant_dimensions and batch_size==1:
            shape_input = (time_series_length, len(inputs_list))
        else:
            shape_input = (batch_size, time_series_length, len(inputs_list))

        # Or RNN...
        net.add(layer_type(
            units=h_size[0],
            batch_input_shape=shape_input,
            return_sequences=True,
            stateful=stateful
        ))
        # Define following layers
        for i in range(1, len(h_size)):
            net.add(layer_type(
                units=h_size[i],
                return_sequences=True,
                stateful=stateful
            ))

    net.add(tf.keras.layers.Dense(units=len(outputs_list), name='layers.{}'.format(h_number)))

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
