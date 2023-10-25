from types import SimpleNamespace

import tensorflow as tf

from SI_Toolkit.Functions.TF.Compile import CompileTF

from differentiable_plasticity import Plastic_Dense_Layer # introduce hebbian term for test

def load_pretrained_net_weights(net, ckpt_path):
    """
    A function loading parameters (weights and biases) from a previous training to a net RNN instance
    :param net: An instance of RNN
    :param ckpt_path: path to .ckpt file storing weights and biases
    :return: No return. Modifies net in place.
    """
    print("Loading Model: ", ckpt_path)
    print('')

    net.load_weights(ckpt_path).expect_partial()


def compose_net_from_net_name(net_name,
                              inputs_list,
                              outputs_list,
                              time_series_length,
                              batch_size=None,
                              stateful=False,
                              F=None, # fisher information matrix
                              bases=None, # previous optimal parameters
                              flag_multi_head=False,
                              **kwargs,
                              ):

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
    elif 'PlasticDense' in names:
        net_type = 'PlasticDense'
        layer_type = Plastic_Dense_Layer
    else:
        net_type = 'RNN-Basic'
        layer_type = tf.keras.layers.SimpleRNN

    net = tf.keras.Sequential()

    # Construct network
    # Either dense...
    if net_type == 'Dense': # consider to test 'relu'
        net.add(tf.keras.Input(shape=(time_series_length, len(inputs_list))))
        for i in range(h_number):
            net.add(layer_type(
                units=h_size[i], activation='tanh', batch_size=batch_size
            ))
    elif net_type == 'PlasticDense':
        net.add(tf.keras.Input(shape=(time_series_length, len(inputs_list))))
        for i in range(h_number):
            net.add(tf.keras.layers.Dense(
                units=h_size[i], activation='tanh', batch_size=batch_size
            ))
        # net.add(layer_type(input_dim=len(inputs_list) , unit=h_size[0]))
        # net.add(tf.keras.layers.Dense(units=h_size[0], activation='tanh', batch_size=batch_size))
        # for i in range(1,h_number):
        #     net.add(layer_type(
        #         input_dim=h_size[i-1] , unit=h_size[i]
        #     ))
    # Or RNN...
    else:
        if F is None:
            net.add(layer_type(
                units=h_size[0],
                batch_input_shape=(batch_size, time_series_length, len(inputs_list)),
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
        else:
            from train_ewc import EWCRegularizer
            net.add(layer_type(
                units=h_size[0],
                batch_input_shape=(batch_size, time_series_length, len(inputs_list)),
                return_sequences=True,
                stateful=stateful,
                kernel_regularizer=EWCRegularizer(F[0],bases[0]),
                recurrent_regularizer=EWCRegularizer(F[1],bases[1]),
                bias_regularizer=EWCRegularizer(F[2],bases[2])
            ))
            # Define following layers
            for i in range(1, len(h_size)):
                net.add(layer_type(
                    units=h_size[i],
                    return_sequences=True,
                    stateful=stateful,
                    kernel_regularizer=EWCRegularizer(F[3],bases[3]),
                    recurrent_regularizer=EWCRegularizer(F[4],bases[4]),
                    bias_regularizer=EWCRegularizer(F[5],bases[5])
                ))

    # net.add(tf.keras.layers.Dense(units=len(outputs_list), activation='tanh'))
    if net_type == 'PlasticDense':
        net.add(layer_type(input_dim=h_size[-1], unit=len(outputs_list), activation=False))
    else:
        # net.add(tf.keras.layers.Dense(units=len(outputs_list)))
        if F is None:
            net.add(tf.keras.layers.Dense(units=len(outputs_list)))
        else:
            # if using regularizer in output layer depends on whether using multi-head or not
            if flag_multi_head:
                print('!!!!!!')
                print('multi head')
                net.add(tf.keras.layers.Dense(units=len(outputs_list)))
            else:
                net.add(tf.keras.layers.Dense(units=len(outputs_list),
                                            kernel_regularizer=EWCRegularizer(F[6],bases[6]),bias_regularizer=EWCRegularizer(F[7],bases[7])))

    print('Constructed a neural network of type {}, with {} hidden layers with sizes {} respectively.'
          .format(net_type, len(h_size), ', '.join(map(str, h_size))))

    # Compose net_info
    net_info = SimpleNamespace()
    net_info.net_name = net_name
    net_info.inputs = inputs_list
    net_info.outputs = outputs_list
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
