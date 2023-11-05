from types import SimpleNamespace

import tensorflow as tf

from SI_Toolkit.Functions.TF.Compile import CompileTF

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
                              args,
                              time_series_length,
                              batch_size=None,
                              stateful=False,
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
    else:
        net_type = 'RNN-Basic'
        layer_type = tf.keras.layers.SimpleRNN

    if args.extend_horizon:
        net = ExtendedHorizonBaselineModel(args)
    else:
        net = tf.keras.Sequential()
    # Construct network
    # Either dense...
    if net_type == 'Dense':
        net.add(tf.keras.Input(batch_size=batch_size, shape=(time_series_length, len(inputs_list))))
        for i in range(h_number):
            net.add(layer_type(
                units=h_size[i], activation='tanh', batch_size=batch_size
            ))
    else:
        # Or RNN...
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

    # net.add(tf.keras.layers.Dense(units=len(outputs_list), activation='tanh'))
    net.add(tf.keras.layers.Dense(units=len(outputs_list)))

    print('Constructed a neural network of type {}, with {} hidden layers with sizes {} respectively.'
          .format(net_type, len(h_size), ', '.join(map(str, h_size))))

    # Compose net_info
    net_info = SimpleNamespace()
    net_info.net_name = net_name
    net_info.inputs = inputs_list
    net_info.outputs = outputs_list
    net_info.net_type = net_type

    return net, net_info


class ExtendedHorizonBaselineModel(tf.keras.Sequential):

    def __init__(self, args):
        super(ExtendedHorizonBaselineModel, self).__init__()
        self.args = args
        self.all_losses = []

    def train_step(self, data):
        features, targets = data

        with tf.GradientTape() as tape:
            x = features[:, :1, :]
            y = targets[:, :1, :]
            y_pred = self(x, training=True)
            batches = [(y, y_pred)]
            shift_labels = features.shape[1]
            for i in range(1, shift_labels):
                y = targets[:, i:i+1, :]
                added_inputs_idx = [self.args.inputs.index(inp) for inp in
                                    self.args.control_inputs]
                for idx in added_inputs_idx:
                    addeds = tf.expand_dims(features[:, i:i+1, idx], axis=-1)
                    x = tf.concat([y_pred[:, :, :idx], addeds, y_pred[:, :, idx:]], axis=2)
                y_pred = self(x, training=True)
                batches.append((y, y_pred))
            ys = tf.concat([pair[0] for pair in batches], axis=0)
            y_preds = tf.concat([pair[1] for pair in batches], axis=0)
            loss = self.compiled_loss(ys, y_preds, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(ys, y_preds)
        # history = {m.name: m.result() / shift_labels for m in self.metrics}
        history = {m.name: m.result() for m in self.metrics}

        return history

    def test_step(self, data):
        features, targets = data

        x = features[:, :1, :]
        y = targets[:, :1, :]
        y_pred = self(x, training=False)
        batches = [(y, y_pred)]
        shift_labels = features.shape[1]
        for i in range(1, shift_labels):
            y = targets[:, i:i + 1, :]
            added_inputs_idx = [self.args.inputs.index(inp) for inp in self.args.control_inputs]
            for idx in added_inputs_idx:
                addeds = tf.expand_dims(features[:, i:i + 1, idx], axis=-1)
                x = tf.concat([y_pred[:, :, :idx], addeds, y_pred[:, :, idx:]], axis=2)
            y_pred = self(x, training=False)
            batches.append((y, y_pred))
        ys = tf.concat([pair[0] for pair in batches], axis=0)
        y_preds = tf.concat([pair[1] for pair in batches], axis=0)
        self.compiled_loss(ys, y_preds, regularization_losses=self.losses)
        self.compiled_metrics.update_state(ys, y_preds)
        history = {m.name: m.result() for m in self.metrics}
        losses = [tf.reduce_mean(self.loss(*x)) for x in batches]
        for idx, loss in enumerate(losses):
            self.all_losses[idx].append(loss)
        history.update({f'loss_{n+1}': sum(x)/len(x) for n, x in enumerate(self.all_losses)})
        history['shift_labels_history'] = shift_labels

        return history

    def set_initial_loss_tracker(self, shift_labels):
        self.all_losses = [[] for x in range(shift_labels)]


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
