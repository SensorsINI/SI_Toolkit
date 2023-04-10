from types import SimpleNamespace

import tensorflow as tf
import numpy as np

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

    # if hasattr(args, 'extend_horizon') and args.extend_horizon:
    #     # net = ExtendedHorizonModel(args)
    #     # net = tf.keras.Sequential()
    #     net = ExtendedHorizonBaselineModel(args)
    #
    # else:
    #     net = tf.keras.Sequential()
    net = ExtendedHorizonBaselineModel(args)
    # net = tf.keras.Sequential()

    # Construct network
    # Either dense...
    if net_type == 'Dense':
        net.add(tf.keras.Input(shape=(time_series_length, len(inputs_list))))
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


class ExtendedHorizonModel(tf.keras.Sequential):

    def __init__(self, args):
        super(ExtendedHorizonModel, self).__init__()
        self.args = args

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        # if self.args.extend_horizon:
        #     samples, targets = data
        #     features, shift_labels, exp_len = samples
        #     shift_labels = int(shift_labels.numpy())
        #     exp_len = int(exp_len.numpy())
        # else:
        #     features, targets = data
        #     exp_len = self.args.post_wash_out_len
        #     shift_labels = self.args.shift_labels
        features, targets = data
        exp_len = self.args.post_wash_out_len
        shift_labels = self.args.shift_labels

        with tf.GradientTape() as tape:
            x = features[:, :exp_len, :]
            y = targets[:, :exp_len, :]
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            batches = [(y, y_pred)]
            for i in range(1, shift_labels):
                y = targets[:, i:i+exp_len, :]
                added_inputs_idx = [self.args.inputs.index(inp) for inp in self.args.control_inputs]
                for idx in added_inputs_idx:
                    addeds = tf.expand_dims(features[:, i:i+exp_len, idx], axis=-1)
                    x = tf.concat([y_pred[:, :, :idx], addeds, y_pred[:, :, idx:]], axis=2)
                y_pred = self(x, training=True)  # Forward pass
                batches.append((y, y_pred))
            if self.args.first_loss:
                loss = self.compiled_loss(*batches[0], regularization_losses=self.losses)
            elif self.args.stack_loss:
                ys = tf.concat([pair[0] for pair in batches], axis=0)
                y_preds = tf.concat([pair[1] for pair in batches], axis=0)
                loss = self.compiled_loss(ys, y_preds, regularization_losses=self.losses)
            elif self.args.sum_loss:
                loss = sum([self.compiled_loss(true, pred, regularization_losses=self.losses)
                        for true, pred in batches])
            else:
                loss = self.compiled_loss(*batches[-1], regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        history = {m.name: m.result() for m in self.metrics}
        return history

    def test_step(self, data):
        # Unpack the data
        # if self.args.extend_horizon:
        #     samples, targets = data
        #     features, shift_labels, exp_len = samples
        #     shift_labels = int(shift_labels.numpy())
        #     exp_len = int(exp_len.numpy())
        # else:
        #     features, targets = data
        #     exp_len, shift_labels = 1, 1
        features, targets = data
        exp_len, shift_labels = 1, 1
        x = features[:, :exp_len, :]
        y = targets[:, :exp_len, :]
        y_pred = self(x, training=False)  # Forward pass
        batches = [(y, y_pred)]
        for i in range(1, shift_labels):
            y = targets[:, i:i + exp_len, :]
            added_inputs_idx = [self.args.inputs.index(inp) for inp in self.args.control_inputs]
            for idx in added_inputs_idx:
                addeds = tf.expand_dims(features[:, i:i + exp_len, idx], axis=-1)
                x = tf.concat([y_pred[:, :, :idx], addeds, y_pred[:, :, idx:]], axis=2)
            y_pred = self(x, training=False)  # Forward pass
            batches.append((y, y_pred))

        first_loss = self.compiled_loss(*batches[0])
        last_loss = self.compiled_loss(*batches[-1])
        if self.args.first_loss:
            loss = self.compiled_loss(*batches[0])
        elif self.args.stack_loss:
            ys = tf.concat([pair[0] for pair in batches], axis=0)
            y_preds = tf.concat([pair[1] for pair in batches], axis=0)
            loss = self.compiled_loss(ys, y_preds)
        elif self.args.sum_loss:
            loss = sum([self.compiled_loss(true, pred) for true, pred in batches])
        else:
            loss = self.compiled_loss(*batches[-1])

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        # history = {m.name: m.result() for m in self.metrics}
        history = {}
        history['first_loss'] = first_loss
        history['last_loss'] = last_loss
        history['loss'] = loss
        return history


class ExtendedHorizonBaselineModel(tf.keras.Sequential):

    def __init__(self, args):
        super(ExtendedHorizonBaselineModel, self).__init__()
        self.args = args

    def train_step(self, data):
        features, targets = data

        with tf.GradientTape() as tape:
            y_pred = self(features, training=True)
            loss = self.compiled_loss(targets, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(targets, y_pred)
        history = {m.name: m.result() for m in self.metrics}
        return history

    def test_step(self, data):
        features, targets = data
        y_pred = self(features, training=False)  # Forward pass
        loss = self.compiled_loss(targets, y_pred)
        history = {'first_loss': loss, 'last_loss': loss, 'loss': loss}
        return history


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
