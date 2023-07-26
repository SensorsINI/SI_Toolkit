import os
import numpy as np

from tensorflow import keras

from SI_Toolkit.Functions.TF.Dataset import Dataset

try:
    from SI_Toolkit_ASF.DataSelector import DataSelector
except:
    print('No DataSelector found.')

from SI_Toolkit.Functions.TF.Loss import loss_msr_sequence_customizable


# Uncomment the @profile(precision=4) to get the report on memory usage after the training
# Warning! It may affect performance. I would discourage you to use it for long training tasks
# @profile(precision=4)
def train_network_core(net, net_info, training_dfs_norm, validation_dfs_norm, test_dfs_norm, a):

    # region Prepare data for training
    # DataSelectorInstance = DataSelector(a)
    # DataSelectorInstance.load_data_into_selector(training_dfs_norm)
    # training_dataset = DataSelectorInstance.return_dataset_for_training(shuffle=True, inputs=net_info.inputs, outputs=net_info.outputs)
    training_dataset = Dataset(training_dfs_norm, a, shuffle=True, inputs=net_info.inputs, outputs=net_info.outputs)

    validation_dataset = Dataset(validation_dfs_norm, a, shuffle=False, inputs=net_info.inputs,
                                 outputs=net_info.outputs)

    del training_dfs_norm, validation_dfs_norm, test_dfs_norm

    print('')
    print('Number of samples in training set: {}'.format(training_dataset.number_of_samples))
    print('The mean number of samples from each experiment used for training is {} with variance {}'.format(np.mean(training_dataset.df_lengths), np.std(training_dataset.df_lengths)))
    print('Number of samples in validation set: {}'.format(validation_dataset.number_of_samples))
    print('')

    # endregion

    # region Set basic training features: optimizer, loss, scheduler...

    # net.compile(
    #     loss="mse",
    #     optimizer=keras.optimizers.Adam(a.lr)
    # )

    net.compile(
        loss=loss_msr_sequence_customizable(wash_out_len=a.wash_out_len,
                                            post_wash_out_len=a.post_wash_out_len,
                                            discount_factor=1.0),
        optimizer=keras.optimizers.Adam(a.lr_initial)
    )

    # region Define callbacks to be used in training

    callbacks_for_training = []

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(net_info.path_to_net, 'ckpt' + '.ckpt'),
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=False)

    callbacks_for_training.append(model_checkpoint_callback)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=a.lr_decrease_factor,
        patience=a.lr_patience,
        min_lr=a.lr_minimal,
        verbose=2
    )

    callbacks_for_training.append(reduce_lr)

    post_epoch_training_loss = []
    class AdditionalValidation(keras.callbacks.Callback):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset
        def on_epoch_end(self, epoch, logs=None):
            post_epoch_training_loss.append(self.model.evaluate(self.dataset))

    callbacks_for_training.append(AdditionalValidation(dataset=training_dataset))


    csv_logger = keras.callbacks.CSVLogger(net_info.path_to_net + 'log_training.csv', append=False, separator=';')
    callbacks_for_training.append(csv_logger)

    # endregion

    # region Print information about the network
    net.summary()
    # endregion

    # endregion

    # region Training loop
    loss_eval = net.evaluate(validation_dataset)
    print('Validation loss before starting training is {}'.format(loss_eval))

    history = net.fit(
        training_dataset,
        epochs=a.num_epochs,
        verbose=True,
        shuffle=False,
        validation_data=validation_dataset,
        callbacks=callbacks_for_training,
    )

    try:
        loss = history.history['loss']
    except KeyError:
        loss = []
    try:
        validation_loss = history.history['val_loss']
    except KeyError:
        validation_loss = []

    # endregion

    # region Save final weights as checkpoint
    net.save_weights(net_info.path_to_net + 'ckpt' + '.ckpt')
    # endregion

    return np.array(loss), validation_loss, post_epoch_training_loss
