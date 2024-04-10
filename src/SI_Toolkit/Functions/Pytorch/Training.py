import numpy as np
import os

import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader
from torch.optim import lr_scheduler
from torch.utils import data

import time
from tqdm import tqdm

from SI_Toolkit.Functions.Pytorch.Dataset import Dataset
from SI_Toolkit.Functions.Pytorch.Network import print_parameter_count, get_device
from SI_Toolkit.Functions.Pytorch.Loss import loss_msr_sequence_customizable

# region
print('')
device = get_device()
print('')
# endregion

# Uncomment the @profile(precision=4) to get the report on memory usage after the training
# Warning! It may affect performance. I would discourage you to use it for long training tasks
# @profile(precision=4)
def train_network_core(net, net_info, training_dfs_norm, validation_dfs_norm, test_dfs_norm, a):

    # region Prepare data for training
    training_dataset = Dataset(training_dfs_norm, a, inputs=net_info.inputs, outputs=net_info.outputs)

    validation_dataset = Dataset(validation_dfs_norm, a, inputs=net_info.inputs,
                                 outputs=net_info.outputs)

    test_dataset = Dataset(test_dfs_norm, a, inputs=net_info.inputs, outputs=net_info.outputs)

    del training_dfs_norm, validation_dfs_norm, test_dfs_norm

    # Create PyTorch dataloaders for train and dev set
    training_generator = data.DataLoader(dataset=training_dataset, batch_size=a.batch_size, shuffle=True, drop_last=True)
    validation_generator = data.DataLoader(dataset=validation_dataset, batch_size=a.batch_size, shuffle=False, drop_last=True)

    print('')
    print('Number of samples in training set: {}'.format(training_dataset.number_of_samples))
    print('The mean number of samples from each experiment used for training is {} with variance {}'.format(np.mean(training_dataset.df_lengths), np.std(training_dataset.df_lengths)))
    print('Number of samples in validation set: {}'.format(validation_dataset.number_of_samples))
    print('')

    del training_dataset, validation_dataset

    # endregion

    # region Set basic training features: optimizer, loss, scheduler...

    # Select Optimizer
    optimizer = optim.Adam(net.parameters(), amsgrad=False, lr=a.lr_initial)

    # TODO: Verify if scheduler is working. Try tweaking parameters of below scheduler and try cyclic lr scheduler
    if a.reduce_lr_on_plateau:
        # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=0.1)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   'min',
                                                   factor=a.lr_decrease_factor,  # sqrt(0.1)
                                                   patience=a.lr_patience,
                                                   min_lr=a.lr_minimal,
                                                   threshold=a.min_delta,
                                                   verbose=True,
                                                   )

    # Select Loss Function
    # criterion = nn.MSELoss()  # Mean square error loss function, might be not the same as TF - MSE, not checked
    criterion = loss_msr_sequence_customizable(
        wash_out_len=a.wash_out_len,
        post_wash_out_len=a.post_wash_out_len,
        discount_factor=1.0
                                            )

    # region Print information about the network
    print_parameter_count(net)  # Seems not to function well
    # endregion

    # endregion

    # region Training loop

    ########################################################
    # Training
    ########################################################
    print("Starting training...")
    print('')
    time.sleep(0.001)

    # Create dictionary to store training history
    dict_history = {}
    dict_history['epoch'] = []
    dict_history['time'] = []
    dict_history['lr'] = []
    dict_history['train_loss'] = []
    dict_history['dev_loss'] = []
    dict_history['dev_gain'] = []
    dict_history['test_loss'] = []
    dev_gain = 1

    dev_loss = validate(net, criterion, validation_generator)
    print('Validation loss before starting training is {}'.format(dev_loss))
    #
    # test_input = torch.tile(torch.reshape(torch.tensor([0.23], dtype=torch.float32), (1, 1, -1)), (1, 30, 1))
    # output = net(test_input)

    # The epoch_saved variable will indicate from which epoch is the last RNN model,
    # which was good enough to be saved
    torch.save(net.state_dict(), os.path.join(net_info.path_to_net, 'ckpt' + '.pt'), _use_new_zipfile_serialization=False)
    epoch_saved = -1
    post_epoch_training_loss = []
    for epoch in range(a.num_epochs):

        ###########################################################################################################
        # Training - Iterate batches
        ###########################################################################################################
        train_loss = train(net, criterion, training_generator, optimizer)

        ###########################################################################################################
        # Validation - Iterate batches
        ###########################################################################################################
        dev_loss = validate(net, criterion, validation_generator)


        # Get current learning rate

        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

        if a.reduce_lr_on_plateau:
            scheduler.step(dev_loss)

        # Write the summary information about the training for the just completed epoch to a dictionary

        dict_history['epoch'].append(epoch)
        dict_history['lr'].append(lr_curr)
        dict_history['train_loss'].append(
            train_loss.detach().cpu().numpy())
        dict_history['dev_loss'].append(
            dev_loss.detach().cpu().numpy())

        # Get relative loss gain for network evaluation
        if epoch >= 1:
            dev_gain = (dict_history['dev_loss'][epoch - 1] - dict_history['dev_loss'][epoch]) / \
                       dict_history['dev_loss'][epoch - 1]
        dict_history['dev_gain'].append(dev_gain)

        # Print the summary information about the training for the just completed epoch
        print('\nEpoch: %3d of %3d | '
              'LR: %1.5f | '
              'Train-L: %6.4e | '
              'Val-L: %6.4e | '
              'Val-Gain: %3.2f |' % (dict_history['epoch'][epoch], a.num_epochs - 1,
                                     dict_history['lr'][epoch],
                                     dict_history['train_loss'][epoch],
                                     dict_history['dev_loss'][epoch],
                                     dict_history['dev_gain'][epoch] * 100))
        print('')

        if a.validate_also_on_training_set:
            post_epoch_training_loss.append(validate(net, criterion, training_generator))

            # Save the best model with the lowest dev loss
        # Always save the model from epoch 0
        # TODO: this is a bug: you should only save the model from epoch 0 if there is no pretraind network
        if epoch == 0:
            min_dev_loss = dev_loss
        # If current loss smaller equal than minimal till now achieved loss,
        # save the current RNN model and save its loss as minimal ever achieved
        if True:
            epoch_saved = epoch
            min_dev_loss = dev_loss
            torch.save(net.state_dict(), os.path.join(net_info.path_to_net, 'ckpt' + '.pt'), _use_new_zipfile_serialization=False)
            print('>>> saving best model from epoch {}'.format(epoch))
            print('')
        else:
            print('>>> We keep model from epoch {}'.format(epoch_saved))
            print('')

    loss = dict_history['train_loss']
    validation_loss = dict_history['dev_loss']

    # endregion

    return loss, validation_loss, post_epoch_training_loss

def train(net, criterion, training_generator, optimizer):
    # Set RNN in training mode
    net = net.train()
    # Define variables accumulating training loss and counting training batchs
    train_loss = 0
    train_batches = 0

    # Iterate training over available batches
    # tqdm() is just a function which displays the progress bar
    # Otherwise the line below is the same as "for batch, labels in train_generator:"
    for batch, labels in tqdm(training_generator):  # Iterate through batches

        # Reset the network (internal states of hidden layers and output history not the weights!)
        net.reset()

        # Further modifying the input and output form to fit network requirements
        batch = batch.float().to(device)
        labels = labels.float().to(device)

        # # Reset memory of gradients
        # optimizer.zero_grad()

        # Warm-up (open loop prediction) to settle the internal state of RNN hidden layers
        # net(network_input=batch[:, :a.wash_out_len, :])

        # Reset memory of gradients
        optimizer.zero_grad()

        # Forward propagation - These are the results from which we calculate the update to RNN weights
        # GRU Input size must be (exp_len, batch, input_size)
        # out = net(network_input=batch[:, a.wash_out_len:, :])
        out = net(network_input=batch)

        # Get loss
        # loss = criterion(out, labels[:, a.wash_out_len:, :])
        loss = criterion(out, labels)

        # Backward propagation
        loss.backward()

        # Gradient clipping - prevent gradient from exploding
        torch.nn.utils.clip_grad_norm_(net.parameters(), 100)

        # Update parameters
        optimizer.step()
        # Update variables for loss calculation
        batch_loss = loss.detach()
        train_loss += batch_loss  # Accumulate loss
        train_batches += 1  # Accumulate count so we can calculate mean later

    training_generator.dataset.on_epoch_end()

    return train_loss / train_batches  # This returns loss per datapoint

def validate(net, criterion, validation_generator):
    #
    # Set the network in evaluation mode
    net = net.eval()

    # Define variables accumulating evaluation loss and counting evaluation batches
    dev_loss = 0
    dev_batches = 0

    for (batch, labels) in tqdm(validation_generator):
        # Reset the network (internal states of hidden layers and output history not the weights!)
        net.reset()

        # Further modifying the input and output form to fit RNN requirements
        batch = batch.float().to(device)
        labels = labels.float().to(device)

        # Warm-up (open loop prediction) to settle the internal state of RNN hidden layers
        out = net(network_input=batch)

        # Get loss
        # For evaluation we always calculate loss over the whole maximal prediction period
        # This allow us to compare RNN models from different epochs
        # loss = criterion(out[:, a.wash_out_len:, :],
        #                  labels[:, a.wash_out_len:, :])
        loss = criterion(out, labels)

        # Update variables for loss calculation
        batch_loss = loss.detach()
        dev_loss += batch_loss  # Accumulate loss
        dev_batches += 1  # Accumulate count so we can calculate mean later

    # Reset the network (internal states of hidden layers and output history not the weights!)
    net.reset()

    return dev_loss/dev_batches  # This returns loss per datapoint
