# GPs

### The GP part is specific for cartpole and not maintained actively.

## DataSelector
DataSelector is a filter which partition statespace into bins,
sort the available data filling the bins up to predefined amount of data, and discard the rest.
This is an attempt to get a more uniform data distribution. 

DataSelector_nn was my (Marcin's) attempt to use the data selector with neural network training.
I think it was working now, but we abandon this direction
and I've never merged it with the previous data selector we used for GPs.
I have no time for it now so I just put it here. 

In TF/Training.py it was imported with

```
try:
    from SI_Toolkit_ASF.DataSelector import DataSelector
except:
    print('No DataSelector found.')
```

and used with:


```
def train_network_core(net, net_info, training_dfs, validation_dfs, test_dfs, a):

    # region Prepare data for training
    # DataSelectorInstance = DataSelector(a)
    # DataSelectorInstance.load_data_into_selector(training_dfs_norm)
    # training_dataset = DataSelectorInstance.return_dataset_for_training(shuffle=True, inputs=net_info.inputs, outputs=net_info.outputs)
    training_dataset = Dataset(training_dfs, a, shuffle=True, inputs=net_info.inputs, outputs=net_info.outputs)
```