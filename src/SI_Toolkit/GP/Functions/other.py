import numpy as np

from types import SimpleNamespace

from SI_Toolkit.GP.DataSelector import DataSelector

def reformat_data_for_vaidation(data_val, inputs):
    X_val = np.empty(shape=[0, 6])
    Y_val = np.empty(shape=[0, 5])
    for df in data_val:
        df = df[inputs].to_numpy()
        X_val = np.vstack([X_val, df[:-1, :]])
        Y_val = np.vstack([Y_val, df[1:, :-1]])
    return X_val, Y_val


def preselect_data_for_GPs(data, inputs, outputs, number_of_bins, training, batch_size):
    """
    This function tries to balance the dataset partitioning the statespace into bins
    and filling the bins with data up to a predefined threshold.
    The redundant data for each bin is discarded.
    This function acts as wrapper for dataselector, TODO: Clean the code for this function and dataselector
    """

    a = SimpleNamespace()

    a.num = number_of_bins - 1
    a.training = training
    a.batch_size = batch_size

    a.wash_out_len = 0
    a.post_wash_out_len = 1

    a.inputs = inputs
    a.outputs = outputs

    DS = DataSelector(a)
    DS.load_data_into_selector(data)
    X, Y = DS.return_dataset_for_training(shuffle=True,
                                          inputs=inputs,
                                          outputs=outputs,
                                          raw=True)
    X = X.squeeze()
    Y = Y.squeeze()

    return X, Y
