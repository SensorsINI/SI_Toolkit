from SI_Toolkit.Functions.General.Dataset import DatasetTemplate
from SI_Toolkit.Functions.General.Dataset import augment_data
import tensorflow.keras as keras
import numpy as np


class Dataset(DatasetTemplate, keras.utils.Sequence):
    def __init__(self,
                 dfs,
                 args,
                 inputs=None,
                 outputs=None,
                 exp_len=None,
                 shuffle=True,
                 batch_size=None,
                 ):
        super(Dataset, self).__init__(dfs, args, inputs, outputs, exp_len, shuffle, batch_size)

    def __len__(self):
        return self.number_of_batches_to_use

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def on_epoch_end(self):
        self.data, self.labels = augment_data(self.data_original, self.labels_original)
        if self.tiv:
            self.scaling_tiv_epoch_factor += 1.0
            print('scaling_tiv_epoch_factor is now {}'.format(self.scaling_tiv_epoch_factor))
        self.shuffle_dataset()


class ExtendedHorizonDataset(DatasetTemplate, keras.utils.Sequence):
    def __init__(self,
                 dfs,
                 args,
                 inputs=None,
                 outputs=None,
                 exp_len=None,
                 shuffle=True,
                 batch_size=None,
                 ):
        # self.horizon_length = args.horizon_length
        super(ExtendedHorizonDataset, self).__init__(dfs, args, inputs, outputs,
                                                     exp_len, shuffle, batch_size)

    def __len__(self):
        return self.number_of_batches_to_use

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def get_series(self, idx, get_time_axis=False):
        """
        Requires the self.data to be a list of pandas dataframes
        """
        # Find index of the dataset in self.data and index of the starting point in this dataset
        idx_data_set = next(i for i, v in enumerate(self.df_lengths_cs) if v > idx)
        if idx_data_set == 0:
            pass
        else:
            idx -= self.df_lengths_cs[idx_data_set - 1]
        features = self.data[idx_data_set].iloc[idx:idx + self.shift_labels, :].to_numpy()
        # Every point in features has its target value corresponding to the next time step:
        targets = self.labels[idx_data_set].iloc[idx + 1:idx + 1 + self.shift_labels, :].to_numpy()

        # Perturb the translation invariant inputs
        if self.tiv:
            self.scaling_tiv = self.scaling_tiv_epoch_factor * np.random.uniform(-2, 2,
                                                                                 size=len(self.tiv))
            features[:, self.tiv_in_inputs_idx] += self.scaling_tiv[self.tiv_for_inputs_idx]
            targets[:, self.tiv_in_outputs_idx] += self.scaling_tiv[self.tiv_for_outputs_idx]

        # If get_time_axis try to obtain a vector of time data for the chosen sample
        if get_time_axis:
            try:
                # As targets and features are shifted by one timestep we have to make time_axis accordingly longer to cover both
                time_axis = self.time_axes[idx_data_set].to_numpy()[
                            idx:idx + self.exp_len + self.shift_labels]
            except IndexError:
                time_axis = []

        # Return results
        if get_time_axis:
            return features, targets, time_axis
        else:
            return features, targets


class ExtendedHorizonDatasetShift(ExtendedHorizonDataset, keras.utils.Sequence):
    def __init__(self,
                 dfs,
                 args,
                 inputs=None,
                 outputs=None,
                 exp_len=None,
                 shuffle=True,
                 batch_size=None,
                 ):
        args.shift_labels = 20
        super(ExtendedHorizonDataset, self).__init__(dfs, args, inputs, outputs,
                                                     exp_len, shuffle, batch_size)
        self.shift_labels = 1

    def increase_shift_labels(self):
        print(f'INCREASING SHIFT LABELS FROM {self.shift_labels} TO {self.shift_labels + 1}!!!')
        self.shift_labels += 1