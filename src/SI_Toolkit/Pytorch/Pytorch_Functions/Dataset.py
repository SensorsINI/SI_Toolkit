from torch.utils import data

import numpy as np

import copy


class Dataset(data.Dataset):
    def __init__(self, dfs, args,
                 inputs=None, outputs=None,
                 exp_len=None):
        'Initialization - divide data in features and labels'

        if inputs is None and args.inputs is not None:
            self.inputs = args.inputs
        else:
            self.inputs = inputs
        if outputs is None and args.outputs is not None:
            self.outputs = args.outputs
        else:
            self.outputs = outputs

        self.data = []
        self.labels = []
        self.time_axes = []

        for df in dfs:
            self.time_axes.append(df['time'])
            # Get Raw Data
            features = copy.deepcopy(df)
            targets = copy.deepcopy(df)

            features.drop(features.tail(1).index, inplace=True)  # Drop last row
            targets.drop(targets.head(1).index, inplace=True)
            features.reset_index(inplace=True)  # Reset index
            targets.reset_index(inplace=True)

            features = features[self.inputs]
            targets = targets[self.outputs]

            self.data.append(features)
            self.labels.append(targets)

        self.args = args

        self.exp_len = None
        self.warm_up_len = self.args.wash_out_len
        self.df_lengths = []
        self.df_lengths_cs = []
        self.number_of_samples = 0

        self.reset_exp_len(exp_len=exp_len)

    def reset_exp_len(self, exp_len=None):
        """
        This method should be used if the user wants to change the exp_len without creating new Dataset
        Please remember that one can reset it again to come back to old configuration
        :param exp_len: Gives new user defined exp_len. Call empty to come back to default.
        """
        if exp_len is None:
            self.exp_len = self.args.exp_len  # Sequence length
        else:
            self.exp_len = exp_len

        self.df_lengths = []
        self.df_lengths_cs = []
        if type(self.data) == list:
            for data_set in self.data:
                self.df_lengths.append(data_set.shape[0] - self.exp_len)
                if not self.df_lengths_cs:
                    self.df_lengths_cs.append(self.df_lengths[0])
                else:
                    self.df_lengths_cs.append(self.df_lengths_cs[-1] + self.df_lengths[-1])
            self.number_of_samples = self.df_lengths_cs[-1]

        else:
            self.number_of_samples = self.data.shape[0] - self.exp_len

    def __len__(self):
        'Total number of samples'
        return self.number_of_samples

    def __getitem__(self, idx, get_time_axis=False):
        """
        Requires the self.data to be a list of pandas dataframes
        """
        # Find index of the dataset in self.data and index of the starting point in this dataset
        idx_data_set = next(i for i, v in enumerate(self.df_lengths_cs) if v > idx)
        if idx_data_set == 0:
            pass
        else:
            idx -= self.df_lengths_cs[idx_data_set - 1]

        # Get data
        features = self.data[idx_data_set].to_numpy()[idx:idx + self.exp_len, :]
        # Every point in features has its target value corresponding to the next time step:
        targets = self.labels[idx_data_set].to_numpy()[idx:idx + self.exp_len]
        # After feeding the whole sequence we just compare the final output of the RNN with the state following afterwards
        # targets = self.labels[idx_data_set].to_numpy()[idx + self.exp_len-1]

        # If get_time_axis try to obtain a vector of time data for the chosen sample
        if get_time_axis:
            try:
                time_axis = self.time_axes[idx_data_set].to_numpy()[idx:idx + self.exp_len + 1]
            except IndexError:
                time_axis = []

        # Return results
        if get_time_axis:
            return features, targets, time_axis
        else:
            return features, targets

    def get_experiment(self, idx=None):
        if self.time_axes is None:
            raise Exception('No time information available!')
        if idx is None:
            idx = np.random.randint(0, self.number_of_samples)
        return self.__getitem__(idx, get_time_axis=True)
