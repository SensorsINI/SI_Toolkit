import tensorflow.keras as keras
import numpy as np


class Dataset(keras.utils.Sequence):
    def __init__(self,
                 dfs,
                 args,
                 inputs=None, outputs=None,
                 batch_size=None,
                 exp_len=None,
                 shuffle=True):
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
            if 'time' in df.columns:
                self.time_axes.append(df['time'])
            self.data.append(df[self.inputs])
            self.labels.append(df[self.outputs])

        self.args = args

        self.shift_labels = self.args.shift_labels

        self.exp_len = None
        self.warm_up_len = self.args.wash_out_len
        self.df_lengths = []
        self.df_lengths_cs = []
        self.number_of_samples = 0

        # NEW PART FOR TENSORFLOW:
        # __get_item__ must return a batch
        self.batch_size = 1
        self.number_of_batches = 1
        self.shuffle = shuffle
        self.indexes = []

        self.reset_exp_len(exp_len=exp_len)
        self.reset_batch_size(batch_size=batch_size)

    def reset_exp_len(self, exp_len=None):
        """
        This method should be used if the user wants to change the exp_len without creating new Dataset
        Please remember that one can reset it again to come back to old configuration (from ParseArgs)
        :param exp_len: Gives new user defined exp_len. Call empty to come back to default. Min is 1!
        """
        if exp_len is None:
            self.exp_len = self.args.wash_out_len+self.args.post_wash_out_len  # Sequence length
        else:
            self.exp_len = exp_len

        if self.exp_len is None or self.exp_len < 1:
            if self.exp_len is None:
                raise('At this point experiment length should not be None')
            else:
                raise ValueError('Experiment length (exp_len) must be 1 at least!')

        self.df_lengths = []
        self.df_lengths_cs = []
        if type(self.data) == list:
            for data_set in self.data:
                self.df_lengths.append(data_set.shape[0] - self.exp_len - self.shift_labels)
                if not self.df_lengths_cs:
                    self.df_lengths_cs.append(self.df_lengths[0])
                else:
                    self.df_lengths_cs.append(self.df_lengths_cs[-1] + self.df_lengths[-1] - self.shift_labels)
            self.number_of_samples = self.df_lengths_cs[-1]

        else:
            self.number_of_samples = self.data.shape[0] - self.exp_len - self.shift_labels

        if np.any(np.array(self.df_lengths) < 1):
            raise ValueError('One of the datasets is too short to use it for training. Remove it manually and try again.')

        self.number_of_batches = int(np.ceil(self.number_of_samples / float(self.batch_size)))

        self.indexes = np.arange(self.number_of_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def reset_batch_size(self, batch_size=None):

        if batch_size is None:
            self.batch_size = self.args.batch_size
        else:
            self.batch_size = batch_size

        self.number_of_batches = int(np.ceil(self.number_of_samples / float(self.batch_size)))

        self.indexes = np.arange(self.number_of_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # In TF it must return the number of batches
        return self.number_of_batches

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



        features = self.data[idx_data_set].to_numpy()[idx:idx + self.exp_len, :]
        # Every point in features has its target value corresponding to the next time step:
        targets = self.labels[idx_data_set].to_numpy()[idx+self.shift_labels:idx + self.exp_len+self.shift_labels, :]


        # If get_time_axis try to obtain a vector of time data for the chosen sample
        if get_time_axis:
            try:
                # As targets and features are shifted by one timestep we have to make time_axis accordingly longer to cover both
                time_axis = self.time_axes[idx_data_set].to_numpy()[idx:idx + self.exp_len + self.shift_labels]
            except IndexError:
                time_axis = []

        # Return results
        if get_time_axis:
            return features, targets, time_axis
        else:
            return features, targets

    def __getitem__(self, idx_batch):
        features_batch = []
        targets_batch = []
        sample_idx = self.indexes[self.batch_size * idx_batch: self.batch_size * (idx_batch + 1)]
        for i in sample_idx:
            features, targets = self.get_series(i)
            features_batch.append(features)
            targets_batch.append(targets)
        features_batch = np.stack(features_batch)
        targets_batch = np.stack(targets_batch)

        return features_batch, targets_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)