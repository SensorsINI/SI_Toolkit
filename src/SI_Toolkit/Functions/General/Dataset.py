import numpy as np
from copy import deepcopy
from SI_Toolkit.Functions.General.value_precision import set_value_precision
from tqdm import tqdm

def augment_data_placeholder(data, labels):
    return data, labels


try:
    from SI_Toolkit_ASF.ToolkitCustomization.data_augmentation import augment_data
except:
    print("Data augmentation function not found. Data will not be transformed no matter the value of AUGMENT_DATA in config_training.")
    augment_data = augment_data_placeholder


class DatasetTemplate:
    def __init__(self,
                 dfs,
                 args,
                 inputs=None,
                 outputs=None,
                 exp_len=None,
                 shuffle=True,
                 batch_size=None,
                 use_only_full_batches=True,
                 ):
        'Initialization - divide data in features and labels'

        if inputs is None and args.inputs is not None:
            self.inputs = args.inputs
        else:
            self.inputs = inputs
        if outputs is None and args.outputs is not None:
            self.outputs = args.outputs
        else:
            self.outputs = outputs

        if hasattr(args, 'translation_invariant_variables'):
            self.tiv = args.translation_invariant_variables
        else:
            self.tiv = []
        self.tiv_in_inputs_idx = [i for i, e in enumerate(self.inputs) if e in self.tiv]
        self.tiv_in_outputs_idx = [i for i, e in enumerate(self.outputs) if e in self.tiv]
        self.tiv_for_inputs_idx = [i for i, e in enumerate(self.tiv) if e in self.inputs]
        self.tiv_for_outputs_idx = [i for i, e in enumerate(self.tiv) if e in self.outputs]

        self.scaling_tiv_epoch_factor = 0.0
        self.scaling_tiv = None

        self.data = []
        self.labels = []
        self.time_axes = []

        dfs_split = []
        for df in dfs:
            if 'experiment_index' in df.columns:
                grouped = df.groupby(df.experiment_index)
                for i in df.experiment_index.unique():
                    dfs_split.append(grouped.get_group(i))
            else:
                dfs_split.append(df)

        dfs = dfs_split

        data_filter = DataFilter(args)

        for df in tqdm(dfs, desc="Processing data files"):
            df = data_filter.apply_filters(df)
            if len(df) == 0:
                continue
            needed_columns = list(set(self.inputs) | set(self.outputs))
            df = df[needed_columns]
            df = df.dropna(axis=0)
            if hasattr(args, 'quantization') and args.quantization['ACTIVATED'] and args.quantization['QUANTIZATION_DATASET'] != 'float':
                df = df.map(lambda x: set_value_precision(x, args.quantization['QUANTIZATION_DATASET']))
            if 'time' in df.columns:
                self.time_axes.append(df['time'])
            self.data.append(df[self.inputs])
            self.labels.append(df[self.outputs])

        self.data_original = deepcopy(self.data)
        self.labels_original = deepcopy(self.labels)

        if args.augment_data:
            self.data, self.labels = augment_data(self.data_original, self.labels_original)

        self.args = args

        self.shift_labels = self.args.shift_labels
        self.warm_up_len = self.args.wash_out_len

        self.exp_len = None
        self.df_lengths = []
        self.df_lengths_cs = []

        self.indices = []
        self.number_of_samples = 0
        self.number_of_batches = 0

        self.indices_to_use = []
        self.number_of_samples_to_use = 0
        self.number_of_batches_to_use = 0

        self.indices_subset = []
        self.number_of_samples_in_subset = 0
        self.number_of_batches_in_subset = 0

        self.shuffle = shuffle

        self.use_only_full_batches = use_only_full_batches

        self.reset_exp_len(exp_len=exp_len)
        self.reset_number_of_samples()
        self.shuffle_dataset()

        # Batch is of relevance only for TF, Pytorch does not use Dataset to form batches.
        self.batch_size = 1
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


    def reset_number_of_samples(self):

        self.df_lengths = []
        self.df_lengths_cs = []
        if type(self.data) == list:
            for data_set in self.data:
                self.df_lengths.append(data_set.shape[0] - (self.exp_len-1) - self.shift_labels)
                if not self.df_lengths_cs:
                    self.df_lengths_cs.append(self.df_lengths[0])
                else:
                    self.df_lengths_cs.append(self.df_lengths_cs[-1] + self.df_lengths[-1] - self.shift_labels)
            self.number_of_samples = self.df_lengths_cs[-1]

        else:
            self.number_of_samples = self.data.shape[0] - (self.exp_len-1) - self.shift_labels

        if np.any(np.array(self.df_lengths) < 1):
            raise ValueError('One of the datasets is too short to use it for training. Remove it manually and try again.')

        self.indices = np.arange(self.number_of_samples)
        self.indices_to_use = self.indices
        self.number_of_samples_to_use = self.number_of_samples


    def shuffle_dataset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

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

        features = self.data[idx_data_set].iloc[idx:idx + self.exp_len, :].to_numpy()
        # Every point in features has its target value corresponding to the next time step:
        targets = self.labels[idx_data_set].iloc[idx + self.shift_labels:idx + self.exp_len + self.shift_labels, :].to_numpy()

        # Perturb the translation invariant inputs
        if self.tiv:
            self.scaling_tiv = self.scaling_tiv_epoch_factor * np.random.uniform(-2, 2, size=len(self.tiv))
            features[:, self.tiv_in_inputs_idx] +=  self.scaling_tiv[self.tiv_for_inputs_idx]
            targets[:, self.tiv_in_outputs_idx] += self.scaling_tiv[self.tiv_for_outputs_idx]


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

    def __len__(self):
        raise NotImplementedError('You need to implement __len__ method for Dataset class.')

    def __getitem__(self, idx):
        raise NotImplementedError('You need to implement __getitem__ method for Dataset class.')

    # region Batch is of relevance only for TF, Pytorch does not use Dataset to form batches.
    def get_batch(self, idx_batch):
        features_batch = []
        targets_batch = []
        sample_idx = self.indices_to_use[self.batch_size * idx_batch: self.batch_size * (idx_batch + 1)]
        for i in sample_idx:
            features, targets = self.get_series(i)
            features_batch.append(features)
            targets_batch.append(targets)
        features_batch = np.stack(features_batch)
        targets_batch = np.stack(targets_batch)

        return features_batch, targets_batch

    def reset_batch_size(self, batch_size=None):

        if batch_size is None:
            self.batch_size = self.args.batch_size
        else:
            self.batch_size = batch_size

        self.number_of_batches = self.calculate_number_of_batches(self.number_of_samples, self.batch_size, self.use_only_full_batches)
        self.number_of_batches_to_use = self.number_of_batches

    @staticmethod
    def calculate_number_of_batches(number_of_samples, batch_size, use_only_full_batches=True):
        if use_only_full_batches:
            return int(np.floor(number_of_samples / float(batch_size)))
        else:
            return int(np.ceil(number_of_samples / float(batch_size)))

    # endregion

    def create_subset(self, number_of_samples_in_subset=None, shuffle=True):
        if number_of_samples_in_subset is None:
            number_of_samples_in_subset = self.number_of_samples
        if number_of_samples_in_subset > self.number_of_samples:
            raise ValueError('Requested subset bigger than whole dataset')

        self.number_of_samples_in_subset = number_of_samples_in_subset
        if shuffle:
            self.indices_subset = np.random.choice(self.number_of_samples, number_of_samples_in_subset, replace=False)
        else:
            self.indices_subset = self.indices[:number_of_samples_in_subset]

        self.number_of_batches_in_subset = self.calculate_number_of_batches(number_of_samples_in_subset, self.batch_size, self.use_only_full_batches)

    def use_subset(self):
        self.indices_to_use = self.indices_subset
        self.number_of_samples_to_use = self.number_of_samples_in_subset
        self.number_of_batches_to_use = self.number_of_batches_in_subset

    def use_full_set(self):
        self.indices_to_use = self.indices
        self.number_of_samples_to_use = self.number_of_samples
        self.number_of_batches_to_use = self.number_of_batches

    def on_epoch_end(self):
        if self.args.augment_data:
            self.data, self.labels = augment_data(self.data_original, self.labels_original)
            self.reset_exp_len(self.exp_len)
            self.reset_number_of_samples()
            # Batch is of relevance only for TF, Pytorch does not use Dataset to form batches.
            self.reset_batch_size(self.batch_size)
        if self.tiv:
            self.scaling_tiv_epoch_factor += 1.0
            print('scaling_tiv_epoch_factor is now {}'.format(self.scaling_tiv_epoch_factor))
        self.shuffle_dataset()


class DataFilter:
    def __init__(self, args):
        # Check if 'filters' field exists in the configuration
        if hasattr(args, 'filters') and isinstance(args.filters, list):
            self.filter_funcs = []
            for data_filter in args.filters:
                column = data_filter['column']
                condition = data_filter['condition']
                operator, value_str = condition.split(" ", 1)
                value = float(value_str)
                use_absolute = data_filter.get('absolute', False)

                # Function to apply or bypass abs()
                def apply_abs_if_needed(dataset_values, use_abs):
                    return abs(dataset_values) if use_abs else dataset_values

                # Creating lambda functions based on the operator
                if operator == '<':
                    self.filter_funcs.append(lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) < v])
                elif operator == '<=':
                    self.filter_funcs.append(lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) <= v])
                elif operator == '>':
                    self.filter_funcs.append(lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) > v])
                elif operator == '>=':
                    self.filter_funcs.append(lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) >= v])
                elif operator == '==':
                    self.filter_funcs.append(lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) == v])
                elif operator == '!=':
                    self.filter_funcs.append(lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) != v])
        else:
            # If 'filters' field is not in config, setup to bypass filtering
            self.filter_funcs = [lambda df: df]

    def apply_filters(self, df):
        for func in self.filter_funcs:
            df = func(df)
        return df
