from SI_Toolkit.Functions.General.Dataset import DatasetTemplate
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
        super(Dataset, self).__init__(dfs, args, inputs, outputs, exp_len, shuffle)

        self.batch_size = 1
        self.number_of_batches = 1

        self.reset_batch_size(batch_size=batch_size)

    def __len__(self):
        return self.number_of_batches

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def get_batch(self, idx_batch):
        features_batch = []
        targets_batch = []
        sample_idx = self.indices[self.batch_size * idx_batch: self.batch_size * (idx_batch + 1)]
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

        self.number_of_batches = int(np.ceil(self.number_of_samples / float(self.batch_size)))


    def on_epoch_end(self):
        if self.tiv:
            self.scaling_tiv_epoch_factor += 1.0
            print('scaling_tiv_epoch_factor is now {}'.format(self.scaling_tiv_epoch_factor))
        self.shuffle_dataset()
