from SI_Toolkit.Functions.General.Dataset import DatasetTemplate
from SI_Toolkit.Functions.General.Dataset import augment_data
import tensorflow.keras as keras


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
