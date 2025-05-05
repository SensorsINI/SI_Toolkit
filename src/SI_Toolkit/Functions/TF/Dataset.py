from SI_Toolkit.Functions.General.Dataset import DatasetTemplate
import tensorflow.keras as keras
import tensorflow as tf


class Dataset(DatasetTemplate, keras.utils.Sequence):
    def __init__(self,
                 dfs,
                 args,
                 inputs=None,
                 outputs=None,
                 exp_len=None,
                 shuffle=True,
                 batch_size=None,
                 normalization_info=None,
                 ):
        super(Dataset, self).__init__(dfs, args, inputs, outputs, exp_len, shuffle, batch_size,
                                      normalization_info=normalization_info)

    def __len__(self):
        return self.number_of_batches_to_use

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def to_tf_data(self):
        """
        Expose as tf.data.Dataset: cache raw, then augment with pure TF ops, then prefetch.
        """

        def gen():
            for i in range(len(self)):
                yield self.get_batch(i)

        output_signature = (
            tf.TensorSpec((self.batch_size, self.exp_len, len(self.inputs)), tf.float32),
            tf.TensorSpec((self.batch_size, self.exp_len, len(self.outputs)), tf.float32)
        )

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        ds = ds.cache()  # keep batches in memory/disk
        ds = ds.repeat()  # ←––––––––––––––––––––––––– key addition
        ds = ds.map(
            lambda x, y: self.DA.series_modification(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return ds.prefetch(tf.data.AUTOTUNE)