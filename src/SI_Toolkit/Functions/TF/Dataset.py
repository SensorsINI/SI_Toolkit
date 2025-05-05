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
        Expose this DatasetTemplate as a tf.data.Dataset.
        Batches, prefetches, and leverages AUTOTUNE for parallelism.
        """
        # generator yielding (features, targets) numpy batches
        def gen():
            for i in range(len(self)):
                yield self.get_batch(i)

        # signature matches what __getitem__ spits out
        output_signature = (
            tf.TensorSpec(
                shape=(self.batch_size, self.exp_len, len(self.inputs)),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(self.batch_size, self.exp_len, len(self.outputs)),
                dtype=tf.float32
            )
        )

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=output_signature
        )
        return ds.prefetch(tf.data.AUTOTUNE)
