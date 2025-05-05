from SI_Toolkit.Functions.General.Dataset import DatasetTemplate
import tensorflow.keras as keras
import tensorflow as tf
import hashlib
import json
import os

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

        cache_prefix = self._get_cache_prefix()
        ds = ds.cache(cache_prefix)

        ds = ds.repeat()

        ds = ds.map(
            lambda x, y: self.DA.series_modification(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return ds.prefetch(tf.data.AUTOTUNE)

    def _get_cache_prefix(self, cache_dir: str = './tf_dataset_cache') -> str:
        """
        Build a deterministic cache prefix based on dataset parameters.
        """

        # 1) Gather everything that affects the generator
        key_dict = {
            'inputs': self.inputs,
            'outputs': self.outputs,
            'exp_len': self.exp_len,
            'batch_size': self.batch_size,
            'normalization_info': self.args.path_to_normalization_info,
            # include any args fields starting with 'filter_'
            **{k: v for k, v in vars(self.args).items() if k.startswith('filter_')},
        }

        # 2) Deterministic serialization and hashing
        key_str = json.dumps(key_dict, sort_keys=True, default=str)
        key_hash = hashlib.sha1(key_str.encode()).hexdigest()

        # 3) Optional tag (e.g. 'train' or 'val')
        tag_eff = getattr(self, 'cache_tag', '')
        fname = f"cache_{tag_eff}_{key_hash}"  # **no extension**

        # 4) Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        prefix_path = os.path.abspath(os.path.join(cache_dir, fname))

        # 5) Check for the materialized TF cache file (the “.index” suffix)
        index_file = prefix_path + '.index'
        if os.path.exists(index_file):
            print(f"[Cache] Reusing cache prefix: {prefix_path}")
        else:
            print(f"[Cache] Generating new cache at prefix: {prefix_path}")

        return prefix_path
