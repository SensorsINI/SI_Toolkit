# dataset_tf.py
# Copyright (c) 2025 ...
# -----------------------------------------------------------------------------
# A fully ‑vectorised tf.data implementation that mirrors the semantics of the
# legacy Keras Sequence‑based loader found in DatasetTemplate.
# -----------------------------------------------------------------------------

from __future__ import annotations  # Python ≥3.11 – allows list | None type‑hints

import tensorflow as tf
from typing import List, Tuple

from SI_Toolkit.Functions.General.Dataset import DatasetTemplate


class Dataset(DatasetTemplate):
    """
    TF‑native counterpart of the legacy Keras `Sequence.

    The class produces a *functionally identical* data stream while exploiting
    `tf.data for efficiency:

    * identical **window extraction**           (`exp_len & shift_labels)
    * identical **per‑sample augmentation**     (runs *before* batching so that
      user code sees exactly the same shapes as before)
    * identical **shuffle semantics**           (fresh shuffle **every epoch**,
      even when a deterministic `random_seed is supplied – this matches
      `on_epoch_end in the old Python loop)
    * identical **batch sizing** and option to drop incomplete batches
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        dfs,
        args,
        inputs: List[str] | None = None,
        outputs: List[str] | None = None,
        exp_len: int | None = None,
        shuffle: bool = True,
        batch_size: int | None = None,
        normalization_info=None,
    ):
        # Delegates *all* bookkeeping to the shared template
        super().__init__(
            dfs,
            args,
            inputs,
            outputs,
            exp_len,
            shuffle,
            batch_size,
            normalization_info=normalization_info,
        )

        # One‑off NumPy→TF conversion (keeps original dtype precision)
        self._tf_data   = [tf.constant(a, dtype=tf.as_dtype(a.dtype)) for a in self.data]
        self._tf_labels = [tf.constant(a, dtype=tf.as_dtype(a.dtype)) for a in self.labels]

    def _experiments_ds(self) -> tf.data.Dataset:
        spec_feat = tf.TensorSpec(shape=(None, len(self.inputs)),
                                  dtype=self._tf_data[0].dtype)
        spec_targ = tf.TensorSpec(shape=(None, len(self.outputs)),
                                  dtype=self._tf_labels[0].dtype)

        def gen():
            for f, t in zip(self._tf_data, self._tf_labels):
                yield f, t

        return tf.data.Dataset.from_generator(gen,
                                              output_signature=(spec_feat, spec_targ))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _tf_slide_windows(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Vectorised sliding‑window extraction (matches `get_series).

        Each *window* produced has length `exp_len.  The label window is
        shifted *forward* by exactly `shift_labels steps such that shapes are

            features : `[W, exp_len, n_in ]
            targets  : `[W, exp_len, n_out]

        where `W = T − (exp_len + shift_labels) + 1.
        """
        full_len = self.exp_len + self.shift_labels  # == frame_length
        n_in, n_out = len(self.inputs), len(self.outputs)

        combined = tf.concat([x, y], axis=1)                    # «T, n_in + n_out»

        # Sliding framing along *time* axis (axis=0).  No padding: windows are
        # emitted only when *fully* inside the sequence – identical to the
        # Python slicing logic.
        windows = tf.signal.frame(
            combined,
            frame_length=full_len,
            frame_step=1,
            axis=0,
            pad_end=False,
        )                                                       # «W, full_len, …»

        feats = windows[:, :self.exp_len, :n_in]                # «W, exp_len, n_in »
        tars  = windows[:, self.shift_labels:, n_in:]           # «W, exp_len, n_out»
        return feats, tars

    def _wrap_data_augmentation(self, feats: tf.Tensor, tars: tf.Tensor):
        """Runs user‑supplied augmentation inside the TensorFlow graph.

        If the augmentation function cannot operate on tensors (e.g. uses NumPy
        internals), we *seamlessly* fall back to `tf.numpy_function while
        preserving static shapes – this guarantees shape inference downstream
        (especially important before batching).
        """
        try:
            return self.data_augmentation(feats, tars)
        except (TypeError, tf.errors.OperatorNotAllowedInGraphError):
            feats2, tars2 = tf.numpy_function(
                lambda f, t: self.data_augmentation(f, t),
                inp=[feats, tars],
                Tout=[feats.dtype, tars.dtype],
            )
            # numpy_function erases static shape information → restore
            feats2.set_shape(feats.shape)
            tars2.set_shape(tars.shape)
            return feats2, tars2

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def to_tf_data(self) -> tf.data.Dataset:
        """Builds a ready‑to‑consume `tf.data.Dataset pipeline."""

        # 1) *Per‑experiment* dataset
        ds = self._experiments_ds()

        # 2) Vectorised window extraction & flat interleave across experiments
        #    ────────────────────────────────────────────────────────────────
        ds = ds.interleave(
            lambda x, y: tf.data.Dataset.from_tensor_slices(self._tf_slide_windows(x, y)),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
        )  # At this point each element == single (feats, tars) *window*

        cache_prefix = self._get_cache_prefix()
        ds = ds.cache(cache_prefix).repeat()

        # 4) Fresh shuffle **every epoch** – mirrors on_epoch_end
        if self.shuffle:
            ds = ds.shuffle(
                buffer_size=min(self.number_of_samples, 10_000),
                seed=getattr(self.args, "random_seed", None),
                reshuffle_each_iteration=True,  # ◀ crucial for epoch‑wise shuffle
            )

        # 5) *Per‑sample* augmentation (identical call signature to legacy)
        ds = ds.map(self._wrap_data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

        # 6) Batch formation (optionally drop last incomplete batch)
        ds = ds.batch(self.batch_size, drop_remainder=self.use_only_full_batches)

        # 7) Pipeline overlap (CPU prepare ↔ GPU consume)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    # ------------------------------------------------------------------ #
    # Keras‑style helpers for drop‑in replacement
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        """Number of *batches* per epoch (unchanged)."""
        return self.number_of_batches_to_use

    def __iter__(self):
        """Py‑native iterator so that `for x, y in Dataset(...): ... works."""
        return iter(self.to_tf_data())

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
