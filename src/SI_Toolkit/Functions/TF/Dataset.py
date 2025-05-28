# dataset_tf.py
# Copyright (c) 2025 ...
# -----------------------------------------------------------------------------
# A fully ‑vectorised tf.data implementation that mirrors the semantics of the
# legacy Keras Sequence‑based loader found in DatasetTemplate.
# -----------------------------------------------------------------------------

from __future__ import annotations  # Python ≥3.11 – allows list | None type‑hints

import hashlib
import json
import os
import tempfile
import shutil
import atexit

import tensorflow as tf
from typing import List, Tuple

from SI_Toolkit.Functions.General.Dataset import DatasetTemplate

from tqdm.auto import tqdm


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

        # Setup temporary cache directory to store tf.data cache and ensure cleanup
        self._cache_dir_manager = tempfile.TemporaryDirectory()
        self._cache_dir = self._cache_dir_manager.name
        atexit.register(lambda: shutil.rmtree(self._cache_dir, ignore_errors=True))

    def _experiments_ds(self) -> tf.data.Dataset:
        # we know every row has `len(self.inputs)` features, but T can vary
        n_in = len(self.inputs)
        n_out = len(self.outputs)

        spec_feat = tf.TensorSpec(shape=(None, n_in),
                                  dtype=tf.as_dtype(self.data[0].dtype))
        spec_targ = tf.TensorSpec(shape=(None, n_out),
                                  dtype=tf.as_dtype(self.labels[0].dtype))

        return tf.data.Dataset.from_generator(
            lambda: zip(self.data, self.labels),
            output_signature=(spec_feat, spec_targ),
        )

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

        # 1) Build a dataset of full experiments:
        ds = self._experiments_ds()  # yields (full_series_feats, full_series_tars)

        # 2) Cache each full series:
        cache_prefix = self._get_cache_prefix(cache_dir=self._cache_dir)
        ds = ds.cache(cache_prefix)

        # 3) Repeat indefinitely so each epoch re-reads the cache:
        ds = ds.repeat()

        # 4) Extract sliding windows & interleave:
        ds = ds.interleave(
            lambda x, y: tf.data.Dataset.from_tensor_slices(self._tf_slide_windows(x, y)),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # 4) Fresh shuffle **every epoch** – mirrors on_epoch_end
        if self.shuffle:
            ds = ds.shuffle(
                buffer_size=min(self.number_of_samples, 10_000),
                seed=getattr(self.args, "random_seed", None),
                reshuffle_each_iteration=True,
            )

        # ———— 5) Batch *before* augmentation so `map` sees a real [B, T, …] tensor ————
        ds = ds.batch(self.batch_size, drop_remainder=self.use_only_full_batches)

        # 6) *Per-sample* augmentation inside graph (now truly per-sequence in batch)
        ds = ds.map(self._wrap_data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

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


    def build_cache(self, force: bool = False) -> None:
        """
        Materialise the `tf.data.Dataset.cache()` file **once** so the first
        epoch is fast.  Call this *before* `model.fit()`.

        Parameters
        ----------
        n_cpu : int, optional
            Size of the private thread-pool used while writing the cache.
            Defaults to all available physical cores.
        force : bool, default False
            Re-build the cache even if the `.index` file is present.
        """

        # 1) Where will the cache live?
        cache_prefix = self._get_cache_prefix(cache_dir=self._cache_dir)
        index_file   = cache_prefix + ".index"
        if os.path.exists(index_file) and not force:
            print(f"[build_cache] Cache already present: {cache_prefix}")
            return  # nothing to do

        # 2) Minimal pipeline: full experiments → cache → prefetch
        ds = self._experiments_ds()

        opts = tf.data.Options()
        opts.threading.private_threadpool_size = os.cpu_count()
        opts.threading.max_intra_op_parallelism = 1
        ds = ds.with_options(opts)

        ds = ds.cache(cache_prefix)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        # 3) Run a single pass to write .data / .index shards
        print(f"[build_cache] Writing cache to {cache_prefix} "
              f"using {os.cpu_count()} CPU threads …")
        for _ in tqdm(ds, desc="Building tf.data cache"):
            pass
        print("[build_cache] Done.")
