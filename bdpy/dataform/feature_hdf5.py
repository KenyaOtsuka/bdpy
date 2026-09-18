"""Chunked HDF5 storage for DNN features.

The legacy feature layout writes one ``.mat`` file per stimulus, so the only
file boundary is the sample axis and a partial read along the feature axes is
impossible. This module writes the chunked layout instead: one ``<layer>.h5``
per layer, holding the whole layer in a dataset that is explicitly chunked on
both the sample axis and the outermost feature axis.

Schema version 1::

    <layer>.h5
        /features   (n_samples, *feature_shape)
        /labels     (n_samples,)   variable-length UTF-8
        attrs:
            bdpy_format         = "features"
            bdpy_format_version = 1
            layer               = "<layer name>"

One file per layer rather than one file for everything: layers differ in shape
and in optimal chunk shape, and separate files keep regeneration, copying and
parallel writing per-layer.

Compression is off by default. The point of this format is partial-read latency,
and every compressed chunk costs a decompression on the way out; pass
``compression=`` when size matters more than speed.

This file is a part of BdPy.

Examples
--------
Write a whole layer at once::

    save_features('features/conv5.h5', array, labels)

Write incrementally, as feature extraction produces one stimulus at a time::

    with FeatureWriter('features/conv5.h5', feature_shape=(256, 13, 13),
                       dtype=np.float32) as writer:
        for label, feature in extract():
            writer.append(feature, label)

Migrate an existing ``.mat`` tree::

    convert_features_to_hdf5('features_mat', 'features_h5')
"""

import os
from types import TracebackType
from typing import Iterable, Optional, Sequence, Tuple, Type

import h5py
import numpy as np

from ._feature_chunking import DEFAULT_TARGET_CHUNK_BYTES, choose_chunk_shape
from ._feature_store import (
    FEATURES_DATASET,
    FORMAT_ATTR,
    FORMAT_NAME,
    FORMAT_VERSION_ATTR,
    HDF5_EXT,
    LABELS_DATASET,
    SUPPORTED_FORMAT_VERSION,
    MatFeatureStore,
)

__all__ = [
    "FeatureWriter",
    "convert_features_to_hdf5",
    "save_features",
]

#: Number of samples the resizable dataset grows by at a time.
_GROW_BLOCK = 64


def _string_dtype() -> np.dtype:
    dtype: np.dtype = h5py.string_dtype(encoding="utf-8")
    return dtype


def _write_header(f: h5py.File, layer: Optional[str]) -> None:
    f.attrs[FORMAT_ATTR] = FORMAT_NAME
    f.attrs[FORMAT_VERSION_ATTR] = SUPPORTED_FORMAT_VERSION
    if layer is not None:
        f.attrs["layer"] = layer


def save_features(
    path: str,
    features: np.ndarray,
    labels: Sequence[str],
    layer: Optional[str] = None,
    chunks: Optional[Tuple[int, ...]] = None,
    target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
    compression: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
) -> None:
    """Write a whole layer to a chunked HDF5 feature file.

    Parameters
    ----------
    path : str
        Output file. Overwritten if it exists.
    features : numpy.ndarray
        Feature array of shape ``(n_samples, *feature_shape)``.
    labels : sequence of str
        One stimulus label per sample, in the same order as `features`.
    layer : str, optional
        Layer name recorded in the file. Defaults to the file's basename.
    chunks : tuple of int, optional
        Explicit chunk shape. Defaults to :func:`choose_chunk_shape`.
    target_chunk_bytes : int, optional
        Per-chunk byte budget used when `chunks` is not given.
    compression : str, optional
        h5py compression filter, e.g. ``'lzf'`` or ``'gzip'``. ``None``
        (default) stores the features uncompressed, which keeps partial reads
        fast.
    dtype : numpy.dtype, optional
        Dtype to store. Defaults to the dtype of `features`.

    Raises
    ------
    ValueError
        If `labels` does not have one entry per sample, or `features` has fewer
        than two axes.
    """
    features = np.asarray(features)
    if features.ndim < 2:
        raise ValueError(
            "features need a sample axis and at least one feature axis; "
            "got shape {}".format(features.shape)
        )
    labels = list(labels)
    if len(labels) != features.shape[0]:
        raise ValueError(
            "got {} labels for {} samples".format(len(labels), features.shape[0])
        )

    dtype = np.dtype(features.dtype if dtype is None else dtype)
    if chunks is None:
        chunks = choose_chunk_shape(
            features.shape, dtype, target_bytes=target_chunk_bytes
        )
    if layer is None:
        layer = os.path.splitext(os.path.basename(path))[0]

    _makedirs_for(path)
    with h5py.File(path, "w") as f:
        _write_header(f, layer)
        f.create_dataset(
            FEATURES_DATASET,
            data=features.astype(dtype, copy=False),
            chunks=chunks,
            compression=compression,
        )
        f.create_dataset(LABELS_DATASET, data=labels, dtype=_string_dtype())


class FeatureWriter:
    """Incremental writer for a chunked HDF5 feature file.

    Feature extraction produces one stimulus at a time, so the datasets are
    created resizable (``maxshape=(None, *feature_shape)``) and grown in blocks
    as samples arrive. Use it as a context manager, or call :meth:`close`.

    Parameters
    ----------
    path : str
        Output file. Overwritten if it exists.
    feature_shape : sequence of int
        Shape of a single sample's features, without the sample axis.
    dtype : numpy.dtype
        Dtype to store.
    layer : str, optional
        Layer name recorded in the file. Defaults to the file's basename.
    n_samples : int, optional
        Expected number of samples, if known. Used only to pick a chunk shape.
    chunks : tuple of int, optional
        Explicit chunk shape, overriding the byte budget.
    target_chunk_bytes : int, optional
        Per-chunk byte budget used when `chunks` is not given.
    compression : str, optional
        h5py compression filter. ``None`` (default) stores uncompressed.

    Examples
    --------
    >>> with FeatureWriter('conv5.h5', (256, 13, 13), np.float32) as w:  # doctest: +SKIP
    ...     w.append(feature, 'img0001')
    """

    def __init__(
        self,
        path: str,
        feature_shape: Sequence[int],
        dtype: np.dtype,
        layer: Optional[str] = None,
        n_samples: Optional[int] = None,
        chunks: Optional[Tuple[int, ...]] = None,
        target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
        compression: Optional[str] = None,
    ):
        self._feature_shape = tuple(int(s) for s in feature_shape)
        if not self._feature_shape:
            raise ValueError("feature_shape must have at least one axis")
        self._dtype = np.dtype(dtype)
        self._n = 0

        if chunks is None:
            chunks = choose_chunk_shape(
                (n_samples if n_samples is not None else 0, *self._feature_shape),
                self._dtype,
                target_bytes=target_chunk_bytes,
                n_samples_known=n_samples is not None,
            )
        if layer is None:
            layer = os.path.splitext(os.path.basename(path))[0]

        _makedirs_for(path)
        self._file: Optional[h5py.File] = h5py.File(path, "w")
        _write_header(self._file, layer)
        self._features = self._file.create_dataset(
            FEATURES_DATASET,
            shape=(0, *self._feature_shape),
            maxshape=(None, *self._feature_shape),
            dtype=self._dtype,
            chunks=chunks,
            compression=compression,
        )
        self._labels = self._file.create_dataset(
            LABELS_DATASET,
            shape=(0,),
            maxshape=(None,),
            dtype=_string_dtype(),
            chunks=(max(1, _GROW_BLOCK),),
        )

    @property
    def n_samples(self) -> int:
        """Number of samples written so far."""
        return self._n

    def append(self, feature: np.ndarray, label: str) -> None:
        """Append a single sample.

        Parameters
        ----------
        feature : numpy.ndarray
            One sample, either ``feature_shape`` or ``(1, *feature_shape)``.
        label : str
            Its stimulus label.
        """
        feature = np.asarray(feature)
        if feature.shape == self._feature_shape:
            feature = feature[np.newaxis]
        self.extend(feature, [label])

    def extend(self, features: np.ndarray, labels: Sequence[str]) -> None:
        """Append several samples at once.

        Parameters
        ----------
        features : numpy.ndarray
            Samples of shape ``(n, *feature_shape)``.
        labels : sequence of str
            One label per sample.

        Raises
        ------
        ValueError
            If the feature shape does not match the writer's, or the label count
            does not match the sample count.
        RuntimeError
            If the writer is already closed.
        """
        if self._file is None:
            raise RuntimeError("FeatureWriter is closed")
        features = np.asarray(features)
        labels = list(labels)
        if features.shape[1:] != self._feature_shape:
            raise ValueError(
                "expected features of shape (n, {}), got {}".format(
                    ", ".join(str(s) for s in self._feature_shape), features.shape
                )
            )
        if len(labels) != features.shape[0]:
            raise ValueError(
                "got {} labels for {} samples".format(len(labels), features.shape[0])
            )
        if not labels:
            return

        new_n = self._n + len(labels)
        self._features.resize(new_n, axis=0)
        self._labels.resize(new_n, axis=0)
        self._features[self._n:new_n] = features.astype(self._dtype, copy=False)
        self._labels[self._n:new_n] = labels
        self._n = new_n

    def close(self) -> None:
        """Close the underlying file. Idempotent."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "FeatureWriter":
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Close the file on leaving the context manager."""
        self.close()


def convert_features_to_hdf5(
    src_dir: str,
    dst_dir: str,
    layers: Optional[Iterable[str]] = None,
    ext: str = "mat",
    key: str = "feat",
    overwrite: bool = False,
    batch_size: int = 64,
    target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
    compression: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Convert a legacy per-stimulus feature directory to chunked HDF5.

    Reads `src_dir` through the same backend :class:`~bdpy.dataform.Features`
    uses for the legacy layout, so the output is by construction what the legacy
    reader sees. Samples are streamed in batches, so a layer is never held in
    memory in full.

    Parameters
    ----------
    src_dir : str
        Legacy directory, ``<src_dir>/<layer>/<label>.<ext>``.
    dst_dir : str
        Output directory; ``<dst_dir>/<layer>.h5`` is written per layer. Created
        if missing.
    layers : iterable of str, optional
        Layers to convert. Defaults to every layer found in `src_dir`.
    ext : str, optional
        Extension of the legacy files (default: ``'mat'``).
    key : str, optional
        Variable name inside the legacy files (default: ``'feat'``).
    overwrite : bool, optional
        Overwrite an existing output file instead of skipping it
        (default: False).
    batch_size : int, optional
        Number of stimulus files read per batch.
    target_chunk_bytes : int, optional
        Per-chunk byte budget.
    compression : str, optional
        h5py compression filter. ``None`` (default) stores uncompressed.
    verbose : bool, optional
        Print progress.

    Raises
    ------
    KeyError
        If a requested layer is not present in `src_dir`.
    """
    store = MatFeatureStore(src_dir, ext=ext, key=key)
    selected = list(store.layers) if layers is None else list(layers)
    missing = [layer for layer in selected if layer not in store.layers]
    if missing:
        raise KeyError(
            "Layer(s) {} not found in {}".format(", ".join(missing), src_dir)
        )

    os.makedirs(dst_dir, exist_ok=True)
    all_labels = store.labels

    for layer in selected:
        out_path = os.path.join(dst_dir, layer + "." + HDF5_EXT)
        if os.path.exists(out_path) and not overwrite:
            if verbose:
                print("{} already exists. Skipped.".format(out_path))
            continue

        full_shape = store.shape(layer)
        writer = FeatureWriter(
            out_path,
            feature_shape=full_shape[1:],
            dtype=store.dtype(layer),
            layer=layer,
            n_samples=full_shape[0],
            target_chunk_bytes=target_chunk_bytes,
            compression=compression,
        )
        try:
            for start in range(0, len(all_labels), batch_size):
                batch = all_labels[start:start + batch_size]
                writer.extend(store.read(layer, batch), batch)
        finally:
            writer.close()

        if verbose:
            print("Saved {} ({} samples).".format(out_path, full_shape[0]))


def _makedirs_for(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
