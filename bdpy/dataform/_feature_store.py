"""Storage backends for DNN feature collections.

``Features`` used to be welded to one on-disk layout: a directory per layer
holding one ``.mat`` file per stimulus. That layout has a file boundary only on
the sample axis, so reading a few channels still costs a full read of every
stimulus. This module puts a backend interface between ``Features`` and the
bytes, so the legacy layout and the chunked HDF5 layout are interchangeable and
a future backend needs no change to ``Features`` itself.

Every backend answers the same question -- "give me these labels, sliced this
way along the feature axes" -- and differs only in how much it has to read to do
it. The legacy backend loads whole stimulus files and slices in memory; the HDF5
backend pushes the slice down to h5py and reads only the chunks it covers.

This file is a part of BdPy.
"""

import glob
import os
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from . import _mat_v73
from ._feature_chunking import DEFAULT_TARGET_CHUNK_BYTES

__all__ = [
    "FeatureStore",
    "HDF5FeatureStore",
    "MatFeatureStore",
    "detect_format",
]

#: Root attribute marking a file as bdpy chunked feature storage.
FORMAT_ATTR = "bdpy_format"
#: Root attribute holding the schema version.
FORMAT_VERSION_ATTR = "bdpy_format_version"
#: Value of ``FORMAT_ATTR`` for feature files.
FORMAT_NAME = "features"
#: Highest schema version this module can read.
SUPPORTED_FORMAT_VERSION = 1
#: Dataset holding the feature array, shape ``(n_samples, *feature_shape)``.
FEATURES_DATASET = "features"
#: Dataset holding the stimulus labels, shape ``(n_samples,)``.
LABELS_DATASET = "labels"
#: Extension of a chunked feature file.
HDF5_EXT = "h5"

# A slice spec for the feature axes: what ``numpy.s_[...]`` produces.
FeatureSlice = Union[slice, int, Sequence[int], np.ndarray, Tuple, None]


def _load_array_with_key(key: str, path: str) -> np.ndarray:
    # v5 .mat via scipy, v7.3 (HDF5) via h5py; avoids hdf5storage on the load
    # path, which breaks under NumPy 2.0 (see bdpy/dataform/_mat_v73.py).
    # NOTE: key comes first so that partial(..., 'feat') is a Pool.map callable.
    return _mat_v73.loadmat_key(path, key)


def _determine_num_parallel(num_files: int) -> int:
    # NOTE: optimal number of parallel processes is not clear. It could depend
    # on several factors such as the number of files, the size of files, the
    # number of cores, etc. For now, we use a simple heuristic based on the
    # number of files.
    num_parallel: int
    if num_files < 16:
        num_parallel = 1
    elif num_files < 64:
        num_parallel = 16
    else:
        num_parallel = 64
    return num_parallel


def _normalize_feature_slice(feature_slice: FeatureSlice) -> Tuple:
    """Normalize a feature-axis slice spec to a tuple of per-axis indexers."""
    if feature_slice is None:
        return ()
    if isinstance(feature_slice, tuple):
        return feature_slice
    return (feature_slice,)


def _is_fancy(indexer: object) -> bool:
    """Whether an indexer is a fancy (list/array) index rather than a slice."""
    if isinstance(indexer, (slice, int, np.integer)):
        return False
    return isinstance(indexer, (list, tuple, np.ndarray, range))


class FeatureStore(ABC):
    """Interface to a collection of DNN features on disk.

    A store exposes layers and stimulus labels, and reads features for a set of
    labels with an optional slice along the feature axes. Implementations differ
    only in how much I/O a given request costs.
    """

    @property
    @abstractmethod
    def layers(self) -> List[str]:
        """List of DNN layers held by this store."""

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """List of stimulus labels, in the store's own order."""

    @abstractmethod
    def shape(self, layer: str) -> Tuple[int, ...]:
        """Full shape ``(n_samples, *feature_shape)`` of `layer`."""

    @abstractmethod
    def dtype(self, layer: str) -> np.dtype:
        """Dtype of the features in `layer`."""

    @abstractmethod
    def read(
        self,
        layer: str,
        labels: Optional[Sequence[str]] = None,
        feature_slice: FeatureSlice = None,
    ) -> np.ndarray:
        """Read features from `layer`.

        Parameters
        ----------
        layer : str
            DNN layer.
        labels : sequence of str, optional
            Stimulus labels to read. ``None`` reads every label in store order.
            Otherwise rows come back **in the order given**, and repeated labels
            yield repeated rows.
        feature_slice : slice, int, array-like or tuple, optional
            Index applied to the feature axes (axes 1 and up), as produced by
            ``numpy.s_[...]``. ``None`` reads the whole feature tensor.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_labels, *sliced_feature_shape)``.
        """

    def chunk_extent(self, layer: str, axis: int) -> Optional[int]:
        """On-disk chunk extent of `layer` along `axis`, if the store has one.

        Returns ``None`` when the backend has no chunking to align with, in
        which case :meth:`iter_chunks` falls back to a byte budget.
        """
        return None

    def iter_chunks(
        self,
        layer: str,
        labels: Optional[Sequence[str]] = None,
        feature_slice: FeatureSlice = None,
        axis: int = 1,
        size: Optional[int] = None,
    ) -> Iterator[Tuple[slice, np.ndarray]]:
        """Iterate over `layer` in slabs along `axis`.

        Yields ``(sl, block)`` pairs, where `sl` is the slice applied to `axis`
        of the selected array and `block` is that slab. The slice is yielded so
        that callers can place results back without tracking offsets themselves.

        Parameters
        ----------
        layer : str
            DNN layer.
        labels : sequence of str, optional
            Stimulus labels, as in :meth:`read`.
        feature_slice : slice, int, array-like or tuple, optional
            Index applied to the feature axes before iterating.
        axis : int, optional
            Axis of the *selected* array to iterate over. Axis 0 is the sample
            axis; the default, axis 1, is the outermost feature axis, which is
            the one partial feature reads are about.
        size : int, optional
            Number of elements per slab. Defaults to the on-disk chunk extent
            along `axis` when the backend has one, so that iteration is
            chunk-aligned and every element is read exactly once; otherwise a
            size derived from the same byte budget is used.

        Yields
        ------
        tuple of (slice, numpy.ndarray)
            The slice applied to `axis`, and the corresponding slab.
        """
        selected_shape = self._selected_shape(layer, labels, feature_slice)
        if axis < 0:
            axis += len(selected_shape)
        if not 0 <= axis < len(selected_shape):
            raise ValueError(
                "axis {} is out of range for selected shape {}".format(
                    axis, selected_shape
                )
            )

        length = selected_shape[axis]
        if size is None:
            size = self._default_iter_size(layer, axis, selected_shape)
        if size < 1:
            raise ValueError("size must be positive, got {}".format(size))

        base = _normalize_feature_slice(feature_slice)
        for start in range(0, length, size):
            sl = slice(start, min(start + size, length))
            if axis == 0:
                block_labels = (
                    list(self.labels) if labels is None else list(labels)
                )[sl]
                yield sl, self.read(layer, block_labels, feature_slice)
            else:
                yield sl, self.read(
                    layer, labels, _compose_slice(base, axis - 1, sl)
                )

    def _selected_shape(
        self,
        layer: str,
        labels: Optional[Sequence[str]],
        feature_slice: FeatureSlice,
    ) -> Tuple[int, ...]:
        """Shape the selection would have, without reading the data."""
        full = self.shape(layer)
        n_samples = full[0] if labels is None else len(labels)
        feature_shape = _sliced_shape(full[1:], feature_slice)
        return (n_samples, *feature_shape)

    def _default_iter_size(
        self, layer: str, axis: int, selected_shape: Tuple[int, ...]
    ) -> int:
        extent = self.chunk_extent(layer, axis)
        if extent is not None:
            return max(1, min(extent, selected_shape[axis]))
        # No on-disk chunking to align with: fall back to the byte budget.
        itemsize = np.dtype(self.dtype(layer)).itemsize
        other = 1
        for i, n in enumerate(selected_shape):
            if i != axis:
                other *= int(n)
        per_element = max(1, other * itemsize)
        return max(1, min(selected_shape[axis], DEFAULT_TARGET_CHUNK_BYTES // per_element))


def _compose_slice(base: Tuple, axis: int, sl: slice) -> Tuple:
    """Return `base` with `sl` intersected into its entry for `axis`.

    `axis` indexes the *feature* axes (axis 0 here is array axis 1), and `sl`
    is relative to the result of applying `base`.
    """
    out = list(base) + [slice(None)] * max(0, axis + 1 - len(base))
    current = out[axis]
    if isinstance(current, slice) and current == slice(None):
        out[axis] = sl
    elif isinstance(current, slice):
        # Compose the two slices by indexing a range with both.
        out[axis] = _compose_two_slices(current, sl)
    else:
        out[axis] = np.asarray(current)[sl]
    return tuple(out)


def _compose_two_slices(outer: slice, inner: slice) -> slice:
    """Compose ``x[outer][inner]`` into a single slice where possible."""
    # Only the simple (positive-step) case arises from iter_chunks, which
    # always produces contiguous forward slices.
    ostep = 1 if outer.step is None else outer.step
    if ostep < 0:
        # Fall back to an explicit index array for reversed slices.
        raise ValueError("cannot compose a reversed slice; pass an explicit size")
    ostart = 0 if outer.start is None else outer.start
    start = ostart + (0 if inner.start is None else inner.start) * ostep
    stop = ostart + (inner.stop if inner.stop is not None else 0) * ostep
    if outer.stop is not None:
        stop = min(stop, outer.stop)
    return slice(start, stop, ostep)


def _sliced_shape(
    feature_shape: Tuple[int, ...], feature_slice: FeatureSlice
) -> Tuple[int, ...]:
    """Shape of ``numpy.empty(feature_shape)[feature_slice]``, cheaply."""
    indexers = _normalize_feature_slice(feature_slice)
    if not indexers:
        return feature_shape
    # A broadcast dummy allocates nothing but lets NumPy do the index
    # arithmetic, including negative steps, ellipses and newaxis.
    dummy = np.broadcast_to(np.int8(0), feature_shape)
    return tuple(int(n) for n in dummy[indexers].shape)


class MatFeatureStore(FeatureStore):
    """Legacy per-stimulus ``.mat`` feature directory.

    Layout is ``<dpath>/<layer>/<label>.<ext>``, each file holding one sample
    under `key` with a leading sample axis. There is no file boundary on the
    feature axes, so `feature_slice` is applied after the full stimulus files
    have been loaded and concatenated -- the read cost is the same as before,
    and the slice only saves the caller from doing it themselves.

    Parameters
    ----------
    dpath : str
        Feature directory.
    ext : str, optional
        Feature file extension (default: ``'mat'``).
    key : str, optional
        Variable name inside each file (default: ``'feat'``).
    """

    def __init__(self, dpath: str, ext: str = "mat", key: str = "feat"):
        self._dpath = dpath
        self._ext = ext
        self._key = key
        self._layers = self._collect_layers()
        self._labels = self._collect_labels()
        self._file_table: Dict[str, Dict[str, str]] = {
            layer: {
                label: os.path.join(dpath, layer, label + "." + ext)
                for label in self._labels
            }
            for layer in self._layers
        }

    @property
    def layers(self) -> List[str]:
        return self._layers

    @property
    def labels(self) -> List[str]:
        return self._labels

    def path(self, layer: str, label: str) -> str:
        """Path of the file holding `label` in `layer`."""
        return self._file_table[layer][label]

    def shape(self, layer: str) -> Tuple[int, ...]:
        if not self._labels:
            raise RuntimeError("No features found in {}".format(self._dpath))
        sample = _load_array_with_key(self._key, self.path(layer, self._labels[0]))
        return (len(self._labels), *sample.shape[1:])

    def dtype(self, layer: str) -> np.dtype:
        if not self._labels:
            raise RuntimeError("No features found in {}".format(self._dpath))
        sample = _load_array_with_key(self._key, self.path(layer, self._labels[0]))
        return sample.dtype

    def read(
        self,
        layer: str,
        labels: Optional[Sequence[str]] = None,
        feature_slice: FeatureSlice = None,
    ) -> np.ndarray:
        if labels is None:
            labels = self._labels
        paths = [self.path(layer, label) for label in labels]

        num_parallel = _determine_num_parallel(len(paths))
        load = partial(_load_array_with_key, self._key)
        if num_parallel == 1:
            arrays = list(map(load, paths))
        else:
            with Pool(processes=num_parallel) as pool:
                arrays = pool.map(load, paths)
        features = np.concatenate(arrays, axis=0)

        indexers = _normalize_feature_slice(feature_slice)
        if indexers:
            features = features[(slice(None), *indexers)]
        return features

    def _collect_layers(self) -> List[str]:
        return sorted(
            d
            for d in os.listdir(self._dpath)
            if os.path.isdir(os.path.join(self._dpath, d))
        )

    def _collect_labels(self) -> List[str]:
        labels: List[str] = []
        for layer in self._layers:
            layer_dir = os.path.join(self._dpath, layer)
            layer_dir = layer_dir.replace("[", "[[]")  # Use glob.escape for Python 3.4 or later
            files = glob.glob(os.path.join(layer_dir, "*." + self._ext))
            labels_t = sorted(
                os.path.splitext(os.path.basename(f))[0] for f in files
            )
            if not labels:
                labels = labels_t
            elif labels != labels_t:
                raise RuntimeError("Invalid feature file in %s " % self._dpath)
        return labels


class HDF5FeatureStore(FeatureStore):
    """Chunked HDF5 feature directory, one ``<layer>.h5`` file per layer.

    Each file holds ``/features`` with shape ``(n_samples, *feature_shape)`` and
    ``/labels`` with the matching stimulus labels. ``/features`` is explicitly
    chunked, so a slice along the feature axes reads only the chunks it covers
    instead of the whole layer.

    Parameters
    ----------
    dpath : str
        Directory holding ``<layer>.h5`` files.
    """

    def __init__(self, dpath: str):
        self._dpath = dpath
        self._layers = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(glob.escape(dpath), "*." + HDF5_EXT))
        )
        if not self._layers:
            raise RuntimeError("No .{} feature file found in {}".format(HDF5_EXT, dpath))
        self._labels: Optional[List[str]] = None
        self._label_index: Optional[Dict[str, int]] = None
        self._validate_all()

    @property
    def layers(self) -> List[str]:
        return self._layers

    @property
    def labels(self) -> List[str]:
        if self._labels is None:
            with self._open(self._layers[0]) as f:
                self._labels = _decode_labels(f[LABELS_DATASET][()])
            self._label_index = {
                label: i for i, label in enumerate(self._labels)
            }
        return self._labels

    def path(self, layer: str) -> str:
        """Path of the file holding `layer`."""
        return os.path.join(self._dpath, layer + "." + HDF5_EXT)

    def shape(self, layer: str) -> Tuple[int, ...]:
        with self._open(layer) as f:
            return tuple(int(n) for n in f[FEATURES_DATASET].shape)

    def dtype(self, layer: str) -> np.dtype:
        with self._open(layer) as f:
            dtype: np.dtype = np.dtype(f[FEATURES_DATASET].dtype)
        return dtype

    def chunk_extent(self, layer: str, axis: int) -> Optional[int]:
        with self._open(layer) as f:
            chunks = f[FEATURES_DATASET].chunks
        if chunks is None or axis >= len(chunks):
            return None
        return int(chunks[axis])

    def read(
        self,
        layer: str,
        labels: Optional[Sequence[str]] = None,
        feature_slice: FeatureSlice = None,
    ) -> np.ndarray:
        indexers = _normalize_feature_slice(feature_slice)

        with self._open(layer) as f:
            dset = f[FEATURES_DATASET]

            if labels is None:
                return self._read_rows(dset, slice(None), indexers)

            rows = self._row_indices(labels)
            # h5py wants a strictly increasing index list and allows at most one
            # fancy index per selection. Read each distinct row once in order,
            # then restore the caller's order (and any repeats) with `inverse`.
            uniq, inverse = np.unique(rows, return_inverse=True)
            if uniq.size == len(rows) and np.array_equal(uniq, rows):
                return self._read_rows(dset, list(uniq), indexers)
            block = self._read_rows(dset, list(uniq), indexers)
            return block[inverse]

    @staticmethod
    def _read_rows(
        dset: h5py.Dataset, rows: Union[slice, List[int]], indexers: Tuple
    ) -> np.ndarray:
        """Read `rows` from `dset`, pushing as much of `indexers` into h5py as it allows."""
        if not indexers:
            return np.asarray(dset[rows])
        if any(_is_fancy(ix) for ix in indexers):
            # A fancy row list plus a fancy feature index would be two fancy
            # indices in one selection, which h5py rejects. Read the feature
            # axes whole and apply the fancy part in NumPy.
            data = np.asarray(dset[rows])
            return data[(slice(None), *indexers)]
        # Plain slices are not fancy, so they ride along in the same selection
        # and h5py reads only the chunks they cover.
        return np.asarray(dset[(rows, *indexers)])

    def _row_indices(self, labels: Sequence[str]) -> np.ndarray:
        _ = self.labels  # populate the label -> row index map
        assert self._label_index is not None
        try:
            return np.array([self._label_index[label] for label in labels], dtype=int)
        except KeyError as exc:
            raise KeyError(
                "Label {} not found in {}".format(exc.args[0], self._dpath)
            ) from None

    def _open(self, layer: str) -> h5py.File:
        return h5py.File(self.path(layer), "r")

    def _validate_all(self) -> None:
        for layer in self._layers:
            with self._open(layer) as f:
                _validate_format(f, self.path(layer))


def _validate_format(f: h5py.File, path: str) -> None:
    """Check that `f` is bdpy feature storage of a version we can read."""
    fmt = f.attrs.get(FORMAT_ATTR)
    if isinstance(fmt, bytes):
        fmt = fmt.decode("utf-8")
    if fmt != FORMAT_NAME:
        raise RuntimeError(
            "{} is not bdpy feature storage ({} = {!r}); expected {!r}".format(
                path, FORMAT_ATTR, fmt, FORMAT_NAME
            )
        )
    version = int(f.attrs.get(FORMAT_VERSION_ATTR, 0))
    if version > SUPPORTED_FORMAT_VERSION:
        raise RuntimeError(
            "{} uses feature storage format version {}, but this version of "
            "bdpy supports up to version {}. Please upgrade bdpy.".format(
                path, version, SUPPORTED_FORMAT_VERSION
            )
        )
    for name in (FEATURES_DATASET, LABELS_DATASET):
        if name not in f:
            raise RuntimeError("{} has no /{} dataset".format(path, name))


def _decode_labels(raw: np.ndarray) -> List[str]:
    """Decode an HDF5 string dataset into a list of str (h5py 3.x gives bytes)."""
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(raw).ravel().tolist()
    ]


def detect_format(dpath: str, ext: str = "mat") -> str:
    """Guess which storage format `dpath` holds.

    Returns ``'hdf5'`` when the directory holds ``<layer>.h5`` files and no
    layer subdirectories, and ``'mat'`` otherwise. A directory holding both is
    read as ``'mat'``, the historical layout; pass an explicit format to
    override.

    Parameters
    ----------
    dpath : str
        Feature directory.
    ext : str, optional
        Extension of the legacy per-stimulus files (default: ``'mat'``).

    Returns
    -------
    str
        ``'hdf5'`` or ``'mat'``.
    """
    escaped = glob.escape(dpath)
    has_subdirs = any(
        os.path.isdir(os.path.join(dpath, d)) for d in os.listdir(dpath)
    )
    has_h5 = bool(glob.glob(os.path.join(escaped, "*." + HDF5_EXT)))
    if has_h5 and not has_subdirs:
        return "hdf5"
    return "mat"
