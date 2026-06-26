"""Read and write MATLAB v7.3 (HDF5) ``.mat`` files with h5py.

bdpy historically relied on a third-party MATLAB-v7.3 library for both the load
and the save paths, but that library broke under NumPy 2.0 (it referenced the
removed ``np.unicode_``; see issue #106) and added an extra dependency. This
module reimplements the *read* and *write* sides on top of h5py for the data
layouts bdpy actually uses (dense numeric arrays and the sparse-array struct),
so that dependency is no longer required.

Files written here follow the MATLAB v7.3 HDF5 layout (column-major / reversed
dimension order with ``MATLAB_class`` attributes) and carry a MAT-file
userblock header, so they round-trip through :func:`read_dataset` and are
intended to be readable by MATLAB's ``load``. ``Python.Shape`` is additionally
stored so the original NumPy rank/shape is restored on read; this is
bdpy/Python-side metadata for round-tripping (and matches the legacy file
layout), not something MATLAB requires.

This file is a part of BdPy.
"""

import datetime
import platform
from typing import Optional, Tuple

import h5py
import numpy as np
import scipy.io as sio

__all__ = [
    "load_array",
    "loadmat_key",
    "read_cell",
    "read_dataset",
    "savemat",
    "write_dataset",
]

# NumPy dtype name -> (MATLAB_class, MATLAB_int_decode or None, on-disk dtype).
# Only the numeric types bdpy writes are supported; anything else raises in
# ``_matlab_class``. ``bool`` is handled separately (stored as uint8 logical).
_NUMPY_TO_MATLAB = {
    "float64": (b"double", None, np.float64),
    "float32": (b"single", None, np.float32),
    "int8": (b"int8", 1, np.int8),
    "uint8": (b"uint8", 1, np.uint8),
    "int16": (b"int16", 2, np.int16),
    "uint16": (b"uint16", 2, np.uint16),
    "int32": (b"int32", 4, np.int32),
    "uint32": (b"uint32", 4, np.uint32),
    "int64": (b"int64", 8, np.int64),
    "uint64": (b"uint64", 8, np.uint64),
}


def _matlab_class_str(attrs: "h5py.AttributeManager") -> Optional[str]:
    """Return the ``MATLAB_class`` attribute as a ``str`` (or ``None``)."""
    value = attrs.get("MATLAB_class")
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("ascii", "ignore")
    return str(value)


def read_dataset(dset: h5py.Dataset) -> np.ndarray:
    """Read an h5py dataset, undoing MATLAB v7.3 conventions.

    MATLAB stores arrays in Fortran (column-major) order, so multi-dimensional
    datasets are written transposed relative to NumPy's C order. bdpy (and the
    legacy writer that produced existing files) additionally records the
    original Python shape and empty-array flags as ``Python.*`` attributes,
    which we honor to restore the original NumPy array.
    Only ``Python.Empty`` is special-cased; a bare ``MATLAB_empty`` (set by
    MATLAB without ``Python.Shape``) is read through the normal path so that
    empty non-scalar arrays such as ``(0, 3)`` keep their shape. MATLAB
    ``logical`` arrays are stored as uint8 (0/1) and restored to ``bool`` here.

    Parameters
    ----------
    dset : h5py.Dataset
        Dataset to read.

    Returns
    -------
    numpy.ndarray
        The array with its original shape (and bool dtype, for logicals)
        restored.
    """
    attrs = dset.attrs
    if "Python.Empty" in attrs:
        # Only Python.Empty (written by bdpy / the legacy writer) implies a
        # Python.Shape we can trust; fall back to the stored dataset shape if
        # it is missing. A bare MATLAB_empty (written by MATLAB without
        # Python.Shape) must NOT be treated this way -- np.empty(()) would
        # collapse e.g. (0, 3) to 0-d -- so it falls through to the normal
        # read/transpose path below.
        shape = tuple(int(x) for x in attrs.get("Python.Shape", dset.shape))
        arr = np.empty(shape, dtype=dset.dtype)
    else:
        arr = dset[()]
        if "MATLAB_class" in attrs and isinstance(arr, np.ndarray) and arr.ndim >= 2:
            arr = np.transpose(arr)
        if "Python.Shape" in attrs:
            arr = np.asarray(arr).reshape(tuple(int(x) for x in attrs["Python.Shape"]))
        arr = np.asarray(arr)
    if _matlab_class_str(attrs) == "logical":
        arr = arr.astype(bool)
    return arr


def read_cell(f: h5py.File, dset: h5py.Dataset) -> list:
    """Read a MATLAB cell array (or a plain matrix) into a list of arrays.

    Python tuples/lists are stored as MATLAB cell arrays, i.e. an object
    dataset of HDF5 references to the individual elements. Files written by other
    tools (e.g. MATLAB or Julia) may instead store the same information as a plain
    2-D matrix whose rows are the elements; both layouts are handled here.

    Parameters
    ----------
    f : h5py.File
        Open file, used to dereference cell-array element references.
    dset : h5py.Dataset
        The cell (object) dataset or a plain matrix.

    Returns
    -------
    list of numpy.ndarray
        One array per cell element / matrix row.
    """
    data = dset[()]
    if isinstance(data, np.ndarray) and data.dtype == object:
        return [read_dataset(f[ref]) for ref in data.ravel()]
    arr = np.asarray(data)
    return [arr[i] for i in range(arr.shape[0])]


def load_array(path: str, key: str) -> np.ndarray:
    """Load a single dense numeric array from a v7.3 ``.mat`` file.

    Parameters
    ----------
    path : str
        Path to the ``.mat`` file.
    key : str
        Variable name to load.

    Returns
    -------
    numpy.ndarray
        The loaded array.
    """
    with h5py.File(path, "r") as f:
        return read_dataset(f[key])


def loadmat_key(path: str, key: str) -> np.ndarray:
    """Load one variable from a ``.mat`` file, handling both v5 and v7.3.

    MATLAB v5 (and earlier) files are read with :func:`scipy.io.loadmat`; v7.3
    (HDF5) files, which scipy cannot read, fall back to the h5py reader. This
    preserves v5 support while avoiding the legacy MATLAB-v7.3 library on the
    load path (which breaks under NumPy 2.0).

    Parameters
    ----------
    path : str
        Path to the ``.mat`` file.
    key : str
        Variable name to load.

    Returns
    -------
    numpy.ndarray
        The loaded array.
    """
    try:
        array = sio.loadmat(path)[key]
    except (NotImplementedError, ValueError):
        return load_array(path, key)
    return np.asarray(array)


def _matlab_class(array: np.ndarray) -> Tuple[bytes, Optional[int], type]:
    """Return ``(MATLAB_class, MATLAB_int_decode, on_disk_dtype)`` for ``array``.

    Raises
    ------
    TypeError
        If ``array`` has a dtype the MATLAB v7.3 writer does not support
        (e.g. complex, object, or string/unicode).
    """
    if array.dtype == np.bool_:
        return b"logical", 1, np.uint8
    name = array.dtype.name
    if name not in _NUMPY_TO_MATLAB:
        raise TypeError(
            "Unsupported dtype for the MATLAB v7.3 writer: %s" % name)
    return _NUMPY_TO_MATLAB[name]


def write_dataset(group: h5py.Group, key: str, array: np.ndarray) -> h5py.Dataset:
    """Write one numeric array into ``group`` under ``key`` (MATLAB v7.3).

    The array is stored transposed (column-major / reversed dimension order) so
    MATLAB reads back the original shape, with ``MATLAB_class`` describing the
    type. The original NumPy shape is recorded in ``Python.Shape`` so
    :func:`read_dataset` restores the exact rank (1-D arrays are stored as
    MATLAB column vectors, scalars as 1x1).

    Parameters
    ----------
    group : h5py.Group
        Group (or file) to create the dataset in.
    key : str
        Variable name. Must be non-empty and must not contain ``'/'`` (which
        h5py would interpret as a nested group path).
    array : array_like
        The array to store. Complex/object/string dtypes are unsupported.

    Returns
    -------
    h5py.Dataset
        The created dataset.
    """
    if not key or "/" in key:
        raise ValueError(
            "Invalid key for the MATLAB v7.3 writer: %r" % (key,))
    array = np.asarray(array)
    matlab_class, int_decode, on_disk_dtype = _matlab_class(array)
    orig_shape = array.shape

    if array.size == 0:
        # Empty array: store a placeholder and record the shape so the reader
        # can rebuild it via np.empty(Python.Shape, dtype=dset.dtype).
        dset = group.create_dataset(key, shape=(0,), dtype=on_disk_dtype)
        dset.attrs["MATLAB_class"] = np.bytes_(matlab_class)
        dset.attrs["MATLAB_empty"] = np.uint8(1)
        dset.attrs["Python.Empty"] = np.uint8(1)
        dset.attrs["Python.Shape"] = np.array(orig_shape, dtype=np.uint64)
        if int_decode is not None:
            dset.attrs["MATLAB_int_decode"] = np.int32(int_decode)
        return dset

    a = array.astype(on_disk_dtype, copy=False)
    if a.ndim == 0:
        a = a.reshape(1, 1)  # MATLAB has no scalars; store as 1x1
    elif a.ndim == 1:
        a = a.reshape(a.shape[0], 1)  # oned_as='column'
    on_disk = np.ascontiguousarray(np.transpose(a))

    dset = group.create_dataset(key, data=on_disk)
    dset.attrs["MATLAB_class"] = np.bytes_(matlab_class)
    if int_decode is not None:
        dset.attrs["MATLAB_int_decode"] = np.int32(int_decode)
    dset.attrs["Python.Shape"] = np.array(orig_shape, dtype=np.uint64)
    return dset


def _matfile_header() -> bytes:
    """Return the 128-byte MATLAB v7.3 MAT-file userblock header.

    Layout: 116 bytes of (space-padded) descriptive text, 8 bytes of subsystem
    data offset (none here), a 2-byte version field (0x0200) and a 2-byte
    endian indicator (``b'IM'`` for little-endian).
    """
    description = (
        "MATLAB 7.3 MAT-file, Platform: %s, "
        "Created on: %s HDF5 schema 1.00 ."
        % (platform.system(),
           datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y"))
    )
    header = bytearray(b" " * 116)
    text = description.encode("ascii", "replace")[:116]
    header[: len(text)] = text
    header += b"\x00" * 8       # subsystem data offset (none)
    header += b"\x00\x02"       # version 0x0200
    header += b"IM"             # endian indicator (little-endian)
    return bytes(header)


def savemat(fname: str, mdict: dict, mode: str = "w") -> None:
    """Write variables to a MATLAB v7.3 (HDF5) ``.mat`` file.

    Each value in ``mdict`` is stored as a top-level dense numeric array via
    :func:`write_dataset`. When ``mode == 'w'`` (the default) a fresh file is
    created with a MATLAB v7.3 MAT-file userblock header so that MATLAB
    recognizes it; with ``mode == 'a'`` the existing file (and its header) is
    kept and only the listed keys are written, replacing any of the same name.

    Parameters
    ----------
    fname : str
        Output path.
    mdict : dict
        Mapping of variable name to array.
    mode : str, optional
        h5py file mode (``'w'`` to create, ``'a'`` to append/replace keys).
    """
    write_header = mode == "w"
    if write_header:
        f = h5py.File(fname, "w", userblock_size=512)
    else:
        f = h5py.File(fname, mode)
    try:
        for key, value in mdict.items():
            if key in f:
                del f[key]
            write_dataset(f, key, np.asarray(value))
    finally:
        f.close()
    if write_header:
        with open(fname, "r+b") as fh:
            fh.write(_matfile_header())
    return None
