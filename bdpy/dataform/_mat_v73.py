"""Read MATLAB v7.3 (HDF5) ``.mat`` files with h5py.

bdpy historically relied on :func:`hdf5storage.loadmat` for the load path, but
that broke under NumPy 2.0, which removed ``np.unicode_`` that older hdf5storage
referenced (see issue #106). This module reimplements the *read* side on top of
h5py for the data layouts bdpy actually uses (dense numeric arrays and the
sparse-array struct). Saving still goes through hdf5storage.

This file is a part of BdPy.
"""

from typing import Dict, List

import h5py
import numpy as np
import scipy.io as sio

__all__ = [
    "load_array",
    "load_struct",
    "loadmat_key",
    "read_cell",
    "read_dataset",
]


def read_dataset(dset: h5py.Dataset) -> np.ndarray:
    """Read an h5py dataset, undoing MATLAB v7.3 / hdf5storage conventions.

    MATLAB stores arrays in Fortran (column-major) order, so multi-dimensional
    datasets are written transposed relative to NumPy's C order. hdf5storage
    additionally records the original Python shape and empty-array flags as
    ``Python.*`` attributes, which we honor to reproduce ``hdf5storage.loadmat``.
    Only ``Python.Empty`` is special-cased; a bare ``MATLAB_empty`` (set by
    MATLAB without ``Python.Shape``) is read through the normal path so that
    empty non-scalar arrays such as ``(0, 3)`` keep their shape.

    Parameters
    ----------
    dset : h5py.Dataset
        Dataset to read.

    Returns
    -------
    numpy.ndarray
        The array with its original shape restored.
    """
    attrs = dset.attrs
    if "Python.Empty" in attrs:
        # Only Python.Empty (written by hdf5storage) implies a Python.Shape we
        # can trust; fall back to the stored dataset shape if it is missing. A
        # bare MATLAB_empty (written by MATLAB without Python.Shape) must NOT be
        # treated this way -- np.empty(()) would collapse e.g. (0, 3) to 0-d --
        # so it falls through to the normal read/transpose path below.
        shape = tuple(int(x) for x in attrs.get("Python.Shape", dset.shape))
        return np.empty(shape, dtype=dset.dtype)
    arr = dset[()]
    if "MATLAB_class" in attrs and isinstance(arr, np.ndarray) and arr.ndim >= 2:
        arr = np.transpose(arr)
    if "Python.Shape" in attrs:
        arr = np.asarray(arr).reshape(tuple(int(x) for x in attrs["Python.Shape"]))
    return np.asarray(arr)


def read_cell(f: h5py.File, dset: h5py.Dataset) -> list:
    """Read a MATLAB cell array (or a plain matrix) into a list of arrays.

    hdf5storage stores Python tuples/lists as MATLAB cell arrays, i.e. an object
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
    preserves v5 support while avoiding hdf5storage on the load path (which
    breaks under NumPy 2.0).

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


def _struct_field_names(group: h5py.Group) -> List[str]:
    """Return the field names of a struct group, in their original order.

    hdf5storage records the order the fields were written in as the
    ``Python.Fields`` attribute; h5py itself iterates members alphabetically, so
    without it the original order is lost. Files written by other tools (e.g.
    MATLAB) carry no such attribute and fall back to the file's member order.
    """
    fields = group.attrs.get("Python.Fields")
    if fields is None:
        return list(group.keys())
    return [
        name.decode() if isinstance(name, bytes) else str(name) for name in fields
    ]


def _load_struct_v73(path: str, key: str) -> Dict[str, np.ndarray]:
    """Read a struct variable from a v7.3 ``.mat`` file with h5py."""
    with h5py.File(path, "r") as f:
        group = f[key]
        if not isinstance(group, h5py.Group):
            raise TypeError(
                "'%s' in %s is not a struct (got %s)"
                % (key, path, type(group).__name__)
            )
        struct = {}
        for name in _struct_field_names(group):
            member = group[name]
            if not isinstance(member, h5py.Dataset):
                raise TypeError(
                    "field '%s' of '%s' in %s is not an array (got %s); nested "
                    "structs are not supported"
                    % (name, key, path, type(member).__name__)
                )
            struct[name] = read_dataset(member)
        return struct


def _unwrap_object_array(value: np.ndarray) -> np.ndarray:
    """Peel the object-array nesting scipy wraps v5 struct fields in.

    ``scipy.io.loadmat`` returns each field of a struct as a size-1 object array
    holding the actual data, sometimes nested more than once, so unwrap until a
    real array is reached.
    """
    array = np.asarray(value)
    while array.dtype == object and array.size == 1:
        array = np.asarray(array.item())
    return array


def _struct_from_v5(variable: np.ndarray, path: str, key: str) -> Dict[str, np.ndarray]:
    """Convert the structured array scipy returns for a v5 struct to a dict."""
    array = np.asarray(variable)
    names = array.dtype.names
    if names is None:
        raise TypeError("'%s' in %s is not a struct" % (key, path))
    return {name: _unwrap_object_array(array[name]) for name in names}


def load_struct(path: str, key: str) -> Dict[str, np.ndarray]:
    """Load a MATLAB struct / Python dict variable as a dict of arrays.

    :func:`loadmat_key` handles dense numeric arrays only; a struct is stored as
    an ``h5py.Group`` (v7.3) or a structured array (v5), neither of which it can
    read. This is the struct counterpart, used for e.g. the per-layer unit index
    of :class:`bdpy.dataform.Features`. It reproduces what
    ``hdf5storage.loadmat`` used to return before hdf5storage was dropped from
    the load path (it breaks under NumPy 2.0, see issue #106).

    The field arrays are returned exactly as stored. In particular, a struct
    written by MATLAB carries no ``Python.Shape``, so :func:`read_dataset`
    applies the MATLAB transpose and a stored row vector comes back as a 2-D
    ``(1, n)`` array; callers that need a flat index must ravel it themselves.

    Parameters
    ----------
    path : str
        Path to the ``.mat`` file.
    key : str
        Variable name to load.

    Returns
    -------
    dict of str to numpy.ndarray
        One array per struct field, keyed by field name.

    Raises
    ------
    TypeError
        If ``key`` is not a struct, or has a field that is not an array.
    """
    try:
        variable = sio.loadmat(path)[key]
    except (NotImplementedError, ValueError):
        return _load_struct_v73(path, key)
    return _struct_from_v5(variable, path, key)
