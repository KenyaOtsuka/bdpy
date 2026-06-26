"""Sparse array class.

This file is a part of bdpy.
"""

__all__ = ['SparseArray', 'load_array', 'save_array', 'save_multiarrays']


import os

import h5py
import numpy as np

from . import _mat_v73


def load_array(fname, key='data'):
    """Load an array (dense or sparse)."""
    with h5py.File(fname, 'r') as f:
        obj = f[key]
        # Inspect the HDF5 object type rather than reading the whole dataset:
        # a SparseArray is stored as a group, a dense array as a dataset.
        if isinstance(obj, h5py.Group):
            if '__bdpy_sparse_arrray' in obj:
                s_ary = SparseArray(fname, key=key)
                return s_ary.dense
            raise RuntimeError('Unsupported group: %s' % key)
        if isinstance(obj, h5py.Dataset):
            # Dense array (read with h5py; the legacy MAT-v7.3 library breaks under NumPy 2.0)
            return _mat_v73.read_dataset(obj)
        raise RuntimeError('Unsupported data type: %s' % type(obj))


def save_array(fname, array, key='data', dtype=np.float64, sparse=False):
    """Save an array (dense or sparse)."""
    if sparse:
        # Save as a SparseArray
        s_ary = SparseArray(array.astype(dtype))
        s_ary.save(fname, key=key, dtype=dtype)
    else:
        # Save as a dense array (h5py MATLAB v7.3 writer)
        _mat_v73.savemat(fname, {key: array.astype(dtype)})

    return None


def save_multiarrays(fname, arrays):
    """Save arrays (dense)."""
    save_dict = {k: v for k, v in arrays.items()}
    _mat_v73.savemat(fname, save_dict)

    return None


class SparseArray(object):
    """Sparse array class."""
    
    def __init__(self, src=None, key='data', background=0):
        self.__background = background

        if type(src) == np.ndarray:
            # Create sparse array from numpy.ndarray
            self.__make_sparse(src)
        elif os.path.isfile(src):
            # Load data from src
            self.__load(src, key=key)
        else:
            raise ValueError('Unsupported input')

    @property
    def dense(self):
        return self.__make_dense()

    def save(self, fname, key='data', dtype=np.float64):
        # Write the sparse-array struct through _mat_v73.savemat's dict/struct
        # writer, so dense and sparse share the same MATLAB-v7.3-compatible
        # writer (group with MATLAB_class='struct' + MATLAB_fields, each field
        # written in MATLAB layout). Append to an existing file (replacing only
        # ``key``) so other top-level variables are preserved; a brand-new file
        # gets the MAT-file userblock header from savemat.
        index = np.vstack([np.asarray(i, dtype=np.int64).ravel()
                           for i in self.__index])
        payload = {
            u'__bdpy_sparse_arrray': np.array(True),
            u'index': index,
            u'value': self.__value.astype(dtype).ravel(),
            u'shape': np.asarray(self.__shape, dtype=np.int64),
            u'background': np.asarray(self.__background),
        }
        mode = 'a' if os.path.exists(fname) else 'w'
        _mat_v73.savemat(fname, {key: payload}, mode=mode)
        return None

    def __make_sparse(self, array):
        self.__index = np.where(array != self.__background)
        self.__value = array[self.__index]
        self.__shape = array.shape
        return None

    def __make_dense(self):
        dense = np.ones(self.__shape) * self.__background
        dense[self.__index] = self.__value
        return dense

    def __load(self, fname, key='data'):
        # Read with h5py instead of the legacy MAT-v7.3 library (NumPy 2.0).
        # read_cell handles all writer flavors of ``index``/``shape``: legacy
        # cell arrays (object refs), and plain matrices written either by other
        # tools (no MATLAB_class) or by bdpy's MATLAB-style writer (MATLAB_class
        # set, transposed -- un-transposed by read_cell).
        with h5py.File(fname, 'r') as f:
            g = f[key]
            self.__index = tuple(
                np.asarray(c).ravel().astype(int)
                for c in _mat_v73.read_cell(f, g['index'])
            )

            value = np.asarray(_mat_v73.read_dataset(g['value'])).ravel()
            self.__value = value

            self.__shape = tuple(
                int(np.asarray(s).ravel()[0])
                for s in _mat_v73.read_cell(f, g['shape'])
            )

            background = np.asarray(_mat_v73.read_dataset(g['background'])).ravel()
            self.__background = background[0] if background.size else 0
        return None
