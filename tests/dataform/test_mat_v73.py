"""Tests for bdpy.dataform._mat_v73."""

import os
import tempfile
import unittest

import h5py
import hdf5storage
import numpy as np
import scipy.io as sio
from numpy.testing import assert_array_equal

from bdpy.dataform import _mat_v73


class TestReadDataset(unittest.TestCase):

    def test_matlab_empty_without_python_shape_preserves_shape(self):
        # MATLAB-written v7.3 files mark empty arrays with ``MATLAB_empty`` but
        # do not store ``Python.Shape``. Such a dataset must not be collapsed to
        # a 0-d array (the bug fixed here treated MATLAB_empty like Python.Empty
        # and returned ``np.empty(())``).
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'matlab_empty.mat')

            # On-disk shape (0, 3); with MATLAB_class the read path transposes
            # it to (3, 0) like any other MATLAB matrix.
            with h5py.File(fname, 'w') as f:
                dset = f.create_dataset('a', shape=(0, 3), dtype='float64')
                dset.attrs['MATLAB_empty'] = np.uint8(1)
                dset.attrs['MATLAB_class'] = np.bytes_(b'double')
                self.assertNotIn('Python.Shape', dset.attrs)

            with h5py.File(fname, 'r') as f:
                out = _mat_v73.read_dataset(f['a'])

            self.assertNotEqual(out.shape, ())  # would fail with the old code
            self.assertEqual(out.ndim, 2)
            self.assertEqual(out.size, 0)
            self.assertEqual(out.shape, (3, 0))

    def test_matlab_empty_without_matlab_class_keeps_on_disk_shape(self):
        # Without MATLAB_class there is no transpose, so the empty non-scalar
        # shape is returned as stored.
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'matlab_empty_no_class.mat')

            with h5py.File(fname, 'w') as f:
                dset = f.create_dataset('a', shape=(3, 0), dtype='float64')
                dset.attrs['MATLAB_empty'] = np.uint8(1)
                self.assertNotIn('Python.Shape', dset.attrs)

            with h5py.File(fname, 'r') as f:
                out = _mat_v73.read_dataset(f['a'])

            self.assertNotEqual(out.shape, ())
            self.assertEqual(out.shape, (3, 0))
            self.assertEqual(out.size, 0)


class TestLoadStruct(unittest.TestCase):
    """The struct load path (``loadmat_key`` handles dense arrays only)."""

    def test_v73_struct_roundtrips_as_a_dict_of_arrays(self):
        # hdf5storage stores a dict as a struct group whose members are the
        # fields, and records the original field order in Python.Fields.
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'index.mat')
            hdf5storage.savemat(
                fname,
                {'index': {'fc8': np.array([0, 5, 11, 19]),
                           'conv5': np.array([2, 7])}},
                format='7.3',
                store_python_metadata=True)

            out = _mat_v73.load_struct(fname, 'index')

            self.assertEqual(list(out.keys()), ['fc8', 'conv5'])  # written order
            assert_array_equal(out['fc8'], np.array([0, 5, 11, 19]))
            assert_array_equal(out['conv5'], np.array([2, 7]))

    def test_v73_struct_without_python_metadata_keeps_matlab_shape(self):
        # A struct written by MATLAB (or by hdf5storage without Python
        # metadata) has no Python.Shape, so read_dataset applies the MATLAB
        # transpose and a stored column comes back as a 2-D (1, n) row. The
        # loader returns it as stored; Features ravels it before indexing.
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'index_matlab.mat')

            with h5py.File(fname, 'w') as f:
                group = f.create_group('index')
                group.attrs['MATLAB_class'] = np.bytes_(b'struct')
                dset = group.create_dataset(
                    'fc8', data=np.array([[0], [5], [11], [19]], dtype=np.int64))
                dset.attrs['MATLAB_class'] = np.bytes_(b'int64')
                self.assertNotIn('Python.Shape', dset.attrs)

            out = _mat_v73.load_struct(fname, 'index')

            self.assertEqual(out['fc8'].shape, (1, 4))
            assert_array_equal(out['fc8'].ravel(), np.array([0, 5, 11, 19]))

    def test_v5_struct_roundtrips_as_a_dict_of_arrays(self):
        # scipy returns a v5 struct as a (1, 1) structured array whose fields
        # are size-1 object arrays holding the data.
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'index_v5.mat')
            sio.savemat(
                fname,
                {'index': {'fc8': np.array([0, 5, 11, 19]),
                           'conv5': np.array([2, 7])}},
                format='5')

            out = _mat_v73.load_struct(fname, 'index')

            self.assertEqual(sorted(out.keys()), ['conv5', 'fc8'])
            assert_array_equal(out['fc8'].ravel(), np.array([0, 5, 11, 19]))
            assert_array_equal(out['conv5'].ravel(), np.array([2, 7]))

    def test_v73_dense_array_raises_type_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'dense.mat')
            hdf5storage.savemat(
                fname, {'index': np.arange(4)},
                format='7.3', store_python_metadata=True)

            with self.assertRaises(TypeError):
                _mat_v73.load_struct(fname, 'index')

    def test_v5_dense_array_raises_type_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'dense_v5.mat')
            sio.savemat(fname, {'index': np.arange(4)}, format='5')

            with self.assertRaises(TypeError):
                _mat_v73.load_struct(fname, 'index')

    def test_dense_array_load_path_is_unchanged(self):
        # loadmat_key must keep returning a bare array for dense variables.
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'dense.mat')
            data = np.arange(6).reshape(2, 3).astype(np.float64)
            hdf5storage.savemat(
                fname, {'feat': data}, format='7.3', store_python_metadata=True)

            assert_array_equal(_mat_v73.loadmat_key(fname, 'feat'), data)


if __name__ == '__main__':
    unittest.main()
