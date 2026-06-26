"""Tests for bdpy.dataform._mat_v73."""

import os
import tempfile
import unittest

import h5py
import numpy as np

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


def _make_array(shape, dtype):
    """Build a deterministic test array of the given shape and dtype."""
    dt = np.dtype(dtype)
    size = int(np.prod(shape)) if len(shape) else 1
    if dt == np.dtype(bool):
        values = (np.arange(size) % 2).astype(bool)
    elif np.issubdtype(dt, np.integer):
        values = (np.arange(size) % 7 - 3).astype(dt)
    else:
        values = (np.arange(size, dtype=np.float64) * 1.5 + 0.25).astype(dt)
    return values.reshape(shape)


class TestWriteReadRoundTrip(unittest.TestCase):

    shapes = [(), (1,), (10,), (3, 2), (1, 1000), (4, 3, 2)]
    dtypes = [np.float64, np.float32, np.int32, np.int64, np.uint8, np.bool_]

    def test_round_trip_shapes_and_dtypes(self):
        # Writing with savemat and reading back with read_dataset must preserve
        # values, shape (incl. 1-D / scalar) and dtype (incl. bool -> bool).
        with tempfile.TemporaryDirectory() as tmpdir:
            for shape in self.shapes:
                for dtype in self.dtypes:
                    with self.subTest(shape=shape, dtype=np.dtype(dtype).name):
                        original = _make_array(shape, dtype)
                        fname = os.path.join(tmpdir, 'rt.mat')
                        _mat_v73.savemat(fname, {'x': original})
                        out = _mat_v73.load_array(fname, 'x')

                        np.testing.assert_array_equal(out, original)
                        self.assertEqual(out.shape, original.shape)
                        self.assertEqual(out.dtype, np.dtype(dtype))
            # bool specifically must come back as np.bool_, not uint8.
            self.assertEqual(
                _make_array((3,), np.bool_).dtype, np.dtype(bool))

    def test_empty_arrays_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for shape in [(0, 5), (5, 0), (0,)]:
                with self.subTest(shape=shape):
                    original = np.empty(shape, dtype=np.float64)
                    fname = os.path.join(tmpdir, 'empty.mat')
                    _mat_v73.savemat(fname, {'x': original})
                    out = _mat_v73.load_array(fname, 'x')

                    self.assertEqual(out.shape, shape)
                    self.assertEqual(out.size, 0)
                    self.assertEqual(out.dtype, np.dtype(np.float64))


class TestOnDiskLayout(unittest.TestCase):

    def test_layout_and_attributes(self):
        # White-box checks pinning the MATLAB v7.3 on-disk layout so future
        # refactors do not silently break MATLAB compatibility.
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'layout.mat')
            _mat_v73.savemat(fname, {
                'vec': np.arange(10, dtype=np.float64),
                'mat': np.arange(6, dtype=np.float64).reshape(3, 2),
                'ints': np.arange(6, dtype=np.int32).reshape(3, 2),
                'flag': np.array([True, False, True]),
            })
            with h5py.File(fname, 'r') as f:
                # 1-D stored as a column vector -> on-disk (1, n).
                self.assertEqual(f['vec'].shape, (1, 10))
                self.assertEqual(bytes(f['vec'].attrs['MATLAB_class']), b'double')
                np.testing.assert_array_equal(f['vec'].attrs['Python.Shape'], [10])

                # 2-D stored transposed (reversed dims).
                self.assertEqual(f['mat'].shape, (2, 3))
                np.testing.assert_array_equal(f['mat'].attrs['Python.Shape'], [3, 2])

                # Integers carry MATLAB_int_decode (element byte size).
                self.assertEqual(bytes(f['ints'].attrs['MATLAB_class']), b'int32')
                self.assertEqual(int(f['ints'].attrs['MATLAB_int_decode']), 4)

                # bool -> logical stored as uint8.
                self.assertEqual(bytes(f['flag'].attrs['MATLAB_class']), b'logical')
                self.assertEqual(int(f['flag'].attrs['MATLAB_int_decode']), 1)
                self.assertEqual(f['flag'].dtype, np.uint8)


class TestMatfileHeader(unittest.TestCase):

    def test_userblock_header(self):
        # savemat must prepend a MAT-file v7.3 userblock header so MATLAB
        # recognizes the file, while staying a valid HDF5 file for h5py.
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'hdr.mat')
            _mat_v73.savemat(fname, {'x': np.zeros((2, 2))})

            with open(fname, 'rb') as fh:
                header = fh.read(128)
            self.assertTrue(header.startswith(b'MATLAB 7.3 MAT-file'))
            self.assertEqual(header[124:126], b'\x00\x02')  # version 0x0200
            self.assertEqual(header[126:128], b'IM')         # little-endian

            np.testing.assert_array_equal(
                _mat_v73.load_array(fname, 'x'), np.zeros((2, 2)))


class TestSavemat(unittest.TestCase):

    def test_multiple_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'multi.mat')
            a = np.arange(6, dtype=np.float64).reshape(2, 3)
            b = np.arange(4, dtype=np.float64).reshape(2, 2)
            _mat_v73.savemat(fname, {'a': a, 'b': b})

            with h5py.File(fname, 'r') as f:
                self.assertIn('a', f)
                self.assertIn('b', f)
            np.testing.assert_array_equal(_mat_v73.load_array(fname, 'a'), a)
            np.testing.assert_array_equal(_mat_v73.load_array(fname, 'b'), b)

    def test_append_mode_preserves_and_replaces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'append.mat')
            keep = np.arange(3, dtype=np.float64)
            _mat_v73.savemat(fname, {'keep': keep})
            _mat_v73.savemat(fname, {'add': np.arange(4, dtype=np.float64)}, mode='a')

            np.testing.assert_array_equal(_mat_v73.load_array(fname, 'keep'), keep)
            np.testing.assert_array_equal(
                _mat_v73.load_array(fname, 'add'), np.arange(4, dtype=np.float64))


class TestSaveFeatureRoundTrip(unittest.TestCase):

    def test_save_feature_round_trip(self):
        from bdpy.dataform.features import save_feature

        with tempfile.TemporaryDirectory() as tmpdir:
            for shape in [(1, 1000), (1, 256, 13, 13)]:
                feature = np.random.rand(*shape)
                label = 'label_ndim%d' % len(shape)
                save_feature(feature, tmpdir, 'layerX', label)
                path = os.path.join(tmpdir, 'layerX', label + '.mat')
                out = _mat_v73.load_array(path, 'feat')
                np.testing.assert_array_equal(out, feature)


class TestWriterErrors(unittest.TestCase):

    def test_invalid_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'err.mat')
            with self.assertRaises(ValueError):
                _mat_v73.savemat(fname, {'': np.zeros(2)})
            with self.assertRaises(ValueError):
                _mat_v73.savemat(fname, {'a/b': np.zeros(2)})

    def test_unsupported_dtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'err.mat')
            with self.assertRaises(TypeError):
                _mat_v73.savemat(fname, {'x': np.array([1 + 2j, 3 + 4j])})
            with self.assertRaises(TypeError):
                _mat_v73.savemat(fname, {'x': np.array([1, 'a'], dtype=object)})
            with self.assertRaises(TypeError):
                _mat_v73.savemat(fname, {'x': np.array(['a', 'b'])})


if __name__ == '__main__':
    unittest.main()
