'''Tests for dataform'''

import os
import tempfile
import unittest

import h5py
import numpy as np

from bdpy.dataform.sparse import load_array, save_array


class TestSparse(unittest.TestCase):

    def test_load_save_dense_array(self):
        payloads = [
            [(10,), 'test_array_dense_ndim1.mat'],  # ndim = 1
            [(3, 2), 'test_array_dense_ndim2.mat'],  # ndim = 2
            [(4, 3, 2), 'test_array_dense_ndim3.mat']  # ndim = 3
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for shape, fname in payloads:
                original_data = np.random.rand(*shape)
                save_array(tmpdir + '/' + fname, original_data, key='testdata')
                from_file = load_array(tmpdir + '/' + fname, key='testdata')

                np.testing.assert_array_equal(original_data, from_file)

    def test_load_save_sparse_array(self):
        payloads = [
            [(10,), 'test_array_sparse_ndim1.mat'],  # ndim = 1
            [(3, 2), 'test_array_sparse_ndim2.mat'],  # ndim = 2
            [(4, 3, 2), 'test_array_sparse_ndim3.mat']  # ndim = 3
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for shape, fname in payloads:
                original_data = np.random.rand(*shape)
                original_data[original_data < 0.8] = 0

                save_array(tmpdir + '/' + fname, original_data, key='testdata', sparse=True)
                from_file = load_array(tmpdir + '/' + fname, key='testdata')

                np.testing.assert_array_equal(original_data, from_file)

    def test_sparse_save_preserves_other_variables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test_sparse_preserve.mat')

            with h5py.File(fname, 'w') as f:
                f.create_dataset('other', data=np.array([1, 2, 3]))

            original_data = np.random.rand(3, 2)
            original_data[original_data < 0.8] = 0

            save_array(fname, original_data, key='data', sparse=True)
            from_file = load_array(fname, key='data')

            np.testing.assert_array_equal(original_data, from_file)

            with h5py.File(fname, 'r') as f:
                np.testing.assert_array_equal(f['other'][()], np.array([1, 2, 3]))

    def test_sparse_save_int_and_bool_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cases = [
                (np.array([[0, 2, 0], [3, 0, 0]], dtype=np.int64), np.int64),
                (np.array([[True, False], [False, True]]), np.bool_),
            ]
            for i, (original, dtype) in enumerate(cases):
                fname = os.path.join(tmpdir, 'sparse_dtype_%d.mat' % i)
                save_array(fname, original, key='data', sparse=True, dtype=dtype)
                from_file = load_array(fname, key='data')
                np.testing.assert_array_equal(original, from_file)

    def test_sparse_save_writes_matlab_v73_struct(self):
        # A freshly created sparse file must be a MATLAB v7.3 struct: a group
        # tagged MATLAB_class='struct' with MATLAB_fields, fields carrying
        # MATLAB_class, and a MAT-file userblock header at the start of the file.
        # (Kept separate from the preserve-other-variables test, whose
        # pre-existing plain file has no userblock.)
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'sparse_struct.mat')
            original = np.array([[1., 0, 0, 0], [2, 2, 0, 0], [3, 3, 3, 0]])
            save_array(fname, original, key='data', sparse=True)

            with open(fname, 'rb') as fh:
                header = fh.read(128)
            self.assertTrue(header.startswith(b'MATLAB 7.3 MAT-file'))
            self.assertEqual(header[124:126], b'\x00\x02')
            self.assertEqual(header[126:128], b'IM')

            with h5py.File(fname, 'r') as f:
                g = f['data']
                self.assertIsInstance(g, h5py.Group)
                self.assertEqual(bytes(g.attrs['MATLAB_class']), b'struct')
                fields = [b''.join(list(x)).decode('ascii')
                          for x in g.attrs['MATLAB_fields']]
                self.assertEqual(
                    set(fields),
                    {'__bdpy_sparse_arrray', 'index', 'value', 'shape',
                     'background'})
                for name in fields:
                    self.assertIn('MATLAB_class', g[name].attrs)

    def test_load_array_jl(self):
        data = np.array([[1, 0, 0, 0],
                         [2, 2, 0, 0],
                         [3, 3, 3, 0]])
        data_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, 'data'
        ))

        testdata = load_array(
            os.path.join(data_dir, 'array_jl_dense_v1.mat'), key='a')
        np.testing.assert_array_equal(data, testdata)

        testdata = load_array(
            os.path.join(data_dir, 'array_jl_sparse_v1.mat'), key='a')
        np.testing.assert_array_equal(data, testdata)


if __name__ == '__main__':
    unittest.main()
