import os
import tempfile
import unittest
import warnings

import h5py
import numpy as np
from numpy.testing import assert_array_equal

from bdpy.dataform import Features
from bdpy.dataform._feature_store import (
    FORMAT_ATTR,
    FORMAT_NAME,
    FORMAT_VERSION_ATTR,
    SUPPORTED_FORMAT_VERSION,
    HDF5FeatureStore,
)
from bdpy.dataform.feature_hdf5 import (
    FeatureWriter,
    convert_features_to_hdf5,
    save_features,
)

from .test_features import prepare_mat_features


class TestSaveFeatures(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.labels = ['img%04d' % i for i in range(12)]
        self.data = np.random.rand(12, 32, 5, 5).astype(np.float32)
        self.path = os.path.join(self.tmpdir.name, 'conv5.h5')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_writes_schema_v1(self):
        save_features(self.path, self.data, self.labels)

        with h5py.File(self.path, 'r') as f:
            self.assertEqual(f.attrs[FORMAT_ATTR], FORMAT_NAME)
            self.assertEqual(
                int(f.attrs[FORMAT_VERSION_ATTR]), SUPPORTED_FORMAT_VERSION
            )
            self.assertEqual(f.attrs['layer'], 'conv5')
            assert_array_equal(f['features'][()], self.data)
            self.assertEqual(
                [s.decode('utf-8') for s in f['labels'][()]], self.labels
            )

    def test_features_dataset_is_chunked_and_uncompressed(self):
        # Chunking is the whole point; compression would tax every partial read.
        save_features(self.path, self.data, self.labels)
        with h5py.File(self.path, 'r') as f:
            self.assertIsNotNone(f['features'].chunks)
            self.assertIsNone(f['features'].compression)

    def test_explicit_chunks_and_compression(self):
        save_features(
            self.path, self.data, self.labels,
            chunks=(4, 8, 5, 5), compression='gzip',
        )
        with h5py.File(self.path, 'r') as f:
            self.assertEqual(f['features'].chunks, (4, 8, 5, 5))
            self.assertEqual(f['features'].compression, 'gzip')
        # Compression must not change what comes back out.
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data
        )

    def test_dtype_is_preserved(self):
        for dtype in (np.float32, np.float64, np.int32):
            with self.subTest(dtype=dtype):
                path = os.path.join(self.tmpdir.name, 'l_%s.h5' % np.dtype(dtype).name)
                save_features(path, self.data.astype(dtype), self.labels)
                with h5py.File(path, 'r') as f:
                    self.assertEqual(f['features'].dtype, np.dtype(dtype))

    def test_rejects_mismatched_labels(self):
        with self.assertRaises(ValueError):
            save_features(self.path, self.data, self.labels[:-1])

    def test_rejects_1d_features(self):
        with self.assertRaises(ValueError):
            save_features(self.path, np.arange(5), ['a'] * 5)


class TestFeatureWriter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, 'conv5.h5')
        self.labels = ['img%04d' % i for i in range(150)]
        self.data = np.random.rand(150, 16, 3, 3).astype(np.float32)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_append_round_trip(self):
        # 150 samples crosses several resize blocks, which is where an
        # incremental writer would lose data if the resize were wrong.
        with FeatureWriter(self.path, (16, 3, 3), np.float32) as writer:
            for label, feature in zip(self.labels, self.data):
                writer.append(feature, label)
            self.assertEqual(writer.n_samples, len(self.labels))

        store = HDF5FeatureStore(self.tmpdir.name)
        self.assertEqual(store.labels, self.labels)
        assert_array_equal(store.read('conv5'), self.data)

    def test_append_accepts_leading_sample_axis(self):
        with FeatureWriter(self.path, (16, 3, 3), np.float32) as writer:
            writer.append(self.data[0][np.newaxis], self.labels[0])
            writer.append(self.data[1], self.labels[1])
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data[:2]
        )

    def test_extend(self):
        with FeatureWriter(self.path, (16, 3, 3), np.float32) as writer:
            writer.extend(self.data[:40], self.labels[:40])
            writer.extend(self.data[40:], self.labels[40:])
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data
        )

    def test_writes_valid_header(self):
        with FeatureWriter(self.path, (16, 3, 3), np.float32, layer='conv5') as writer:
            writer.append(self.data[0], self.labels[0])
        with h5py.File(self.path, 'r') as f:
            self.assertEqual(f.attrs[FORMAT_ATTR], FORMAT_NAME)
            self.assertEqual(f.attrs['layer'], 'conv5')
            self.assertIsNotNone(f['features'].chunks)

    def test_rejects_wrong_shape(self):
        with FeatureWriter(self.path, (16, 3, 3), np.float32) as writer:
            with self.assertRaises(ValueError):
                writer.extend(np.zeros((2, 8, 3, 3)), ['a', 'b'])

    def test_rejects_label_count_mismatch(self):
        with FeatureWriter(self.path, (16, 3, 3), np.float32) as writer:
            with self.assertRaises(ValueError):
                writer.extend(np.zeros((2, 16, 3, 3)), ['a'])

    def test_write_after_close_raises(self):
        writer = FeatureWriter(self.path, (16, 3, 3), np.float32)
        writer.close()
        writer.close()  # idempotent
        with self.assertRaises(RuntimeError):
            writer.append(self.data[0], self.labels[0])


class TestConvertFeaturesToHDF5(unittest.TestCase):
    """The converter must reproduce exactly what the legacy reader sees."""

    def setUp(self):
        warnings.simplefilter('ignore', FutureWarning)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.matdir = os.path.join(self.tmpdir.name, 'mat')
        self.h5dir = os.path.join(self.tmpdir.name, 'h5')
        os.makedirs(self.matdir)
        self.labels = ['img%04d' % i for i in range(10)]
        self.stacked = prepare_mat_features(
            self.matdir, ['conv5', 'fc8'], self.labels,
            [(1, 16, 3, 3), (1, 50)],
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_converted_features_match_source(self):
        convert_features_to_hdf5(self.matdir, self.h5dir)

        from_mat = Features(self.matdir)
        from_h5 = Features(self.h5dir)
        self.assertEqual(from_h5.layers, from_mat.layers)
        self.assertEqual(from_h5.labels, from_mat.labels)
        for layer in from_mat.layers:
            with self.subTest(layer=layer):
                assert_array_equal(from_h5.get(layer), self.stacked[layer])
                assert_array_equal(from_h5.get(layer), from_mat.get(layer))

    def test_selected_layers_only(self):
        convert_features_to_hdf5(self.matdir, self.h5dir, layers=['fc8'])
        self.assertEqual(Features(self.h5dir).layers, ['fc8'])

    def test_unknown_layer_raises(self):
        with self.assertRaises(KeyError):
            convert_features_to_hdf5(self.matdir, self.h5dir, layers=['nope'])

    def test_existing_file_is_skipped_unless_overwrite(self):
        convert_features_to_hdf5(self.matdir, self.h5dir)
        marker = os.path.join(self.h5dir, 'fc8.h5')
        mtime = os.path.getmtime(marker)
        os.utime(marker, (mtime - 100, mtime - 100))

        convert_features_to_hdf5(self.matdir, self.h5dir)
        self.assertEqual(os.path.getmtime(marker), mtime - 100)

        convert_features_to_hdf5(self.matdir, self.h5dir, overwrite=True)
        self.assertGreater(os.path.getmtime(marker), mtime - 100)

    def test_small_batches_produce_the_same_file(self):
        # Batching is an implementation detail; it must not affect the result.
        convert_features_to_hdf5(self.matdir, self.h5dir, batch_size=3)
        assert_array_equal(Features(self.h5dir).get('conv5'), self.stacked['conv5'])


class TestFormatValidation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, 'conv5.h5')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_foreign_hdf5_is_rejected(self):
        # A file that merely happens to have /features is not bdpy storage.
        with h5py.File(self.path, 'w') as f:
            f.create_dataset('features', data=np.zeros((2, 3)))
            f.create_dataset('labels', data=['a', 'b'],
                             dtype=h5py.string_dtype(encoding='utf-8'))
        with self.assertRaises(RuntimeError):
            HDF5FeatureStore(self.tmpdir.name)

    def test_future_version_is_rejected_with_a_clear_message(self):
        save_features(self.path, np.zeros((2, 3)), ['a', 'b'])
        with h5py.File(self.path, 'a') as f:
            f.attrs[FORMAT_VERSION_ATTR] = SUPPORTED_FORMAT_VERSION + 1
        with self.assertRaises(RuntimeError) as ctx:
            HDF5FeatureStore(self.tmpdir.name)
        self.assertIn('upgrade bdpy', str(ctx.exception))

    def test_missing_dataset_is_rejected(self):
        save_features(self.path, np.zeros((2, 3)), ['a', 'b'])
        with h5py.File(self.path, 'a') as f:
            del f['labels']
        with self.assertRaises(RuntimeError):
            HDF5FeatureStore(self.tmpdir.name)

    def test_empty_directory_is_rejected(self):
        with self.assertRaises(RuntimeError):
            HDF5FeatureStore(self.tmpdir.name)


if __name__ == "__main__":
    unittest.main()
