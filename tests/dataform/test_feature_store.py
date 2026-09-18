"""Backend equivalence for feature storage.

The legacy .mat layout and chunked HDF5 storage must be indistinguishable
through the Features API: same labels, same order, same arrays, for every
combination of label selection and feature slicing. These tests run each query
against both backends and compare.
"""

import os
import tempfile
import unittest
import warnings

import h5py
import numpy as np
from numpy.testing import assert_array_equal

from bdpy.dataform import Features, convert_features_to_hdf5, save_features
from bdpy.dataform._feature_store import (
    HDF5FeatureStore,
    MatFeatureStore,
    detect_format,
)

from .test_features import prepare_mat_features

LAYERS = ['conv5', 'fc8']
SHAPES = [(1, 24, 5, 5), (1, 60)]
LABELS = ['img%04d' % i for i in range(11)]

# Every way a caller might ask for features. Each is run against both backends.
QUERIES = [
    ('all', 'conv5', None, None),
    ('single label', 'conv5', 'img0003', None),
    ('label list', 'conv5', ['img0005', 'img0001', 'img0009'], None),
    ('unsorted labels', 'conv5', ['img0009', 'img0000'], None),
    ('repeated labels', 'conv5', ['img0009', 'img0001', 'img0009', 'img0000'], None),
    ('feature slice', 'conv5', None, np.s_[8:16]),
    ('slice with step', 'conv5', None, np.s_[::2]),
    ('multi-axis slice', 'conv5', None, np.s_[8:16, 1:4, :]),
    ('integer index', 'conv5', None, np.s_[5]),
    ('fancy index', 'conv5', None, np.s_[[3, 1, 7]]),
    ('ellipsis', 'conv5', None, np.s_[..., 1:3]),
    ('labels and slice', 'conv5', ['img0009', 'img0001', 'img0009'], np.s_[8:16]),
    ('2d layer', 'fc8', None, np.s_[10:40]),
    ('2d layer with labels', 'fc8', ['img0002', 'img0000'], np.s_[10:40]),
]


class _BackendPair(unittest.TestCase):
    """Builds the same features as a .mat tree and as chunked HDF5."""

    def setUp(self):
        warnings.simplefilter('ignore', FutureWarning)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.matdir = os.path.join(self.tmpdir.name, 'mat')
        self.h5dir = os.path.join(self.tmpdir.name, 'h5')
        os.makedirs(self.matdir)
        self.stacked = prepare_mat_features(self.matdir, LAYERS, LABELS, SHAPES)
        convert_features_to_hdf5(self.matdir, self.h5dir)
        self.from_mat = Features(self.matdir)
        self.from_h5 = Features(self.h5dir)

    def tearDown(self):
        self.tmpdir.cleanup()


class TestBackendEquivalence(_BackendPair):
    def test_metadata_matches(self):
        self.assertEqual(self.from_h5.layers, self.from_mat.layers)
        self.assertEqual(self.from_h5.labels, self.from_mat.labels)
        assert_array_equal(self.from_h5.index, self.from_mat.index)
        for layer in LAYERS:
            self.assertEqual(self.from_h5.shape(layer), self.from_mat.shape(layer))

    def test_queries_match_across_backends(self):
        for name, layer, label, feature_slice in QUERIES:
            with self.subTest(query=name):
                from_mat = self.from_mat.get(
                    layer, label=label, feature_slice=feature_slice
                )
                from_h5 = self.from_h5.get(
                    layer, label=label, feature_slice=feature_slice
                )
                self.assertEqual(from_h5.shape, from_mat.shape)
                assert_array_equal(from_h5, from_mat)

    def test_queries_match_ground_truth(self):
        # Both backends agreeing is not enough if both are wrong the same way.
        for name, layer, label, feature_slice in QUERIES:
            with self.subTest(query=name):
                if label is None:
                    rows = slice(None)
                elif isinstance(label, str):
                    rows = [LABELS.index(label)]
                else:
                    rows = [LABELS.index(s) for s in label]
                expected = self.stacked[layer][rows]
                if feature_slice is not None:
                    indexers = (
                        feature_slice
                        if isinstance(feature_slice, tuple)
                        else (feature_slice,)
                    )
                    expected = expected[(slice(None), *indexers)]
                assert_array_equal(
                    self.from_h5.get(layer, label=label, feature_slice=feature_slice),
                    expected,
                )

    def test_statistic_matches(self):
        for statistic in ('mean', 'std', 'std, ddof=0'):
            for layer in LAYERS:
                with self.subTest(statistic=statistic, layer=layer):
                    assert_array_equal(
                        self.from_h5.statistic(statistic, layer=layer),
                        self.from_mat.statistic(statistic, layer=layer),
                    )


class TestIterChunks(_BackendPair):
    def test_reassembles_to_a_full_read(self):
        for axis in (0, 1):
            for features in (self.from_mat, self.from_h5):
                with self.subTest(axis=axis, backend=type(features).__name__):
                    blocks = list(features.iter_chunks('conv5', axis=axis, size=4))
                    joined = np.concatenate([b for _, b in blocks], axis=axis)
                    assert_array_equal(joined, self.stacked['conv5'])

    def test_yielded_slices_address_the_right_block(self):
        full = self.from_h5.get('conv5')
        for sl, block in self.from_h5.iter_chunks('conv5', axis=1, size=5):
            assert_array_equal(block, full[:, sl])

    def test_slices_cover_the_axis_without_overlap(self):
        length = self.from_h5.shape('conv5')[1]
        covered = []
        for sl, _ in self.from_h5.iter_chunks('conv5', axis=1, size=5):
            covered.extend(range(*sl.indices(length)))
        self.assertEqual(covered, list(range(length)))

    def test_composes_with_a_feature_slice(self):
        expected = self.stacked['conv5'][:, 4:20]
        blocks = list(
            self.from_h5.iter_chunks('conv5', feature_slice=np.s_[4:20], axis=1, size=6)
        )
        assert_array_equal(np.concatenate([b for _, b in blocks], axis=1), expected)

    def test_restricts_to_labels(self):
        labels = ['img0009', 'img0001']
        blocks = list(self.from_h5.iter_chunks('conv5', label=labels, axis=1, size=5))
        joined = np.concatenate([b for _, b in blocks], axis=1)
        assert_array_equal(joined, self.stacked['conv5'][[9, 1]])

    def test_default_size_is_chunk_aligned(self):
        store = HDF5FeatureStore(self.h5dir)
        extent = store.chunk_extent('conv5', 1)
        sizes = {
            sl.stop - sl.start
            for sl, _ in self.from_h5.iter_chunks('conv5', axis=1)
        }
        # Every block but the last is a whole chunk along the axis.
        self.assertLessEqual(max(sizes), max(1, extent))

    def test_rejects_bad_axis(self):
        with self.assertRaises(ValueError):
            list(self.from_h5.iter_chunks('conv5', axis=9))


class TestPartialReads(_BackendPair):
    def test_slice_is_pushed_down_to_h5py(self):
        # The point of the format: h5py must receive the narrowed selection, so
        # that only the covering chunks are read, rather than a full read that
        # NumPy then slices.
        seen = []
        original = h5py.Dataset.__getitem__

        def spy(dataset, args):
            seen.append(args)
            return original(dataset, args)

        h5py.Dataset.__getitem__ = spy
        try:
            self.from_h5.get('conv5', feature_slice=np.s_[8:16])
        finally:
            h5py.Dataset.__getitem__ = original

        self.assertEqual(len(seen), 1)
        rows, feature_index = seen[0]
        self.assertEqual(rows, slice(None))
        self.assertEqual(feature_index, slice(8, 16, None))

    def test_labels_are_read_once_and_in_order(self):
        # Duplicated and out-of-order labels must reach h5py as a single
        # strictly increasing index list, which is all h5py accepts.
        seen = []
        original = h5py.Dataset.__getitem__

        def spy(dataset, args):
            seen.append(args)
            return original(dataset, args)

        h5py.Dataset.__getitem__ = spy
        try:
            out = self.from_h5.get('conv5', label=['img0009', 'img0001', 'img0009'])
        finally:
            h5py.Dataset.__getitem__ = original

        self.assertEqual(len(seen), 1)
        rows = seen[0] if not isinstance(seen[0], tuple) else seen[0][0]
        self.assertEqual(rows, [1, 9])
        assert_array_equal(out, self.stacked['conv5'][[9, 1, 9]])

    def test_dataset_is_chunked_on_both_sliceable_axes(self):
        with h5py.File(os.path.join(self.h5dir, 'conv5.h5'), 'r') as f:
            chunks = f['features'].chunks
            shape = f['features'].shape
        self.assertIsNotNone(chunks)
        self.assertEqual(chunks[2:], shape[2:])  # spatial axes kept whole


class TestFormatDetection(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', FutureWarning)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.matdir = os.path.join(self.tmpdir.name, 'mat')
        self.h5dir = os.path.join(self.tmpdir.name, 'h5')
        os.makedirs(self.matdir)
        self.stacked = prepare_mat_features(self.matdir, LAYERS, LABELS, SHAPES)
        convert_features_to_hdf5(self.matdir, self.h5dir)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_detects_each_layout(self):
        self.assertEqual(detect_format(self.matdir), 'mat')
        self.assertEqual(detect_format(self.h5dir), 'hdf5')

    def test_explicit_format_overrides_detection(self):
        self.assertIsInstance(
            Features(self.h5dir, format='hdf5')._Features__stores[0],
            HDF5FeatureStore,
        )
        self.assertIsInstance(
            Features(self.matdir, format='mat')._Features__stores[0],
            MatFeatureStore,
        )

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            Features(self.h5dir, format='parquet')

    def test_mixed_directories(self):
        # One dpath per layout, read through a single Features.
        h5_only = os.path.join(self.tmpdir.name, 'h5b')
        os.makedirs(h5_only)
        other_labels = ['other%04d' % i for i in range(4)]
        for layer, shape in zip(LAYERS, SHAPES):
            data = np.random.rand(len(other_labels), *shape[1:])
            save_features(
                os.path.join(h5_only, layer + '.h5'), data, other_labels, layer=layer
            )

        features = Features([self.matdir, h5_only])
        self.assertEqual(features.labels, LABELS + other_labels)
        # A query spanning both directories keeps the caller's order.
        got = features.get('conv5', label=['other0002', 'img0003', 'other0000'])
        assert_array_equal(got[1], self.stacked['conv5'][3])
        self.assertEqual(got.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
