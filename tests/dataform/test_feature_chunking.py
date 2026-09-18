import unittest

import numpy as np

from bdpy.dataform._feature_chunking import (
    DEFAULT_TARGET_CHUNK_BYTES,
    choose_chunk_shape,
)


def _nbytes(chunk, dtype):
    return int(np.prod(chunk)) * np.dtype(dtype).itemsize


class TestChooseChunkShape(unittest.TestCase):
    """Chunk shape decides partial-read cost, so pin the policy down."""

    shapes = [
        (1200, 1000),
        (1200, 256, 13, 13),
        (50, 1000),
        (1, 256, 13, 13),
        (3, 4096, 7, 7),
        (1000000, 1),
        (2, 1000000),
        (17, 63, 5),
    ]

    def test_invariants(self):
        for dtype in (np.float32, np.float64, np.int16):
            for shape in self.shapes:
                chunk = choose_chunk_shape(shape, np.dtype(dtype))
                with self.subTest(shape=shape, dtype=dtype):
                    self.assertEqual(len(chunk), len(shape))
                    # Never zero, never larger than the dataset.
                    for c, s in zip(chunk, shape):
                        self.assertGreaterEqual(c, 1)
                        self.assertLessEqual(c, s)
                    # Trailing axes are kept whole.
                    self.assertEqual(chunk[2:], tuple(shape[2:]))

    def test_chunk_stays_within_budget(self):
        dtype = np.dtype(np.float32)
        for shape in self.shapes:
            chunk = choose_chunk_shape(shape, dtype)
            with self.subTest(shape=shape):
                if _nbytes(shape, dtype) <= DEFAULT_TARGET_CHUNK_BYTES:
                    # Small enough to store as a single chunk.
                    self.assertEqual(chunk, tuple(shape))
                else:
                    self.assertLessEqual(
                        _nbytes(chunk, dtype), DEFAULT_TARGET_CHUNK_BYTES
                    )

    def test_budget_is_actually_used(self):
        # A chunk far below budget means needless read amplification: every
        # oversized array should fill at least a quarter of the budget.
        dtype = np.dtype(np.float32)
        for shape in self.shapes:
            if _nbytes(shape, dtype) <= DEFAULT_TARGET_CHUNK_BYTES:
                continue
            chunk = choose_chunk_shape(shape, dtype)
            with self.subTest(shape=shape):
                self.assertGreater(
                    _nbytes(chunk, dtype), DEFAULT_TARGET_CHUNK_BYTES // 4
                )

    def test_both_sliceable_axes_are_chunked(self):
        # The point of #144: a channel slice must not have to read every channel.
        shape = (1200, 256, 13, 13)
        chunk = choose_chunk_shape(shape, np.dtype(np.float32))
        self.assertLess(chunk[0], shape[0])
        self.assertLess(chunk[1], shape[1])

    def test_whole_array_below_budget_is_one_chunk(self):
        shape = (50, 1000)
        self.assertEqual(
            choose_chunk_shape(shape, np.dtype(np.float32)), shape
        )

    def test_single_sample(self):
        chunk = choose_chunk_shape((1, 4096, 7, 7), np.dtype(np.float32))
        self.assertEqual(chunk[0], 1)

    def test_target_bytes_is_respected(self):
        dtype = np.dtype(np.float32)
        shape = (1200, 256, 13, 13)
        small = choose_chunk_shape(shape, dtype, target_bytes=64 * 1024)
        large = choose_chunk_shape(shape, dtype, target_bytes=4 * 1024 * 1024)
        self.assertLessEqual(_nbytes(small, dtype), 64 * 1024)
        self.assertLessEqual(_nbytes(large, dtype), 4 * 1024 * 1024)
        self.assertLess(_nbytes(small, dtype), _nbytes(large, dtype))

    def test_unknown_sample_count_is_not_capped(self):
        # Resizable datasets start empty; the sample extent must come from the
        # budget, not from the current (zero) size.
        chunk = choose_chunk_shape(
            (0, 256, 13, 13), np.dtype(np.float32), n_samples_known=False
        )
        self.assertGreater(chunk[0], 1)

    def test_deterministic(self):
        shape = (1200, 256, 13, 13)
        dtype = np.dtype(np.float32)
        self.assertEqual(
            choose_chunk_shape(shape, dtype), choose_chunk_shape(shape, dtype)
        )

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            choose_chunk_shape((100,), np.dtype(np.float32))
        with self.assertRaises(ValueError):
            choose_chunk_shape((100, 10), np.dtype(np.float32), target_bytes=0)


if __name__ == "__main__":
    unittest.main()
