"""Regenerate the fixture models read by tests/ml/test_learning.py::TestModelTest.

Run from anywhere:

    python tests/data/test_models/make_test_models.py

The models are written next to this script, one directory per model, each of
which is removed first (ModelTraining skips a model whose output already
exists). The inputs are tiny -- 6 input features, 3 or (2, 2, 2) outputs -- and
drawn from a fixed seed, so the models only need to load and predict with the
expected shapes; their values carry no meaning beyond that.

`n_feat` is not passed to FastL2LiR: with NumPy 2 its feature-selection path
currently raises (see the fastl2lir/NumPy 2 incompatibility).
"""

import os
import shutil
import tempfile

import numpy as np
from fastl2lir import FastL2LiR
from sklearn.linear_model import LinearRegression

from bdpy.distcomp import DistComp
from bdpy.ml import ModelTraining

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

N_SAMPLES = 8
N_FEATURES = 6


def _train(key, model, X, Y, model_parameters=None, chunk_axis=None, save_format='pickle'):
    save_path = os.path.join(OUT_DIR, key)
    shutil.rmtree(save_path, ignore_errors=True)
    with tempfile.TemporaryDirectory() as lockdir:
        train = ModelTraining(model, X, Y)
        train.id = key
        train.save_path = save_path
        train.save_format = save_format
        # Keep the distcomp database out of the fixture directory.
        train.distcomp = DistComp(backend='sqlite3', db_path=os.path.join(lockdir, key + '.db'))
        if model_parameters is not None:
            train.model_parameters = model_parameters
        if chunk_axis is not None:
            train.chunk_axis = chunk_axis
        if save_format == 'bdmodel':
            train.dtype = np.float32
        train.run()


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    X = rng.random((N_SAMPLES, N_FEATURES))
    Y1 = rng.random((N_SAMPLES, 3))
    Y4 = rng.random((N_SAMPLES, 2, 2, 2))
    params = {'alpha': 100}

    _train('lir-nochunk-pkl', LinearRegression(), X, Y1)
    _train('fastl2lir-nochunk-pkl', FastL2LiR(), X, Y1, params)
    _train('fastl2lir-chunk-pkl', FastL2LiR(), X, Y4, params, chunk_axis=1)
    _train('fastl2lir-nochunk-bd', FastL2LiR(), X, Y1, params, save_format='bdmodel')
    _train('fastl2lir-chunk-bd', FastL2LiR(), X, Y4, params, chunk_axis=1, save_format='bdmodel')
