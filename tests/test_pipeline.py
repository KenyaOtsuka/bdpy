"""Tests for bdpy.pipeline."""


from unittest import TestCase, TestLoader, TextTestRunner
import os
import sys
import tempfile

import yaml
import hydra

from bdpy.pipeline.config import init_hydra_cfg


class TestPipeline(TestCase):
    """Tests for bdpy.pipeline."""

    def setUp(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.config_path = os.path.join(tmpdir.name, "testconf.yaml")
        with open(self.config_path, "w") as f:
            yaml.dump({"key_int": 0, "key_str": "value"}, f)

        # init_hydra_cfg parses sys.argv and initializes the global Hydra
        # instance, so both are restored whatever happens in the test.
        saved_argv = sys.argv
        self.addCleanup(setattr, sys, "argv", saved_argv)
        self.addCleanup(self._clear_hydra)

    @staticmethod
    def _clear_hydra():
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # hydra-core 1.0.6

    def _init_cfg(self, *args):
        """Run init_hydra_cfg as if invoked as `prog <config> <args...>`."""
        sys.argv = ["prog", self.config_path, *args]
        try:
            return init_hydra_cfg()
        finally:
            # Hydra refuses a second initialization until it is cleared.
            self._clear_hydra()

    def test_config_init_hydra_cfg_default(self):
        """Tests for bdpy.pipeline.config.init_hydra_cfg."""
        cfg = self._init_cfg()
        self.assertEqual(cfg.key_int, 0)
        self.assertEqual(cfg.key_str, "value")

    def test_config_init_hydra_cfg_override(self):
        """Tests for bdpy.pipeline.config.init_hydra_cfg."""
        cases = [
            (["key_int=1"], 1, "value"),
            (["key_str=hoge"], 0, "hoge"),
            (["key_str='hoge fuga'"], 0, "hoge fuga"),
            (["key_int=1024", "key_str=foo"], 1024, "foo"),
        ]
        for overrides, key_int, key_str in cases:
            with self.subTest(overrides=overrides):
                cfg = self._init_cfg("-o", *overrides)
                self.assertEqual(cfg.key_int, key_int)
                self.assertEqual(cfg.key_str, key_str)

    def test_config_init_hydra_cfg_run(self):
        """Tests for bdpy.pipeline.config.init_hydra_cfg."""
        # The run name is the stem of the file that calls init_hydra_cfg,
        # i.e. this module, regardless of how the tests are launched.
        cfg = self._init_cfg()
        self.assertEqual(cfg._run_.name, "test_pipeline")

        cfg = self._init_cfg("-a", "overridden_analysis_name")
        self.assertEqual(cfg._run_.name, "test_pipeline")

    def test_config_init_hydra_cfg_analysis(self):
        """Tests for bdpy.pipeline.config.init_hydra_cfg."""
        cfg = self._init_cfg()
        self.assertEqual(cfg._analysis_name_, "test_pipeline")

        cfg = self._init_cfg("-a", "overridden_analysis_name")
        self.assertEqual(cfg._analysis_name_, "overridden_analysis_name")


if __name__ == "__main__":
    suite = TestLoader().loadTestsFromTestCase(TestPipeline)
    TextTestRunner(verbosity=2).run(suite)
