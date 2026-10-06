"""Tests for bdpy.recon.torch.modules.generator."""

import unittest

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from bdpy.recon.torch.modules import generator as generator_module
from ...._torch_seed import seed_torch


class LinearGenerator(generator_module.NNModuleGenerator):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)

    def generate(self, latent):
        return self.fc(latent)

    def reset_states(self) -> None:
        self.fc.apply(generator_module.call_reset_parameters)


class TestCallResetParameters(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.call_reset_parameters.

    The layer types below are the only parameterised ones used by the models
    bdpy ships (bdpy.dl.torch.models), plus nn.MultiheadAttention for the
    `_reset_parameters` branch. Seeding happens once per test, before the
    module is built, so construction and reset draw different numbers.
    """

    def _assert_weights_reset(self, build):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = build()
            before = copy.deepcopy(module)
            with self.assertWarns(UserWarning):
                module.apply(generator_module.call_reset_parameters)
        for (name, p1), (_, p2) in zip(module.named_parameters(), before.named_parameters()):
            if "weight" not in name:
                continue
            self.assertFalse(torch.equal(p1, p2), msg=f"{name} was not reset")

    def test_resets_layers_with_reset_parameters(self):
        builders = {
            "Conv2d": lambda: nn.Conv2d(2, 3, kernel_size=3),
            "Linear": lambda: nn.Linear(4, 3),
            "ConvTranspose2d": lambda: nn.ConvTranspose2d(2, 3, kernel_size=3),
        }
        for name, build in builders.items():
            with self.subTest(layer=name):
                self._assert_weights_reset(build)

    def test_resets_layers_with_only_private_reset_parameters(self):
        self.assertFalse(hasattr(nn.MultiheadAttention, "reset_parameters"))
        self._assert_weights_reset(lambda: nn.MultiheadAttention(4, num_heads=2))

    def test_ignores_modules_without_reset_method(self):
        module = nn.ReLU()
        with self.assertWarns(UserWarning):
            generator_module.call_reset_parameters(module)

    def test_apply_reaches_nested_children(self):
        self._assert_weights_reset(
            lambda: nn.Sequential(
                nn.Linear(4, 4),
                nn.Sequential(nn.ReLU(), nn.Conv2d(1, 1, kernel_size=2)),
            )
        )


class TestBaseGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.BaseGenerator."""

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, generator_module.BaseGenerator)

    def test_call(self):
        """Test __call__."""

        g = torch.Generator().manual_seed(0)
        class ReturnAsIsGenerator(generator_module.BaseGenerator):
            def generate(self, latent):
                return latent

            def reset_states(self) -> None:
                pass

            def parameters(self, recurse=True):
                return iter([])

        generator = ReturnAsIsGenerator()
        latent = torch.randn(1, 3, 64, 64, generator=g)
        generated_image = generator(latent)
        self.assertEqual(generated_image.shape, (1, 3, 64, 64))


class TestNNModuleGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.NNModuleGenerator."""

    def setUp(self):
        """Set up."""
        seed_torch(self)
        self.generator = LinearGenerator()

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, generator_module.NNModuleGenerator)

    def test_call(self):
        """Test __call__."""
        g = torch.Generator().manual_seed(0)
        latent = torch.randn(1, 64, generator=g)
        generated_image = self.generator(latent)
        self.assertEqual(generated_image.shape, (1, 10))
        generated_image.sum().backward()
        self.assertIsNotNone(self.generator.fc.weight.grad)

    def test_reset_states(self):
        """Test reset_states."""
        generator_copy = copy.deepcopy(self.generator)
        for p1, p2 in zip(self.generator.parameters(), generator_copy.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        self.generator.reset_states()
        for p1, p2 in zip(self.generator.parameters(), generator_copy.parameters()):
            self.assertFalse(torch.equal(p1, p2))


class TestBareGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.BareGenerator."""

    def test_call(self):
        """Test __call__."""
        g = torch.Generator().manual_seed(0)
        generator = generator_module.BareGenerator(activation=torch.sigmoid)
        latent = torch.randn(1, 3, 64, 64, generator=g)
        generated_image = generator(latent)
        self.assertEqual(generated_image.shape, (1, 3, 64, 64))
        torch.testing.assert_close(generated_image, torch.sigmoid(latent))


class TestDNNGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.DNNGenerator."""
    def setUp(self):
        seed_torch(self)

    def test_call(self):
        """Test __call__."""
        g = torch.Generator().manual_seed(0)
        generator_network = LinearGenerator()
        generator = generator_module.DNNGenerator(generator_network)
        latent = torch.randn(1, 64, generator=g)
        generated_image = generator(latent)
        self.assertEqual(generated_image.shape, (1, 10))
        generated_image.sum().backward()
        self.assertIsNotNone(generator_network.fc.weight.grad)

    def test_reset_states(self):
        """Test reset_states."""
        generator = generator_module.DNNGenerator(LinearGenerator())
        generator_copy = copy.deepcopy(generator)
        for p1, p2 in zip(generator.parameters(), generator_copy.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        generator.reset_states()
        for p1, p2 in zip(generator.parameters(), generator_copy.parameters()):
            self.assertFalse(torch.equal(p1, p2))


class TestFrozenGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.FrozenGenerator."""
    def setUp(self):
        seed_torch(self)

    def test_call(self):
        """Test __call__."""
        g = torch.Generator().manual_seed(0)
        generator_network = LinearGenerator()
        generator = generator_module.FrozenGenerator(generator_network)
        latent = torch.randn(1, 64, generator=g)
        generated_image = generator(latent)
        self.assertEqual(generated_image.shape, (1, 10))
        self.assertRaises(ValueError, optim.SGD, generator.parameters())

    def test_reset_states(self):
        """Test reset_states."""
        generator = generator_module.FrozenGenerator(LinearGenerator())
        generator_copy = copy.deepcopy(generator)
        for p1, p2 in zip(generator.parameters(), generator_copy.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        generator.reset_states()
        for p1, p2 in zip(generator.parameters(), generator_copy.parameters()):
            self.assertTrue(torch.equal(p1, p2))


class TestBuildGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.build_generator."""
    def setUp(self):
        seed_torch(self)

    def test_build_generator(self):
        """Test build_generator."""
        generator_network = LinearGenerator()
        generator = generator_module.build_generator(generator_network)
        self.assertIsInstance(generator, generator_module.DNNGenerator)
        generator = generator_module.build_generator(generator_network, frozen=True)
        self.assertIsInstance(generator, generator_module.FrozenGenerator)


if __name__ == "__main__":
    unittest.main()
