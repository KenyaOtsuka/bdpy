"""Seeding helper for tests whose random numbers are drawn inside library code."""

import unittest

import torch


def seed_torch(test_case: unittest.TestCase, seed: int = 0) -> None:
    """Seed torch's global CPU RNG for one test and restore it afterwards.

    Use this where no generator can be passed, e.g. ``nn.Module`` parameter
    initialisation or ``nn.init`` calls inside bdpy. Tensors a test creates
    itself should come from a local ``torch.Generator`` instead. This is
    ``torch.random.fork_rng(devices=[])`` stretched over setUp and the test.
    """
    state = torch.random.get_rng_state()
    test_case.addCleanup(torch.random.set_rng_state, state)
    torch.manual_seed(seed)
