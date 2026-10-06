import unittest

import torch
import torch.nn as nn

from bdpy.dl.torch import models


def _removeprefix(text: str, prefix: str) -> str:
    """Remove prefix from text. (Workaround for Python 3.8)"""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


class MockModule(nn.Module):
    def __init__(self):
        super(MockModule, self).__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 1, 3),
            nn.Conv2d(1, 1, 3),
            nn.Module(),
            nn.Sequential(
                nn.Conv2d(1, 1, 4),
                nn.Conv2d(1, 1, 8),
            )
        )
        inner_network = self.layers[-2]
        inner_network.features = nn.Sequential(
            nn.Conv2d(1, 1, 5),
            nn.Conv2d(1, 1, 5)
        )


class TestLayerMap(unittest.TestCase):
    def setUp(self):
        self.kv_pairs = [
            {'net': 'vgg19', 'payload': {'key': 'fc6', 'value': 'classifier[0]'}},
            {'net': 'vgg19', 'payload': {'key': 'conv5_4', 'value': 'features[34]'}},
            {'net': 'alexnet', 'payload': {'key': 'fc6', 'value': 'classifier[0]'}},
            {'net': 'alexnet', 'payload': {'key': 'conv5', 'value': 'features[12]'}}
        ]

    def test_layer_map(self):
        for kv_pair in self.kv_pairs:
            expected = kv_pair['payload']
            output = models.layer_map(kv_pair['net'])
            self.assertIsInstance(output, dict)
            self.assertEqual(output[expected['key']], expected['value'])


class TestParseLayerName(unittest.TestCase):
    def setUp(self):
        self.mock = MockModule()
        self.accessors = [
            {'name': 'layer1', 'type': nn.Linear, 'attrs': {'in_features': 10, 'out_features': 10}},
            {'name': 'layers[0]', 'type': nn.Conv2d, 'attrs': {'kernel_size': (3, 3)}},
            {'name': 'layers[1]', 'type': nn.Conv2d, 'attrs': {'kernel_size': (3, 3)}},
            {'name': 'layers[2].features[0]', 'type': nn.Conv2d, 'attrs': {'kernel_size': (5, 5)}},
            {'name': 'layers[3][0]', 'type': nn.Conv2d, 'attrs': {'kernel_size': (4, 4)}},
            {'name': 'layers[3][1]', 'type': nn.Conv2d, 'attrs': {'kernel_size': (8, 8)}}
        ]

    def test_parse_layer_name(self):
        for accessor in self.accessors:
            layer = models._parse_layer_name(self.mock, accessor['name'])
            self.assertIsInstance(layer, accessor['type'])
            for attr, value in accessor['attrs'].items():
                self.assertEqual(getattr(layer, attr), value)

        # Test non-existing layer access
        self.assertRaises(
            ValueError, models._parse_layer_name, self.mock, 'not_existing_layer')
        # Test invalid layer access
        self.assertRaises(
            ValueError, models._parse_layer_name, self.mock, 'layers["key"]')

    def test_parse_layer_name_for_sequential(self):
        """Test _parse_layer_name for nn.Sequential.

        nn.Sequential is a special case because the submodules are directly
        accessible like a list. For example, `model[0]` will return the first
        module in the model.
        """
        sequential_module = self.mock.layers
        accessors = [accessor for accessor in self.accessors if accessor['name'].startswith('layers')]
        for accessor in accessors:
            accsessor_key = _removeprefix(accessor['name'], 'layers')
            layer = models._parse_layer_name(sequential_module, accsessor_key)
            self.assertIsInstance(layer, accessor['type'])
            for attr, value in accessor['attrs'].items():
                self.assertEqual(getattr(layer, attr), value)


class _ProvidedModelTest:
    """Forward test for one provided model, built once per class.

    Construction, not forward, dominates the cost, so the model is built in
    setUpClass (seeded without touching the global RNG) and released in
    tearDownClass -- a class attribute would otherwise keep it alive for the
    rest of the session. Subclasses set `model_cls`, `input_shape` and
    `output_shape`; this mixin is not a TestCase, so it is not collected itself.
    """

    model_cls: type
    input_shape: tuple
    output_shape: tuple

    @classmethod
    def setUpClass(cls):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            cls.model = cls.model_cls()

    @classmethod
    def tearDownClass(cls):
        del cls.model

    def test_forward(self):
        x = torch.rand(self.input_shape, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            output = self.model(x)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, self.output_shape)

    def _assert_layer_map_resolves(self, net):
        for layer_name in models.layer_map(net).values():
            self.assertIsInstance(
                models._parse_layer_name(self.model, layer_name), nn.Module)


# NOTE: VGG19 and AlexNet end in AdaptiveAvgPool2d, so smaller inputs would
#       also run. 224x224 is kept because it is the resolution the models are
#       used at, where the adaptive pool is the identity -- a smaller input
#       takes the upsampling path instead.


class TestVGG19(_ProvidedModelTest, unittest.TestCase):
    model_cls = models.VGG19
    input_shape = (1, 3, 224, 224)
    output_shape = (1, 1000)

    def test_layer_access(self):
        self._assert_layer_map_resolves('vgg19')


class TestAlexNet(_ProvidedModelTest, unittest.TestCase):
    model_cls = models.AlexNet
    input_shape = (1, 3, 224, 224)
    output_shape = (1, 1000)

    def test_layer_access(self):
        self._assert_layer_map_resolves('alexnet')


class TestAlexNetGenerator(_ProvidedModelTest, unittest.TestCase):
    model_cls = models.AlexNetGenerator
    # A feature vector, not an image: 4096 is what defc7 expects.
    input_shape = (1, 4096)
    output_shape = (1, 3, 256, 256)


if __name__ == '__main__':
    unittest.main()