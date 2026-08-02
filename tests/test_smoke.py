import torch
from whos_that_pokemon.models import UNet, PokemonClassifier


def test_unet_forward_runs():
    model = UNet(base=8)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_classifier_forward_runs():
    clf = PokemonClassifier(num_classes=10, base=4)
    out = clf(torch.randn(2, 1, 64, 64))
    assert out.shape == (2, 10)