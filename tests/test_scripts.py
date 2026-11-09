"""Script Tests."""

import pytest
import torch as th

from scripts.toy_random_projection.projection import OveraModel, VeraModel


def test_zero_vera():
    """Test VeRA zero initialisation and output."""
    model = VeraModel(embed_dim=768, rank=128)
    base_weight = model.base_layer.weight.data
    assert th.equal(base_weight, th.zeros_like(base_weight))
    x = th.randn((1, 1, 768))
    out = model(x)
    assert th.equal(out, th.zeros_like(out))


@pytest.mark.parametrize("init_scheme", ("f", "w", "h", "m"))
@pytest.mark.parametrize("ut_lambda", ("z", "o", "r"))
def test_zero_overa(init_scheme: str, ut_lambda: str):
    """Test Ortho-VeRA zero init and output.

    Args:
        init_scheme (str): Intialisation scheme.
        ut_lambda (str): Untrained lambda init.
    """
    config = {
        "rank": 128,
        "alpha": 1.0,
        "init_scheme": init_scheme,
        "ut_lambda": ut_lambda,
        "wavelet": "Haar",
        "transform_level": 7,
    }
    model = OveraModel(embed_dim=768, overa_config=config)
    base_weight = th.tensor(model.vera_layer.base_layer.weight.data)
    assert th.equal(base_weight, th.zeros_like(base_weight))
    x = th.randn((1, 1, 768))
    out = model(x)
    assert th.equal(out, th.zeros_like(out))
