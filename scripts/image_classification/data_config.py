"""ViT data configuration."""

from typing import Final, Dict, Union

CONFIG: Final[Dict[str, Dict[str, Union[int, bool, float]]]] = {
    "cifar": {
        "alpha": 1.0,
        "logger": False,
        "n_c": 100
    },
    "flowers": {
        "alpha": 1.0,
        "logger": False,
        "n_c": 102
    },
    "resisc": {
        "alpha": 1.0,
        "logger": False,
        "n_c": 45
    },
    "food": {
        "alpha": 0.01,
        "logger": False,
        "n_c": 101
    }
}