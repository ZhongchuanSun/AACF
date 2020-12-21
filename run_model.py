from importlib.util import find_spec
from importlib import import_module
from reckit import Configurator
from reckit import typeassert
import os


def _set_random_seed(seed=2020):
    import numpy as np
    import random
    import tensorflow as tf
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    print("set tensorflow seed")


@typeassert(recommender=str)
def find_recommender(recommender):
    spec_path = ".".join(["model", recommender])
    module = import_module(spec_path) if find_spec(spec_path) else None

    if module is None:
        raise ImportError(f"Recommender: {recommender} not found")

    if hasattr(module, recommender):
        Recommender = getattr(module, recommender)
    else:
        raise ImportError(f"Import {recommender} failed from {module.__file__}!")
    return Recommender


if __name__ == "__main__":
    config = Configurator()
    config.add_config("config.ini", section="config")
    config.parse_cmd()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])
    _set_random_seed(config["seed"])

    Recommender = find_recommender(config.model)

    model_cfg = os.path.join("conf", config.model + ".ini")

    section = "hyperparameters"
    for d_name in ["Gowalla", "Yahoo", "CD", "Kindle", "ML", "LibraryThing"]:
        if d_name.lower() in config["train_file"].lower():
            section = d_name
            break

    config.add_config(model_cfg, section=section, used_as_summary=True)

    recommender = Recommender(config)
    recommender.train_model()
