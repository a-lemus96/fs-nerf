# stdlib modules
import os

# third-party modules
import yaml


def load_or_create_config(path: str, defaults: dict) -> dict:
    """
    Loads a YAML config file, creating it with default values if missing.

    Args:
        path (str): path to the YAML config file
        defaults (dict): default values written to the file if it doesn't
            already exist
    Returns:
        dict: config values loaded from the file
    """
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(defaults, f, sort_keys=False)

    with open(path, "r") as f:
        return yaml.safe_load(f)
