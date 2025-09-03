# utils/config.py
# Loads configs/base.yaml
# Optionally loads configs/modelXX.yaml
# Merges the two (model overrides win)
# Passes a single config dict to the scripts

import yaml
from pathlib import Path
from copy import deepcopy

def load_config(base_path: str, override_path: str = None) -> dict:
    """
    Load a base YAML config and optionally apply overrides from another YAML.
    Returns a merged dict.
    """
    base = {}
    override = {}

    base_file = Path(base_path)
    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}

    if override_path:
        override_file = Path(override_path)
        if override_file.exists():
            with open(override_file, "r", encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}

    return merge_dicts(base, override)


def merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge two dicts (override wins)."""
    result = deepcopy(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result
