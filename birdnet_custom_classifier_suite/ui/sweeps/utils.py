"""Utility functions for sweep parameter parsing and validation."""

from __future__ import annotations


def parse_num_list(s: str, cast) -> list:
    """Parse comma-separated numeric string into list."""
    vals = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(cast(tok))
        except Exception:
            pass
    return vals


def parse_float(s: str, default: float) -> float:
    """Parse string to float with fallback default."""
    try:
        return float(s)
    except Exception:
        return default


def validate_quality(quality_combos: list[list[str]]) -> tuple[bool, list[str]]:
    """
    Validate quality combinations.
    
    Returns:
        (is_valid, invalid_values)
    """
    valid_quality = {"high", "medium", "low"}
    invalid_qualities = []
    for combo in quality_combos:
        for q in combo:
            if q not in valid_quality:
                invalid_qualities.append(q)
    return len(invalid_qualities) == 0, invalid_qualities


def calculate_config_count(axes: dict) -> int:
    """Calculate total number of configurations from axes."""
    config_count = 1
    for axis_values in axes.values():
        config_count *= len(axis_values)
    return config_count
