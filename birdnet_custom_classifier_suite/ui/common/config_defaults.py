"""
Default configuration values and categorization for BirdNET training configs.

Based on external/docs/cli_args documentation.
"""

# Default values from birdnet_analyzer.train
TRAINING_DEFAULTS = {
    # Audio processing
    'fmin': 0,
    'fmax': 15000,
    'audio_speed': 1.0,
    'overlap': 0.0,
    
    # Training parameters
    'threads': 2,
    'batch_size': 32,
    'epochs': 50,
    'val_split': 0.2,
    'learning_rate': 0.0001,
    
    # Model architecture
    'hidden_units': 0,
    'dropout': 0.0,
    
    # Training techniques
    'focal_loss': False,
    'focal_loss_gamma': 2.0,
    'focal_loss_alpha': 0.25,
    'label_smoothing': False,
    'mixup': False,
    
    # Data augmentation
    'upsampling_ratio': 0.0,
    'upsampling_mode': 'repeat',
    
    # Model format
    'model_format': 'tflite',
    'model_save_mode': 'replace',
    
    # Data processing
    'crop_mode': 'center',
}

# Default values from birdnet_analyzer.analyze
ANALYZER_DEFAULTS = {
    'fmin': 0,
    'fmax': 15000,
    'lat': -1,
    'lon': -1,
    'week': -1,
    'sf_thresh': 0.03,
    'sensitivity': 1.0,
    'overlap': 0.0,
    'audio_speed': 1.0,
    'threads': 2,
    'min_conf': 0.25,
    'locale': 'en',
    'batch_size': 1,
    'rtype': 'table',
    'combine_results': False,
    'skip_existing_results': False,
    'merge_consecutive': 1,
    'use_perch': False,
}

# Category mappings for config keys (by base name)
CONFIG_CATEGORIES = {
    'dataset': [
        'crop_mode',
        'val_split',
        'test_data',
        'upsampling_ratio',
        'upsampling_mode',
        'filters.quality',
        'filters.balance',
    ],
    'training': [
        'epochs',
        'batch_size',
        'learning_rate',
        'hidden_units',
        'dropout',
        'focal_loss',
        'focal_loss_gamma',
        'focal_loss_alpha',
        'label_smoothing',
        'mixup',
        'upsample_ratio',
        'upsampling.ratio',
        'upsampling.factor',
        'upsampling.mode',
    ],
    'audio_processing': [
        'fmin',
        'fmax',
        'audio_speed',
        'overlap',
    ],
    'analyzer_args': [
        'lat',
        'lon',
        'week',
        'sf_thresh',
        'sensitivity',
        'min_conf',
        'locale',
        'rtype',
        'combine_results',
        'skip_existing_results',
        'merge_consecutive',
        'use_perch',
    ],
    'system': [
        'threads',
        'model_format',
        'model_save_mode',
        'cache_mode',
        'cache_file',
    ],
}

# Preferred category by dotted prefix (first token)
PREFIX_CATEGORY = {
    'training': 'training',
    'dataset': 'dataset',
    'analyzer_args': 'analyzer_args',
    'analyzer': 'analyzer_args',
    'training_args': 'training',
    'audio': 'audio_processing',
    'system': 'system',
}


def get_default_value(key: str) -> any:
    """Get the default value for a config key."""
    # Support dotted keys by comparing the last segment and common aliases
    base = key.split('.')[-1]
    # Normalize common aliases
    aliases = {
        'focal-loss': 'focal_loss',
        'focal-loss-gamma': 'focal_loss_gamma',
        'focal-loss-alpha': 'focal_loss_alpha',
        'label-smoothing': 'label_smoothing',
        'upsampling_ratio': 'upsampling_ratio',
    }
    base_norm = aliases.get(base, base)
    # Try training defaults first, then analyzer
    return (
        TRAINING_DEFAULTS.get(base_norm,
        ANALYZER_DEFAULTS.get(base_norm))
    )


def get_category(key: str) -> str:
    """Get the category for a config key."""
    # 1) Dotted prefix mapping
    if '.' in key:
        prefix = key.split('.')[0]
        if prefix in PREFIX_CATEGORY:
            return PREFIX_CATEGORY[prefix]

    # 2) Match by known base names (supports nested segments)
    def matches_category(cat_keys):
        for name in cat_keys:
            if key.endswith(name) or ('.' in key and name in key):
                return True
        return False

    for category, keys in CONFIG_CATEGORIES.items():
        if matches_category(keys):
            return category

    return 'other'


def is_default_value(key: str, value: any) -> bool:
    """Check if a config value matches the default."""
    # Multi-valued configs are never considered default
    if isinstance(value, str) and value.startswith('Multiple:'):
        return False

    default = get_default_value(key)
    if default is None:
        return False
    
    # Handle type conversions and near-equality for floats
    try:
        # Handle booleans first to avoid casting True/False to 1.0/0.0
        if isinstance(default, bool):
            # Handle string representations of booleans
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ('true', 'false'):
                    return (lowered == 'true') == default
                # Accept 1/0 as bool
                if lowered in ('1', '0'):
                    return (lowered == '1') == default
                return False
            return bool(value) == default
        # Numeric comparison for ints and floats (tolerant)
        elif isinstance(default, (int, float)):
            v = float(value)
            return abs(v - float(default)) < 1e-6
        else:
            return str(value).strip() == str(default).strip()
    except (ValueError, TypeError):
        return False
