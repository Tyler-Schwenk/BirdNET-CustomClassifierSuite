"""Input validation for hard-negative mining."""
from pathlib import Path
from typing import Optional


def validate_input_directory(path: Path) -> Optional[str]:
    """
    Validate input directory exists and contains files.
    
    Args:
        path: Directory path to validate
    
    Returns:
        Error message if invalid, None if valid
    """
    if not path.exists():
        return f"Directory not found: {path}"
    
    if not path.is_dir():
        return f"Not a directory: {path}"
    
    try:
        if not any(path.iterdir()):
            return f"Directory is empty: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    
    return None


def validate_model_file(path: Path, allowed_extensions: tuple = ('.tflite', '.h5', '.pt')) -> Optional[str]:
    """
    Validate model file exists and has correct extension.
    
    Args:
        path: Model file path to validate
        allowed_extensions: Tuple of valid extensions
    
    Returns:
        Error message if invalid, None if valid
    """
    if not path.exists():
        return f"Model file not found: {path}"
    
    if not path.is_file():
        return f"Not a file: {path}"
    
    if path.suffix not in allowed_extensions:
        return f"Invalid extension: {path.suffix}. Expected one of: {allowed_extensions}"
    
    return None


def validate_target_species(label: str) -> Optional[str]:
    """
    Validate species label format.
    
    Args:
        label: Species label to validate
    
    Returns:
        Error message if invalid, None if valid
    """
    if not label or not label.strip():
        return "Species label cannot be empty"
    
    # Allow alphanumeric, hyphens, underscores, and spaces
    cleaned = label.replace('_', '').replace('-', '').replace(' ', '')
    if not cleaned.isalnum():
        return f"Invalid species label: '{label}'. Use alphanumeric characters, spaces, hyphens, or underscores."
    
    return None


def validate_output_directory(path: Path, create_if_missing: bool = True) -> Optional[str]:
    """
    Validate output directory can be created/written to.
    
    Args:
        path: Output directory path
        create_if_missing: Whether to create directory if it doesn't exist
    
    Returns:
        Error message if invalid, None if valid
    """
    if path.exists():
        if not path.is_dir():
            return f"Not a directory: {path}"
        
        # Check write permissions
        test_file = path / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            return f"No write permission: {path}"
        except Exception as e:
            return f"Cannot write to directory: {e}"
    else:
        if create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return f"Cannot create directory: {e}"
        else:
            return f"Directory does not exist: {path}"
    
    return None


def validate_experiment_name(name: str, experiments_root: Path) -> Optional[str]:
    """
    Validate experiment exists.
    
    Args:
        name: Experiment name
        experiments_root: Root directory containing experiments
    
    Returns:
        Error message if invalid, None if valid
    """
    if not name or name == '(none)':
        return "No experiment selected"
    
    exp_dir = experiments_root / name
    if not exp_dir.exists():
        return f"Experiment not found: {name}"
    
    if not exp_dir.is_dir():
        return f"Not a directory: {exp_dir}"
    
    return None
