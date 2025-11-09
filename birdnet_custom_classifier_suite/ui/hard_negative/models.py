"""Data models for hard-negative mining workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st


class ModelSource(Enum):
    """Source type for model selection."""
    EXPERIMENT_CANONICAL = "experiment_canonical"
    EXPERIMENT_MODEL_FILE = "experiment_model_file"
    UPLOADED_FILE = "uploaded_file"


class SelectionMode(Enum):
    """File selection method."""
    TOP_K = "top_k"
    TOP_PERCENT = "top_pct"
    THRESHOLD = "threshold"
    MANUAL_INDEX = "manual_index"
    MANUAL_PICK = "manual_pick"


class ExportMethod(Enum):
    """Export destination type."""
    FOLDER = "folder"
    ZIP = "zip"


class LinkMethod(Enum):
    """File operation type for folder export."""
    COPY = "copy"
    HARDLINK = "hardlink"
    SYMLINK = "symlink"


@dataclass
class InferenceConfig:
    """Configuration for running inference."""
    input_dir: Path
    model_source: ModelSource
    target_species: str = "RADR"
    selected_experiment: Optional[str] = None
    model_path: Optional[Path] = None
    uploaded_model: Optional[st.runtime.uploaded_file_manager.UploadedFile] = None
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.input_dir.exists():
            errors.append(f"Input directory not found: {self.input_dir}")
        elif not self.input_dir.is_dir():
            errors.append(f"Not a directory: {self.input_dir}")
        elif not any(self.input_dir.iterdir()):
            errors.append(f"Input directory is empty: {self.input_dir}")
        
        if self.model_source == ModelSource.EXPERIMENT_CANONICAL:
            if not self.selected_experiment or self.selected_experiment == '(none)':
                errors.append("No experiment selected")
        
        elif self.model_source == ModelSource.EXPERIMENT_MODEL_FILE:
            if not self.model_path:
                errors.append("No model file selected")
            elif not self.model_path.exists():
                errors.append(f"Model file not found: {self.model_path}")
        
        elif self.model_source == ModelSource.UPLOADED_FILE:
            if not self.uploaded_model:
                errors.append("No model file uploaded")
        
        if not self.target_species or not self.target_species.strip():
            errors.append("Target species label cannot be empty")
        
        return errors


@dataclass
class InferenceResult:
    """Result of an inference operation."""
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    output_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    command: Optional[List[str]] = None
    errors: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)


@dataclass
class SelectionConfig:
    """Configuration for file selection."""
    mode: SelectionMode
    top_k: int = 50
    top_percent: float = 10.0
    threshold: float = 0.5
    manual_indices: List[int] = field(default_factory=list)
    
    def get_selected_indices(self, df_size: int) -> List[int]:
        """Calculate which indices to select based on mode and parameters."""
        if self.mode == SelectionMode.TOP_K:
            return list(range(min(self.top_k, df_size)))
        
        elif self.mode == SelectionMode.TOP_PERCENT:
            count = max(1, int(df_size * (self.top_percent / 100.0)))
            return list(range(min(count, df_size)))
        
        elif self.mode in (SelectionMode.MANUAL_INDEX, SelectionMode.MANUAL_PICK):
            return [i for i in self.manual_indices if i < df_size]
        
        else:  # THRESHOLD - handled differently, returns all matching
            return []


@dataclass
class ExportConfig:
    """Configuration for exporting selected files."""
    method: ExportMethod
    output_root: Path
    subfolder: str = "hardneg_manual"
    link_method: LinkMethod = LinkMethod.COPY
    zip_filename: str = "hardneg_selection.zip"
    
    def get_destination(self) -> Path:
        """Get the destination path for folder export."""
        return self.output_root / self.subfolder


@dataclass
class ExportResult:
    """Result of an export operation."""
    success: bool
    files_exported: int = 0
    total_files: int = 0
    destination: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
