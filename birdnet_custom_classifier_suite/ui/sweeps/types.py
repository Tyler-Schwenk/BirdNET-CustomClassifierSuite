"""Types and state management for sweep UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SweepState:
    """State container for sweep configuration UI."""
    
    # Sweep metadata
    stage: int = 1
    sweep_name: str = ""
    out_dir: str = ""
    
    # Base parameters (fixed across all configs)
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.0005
    dropout: float = 0.25
    upsampling_ratio: float = 0.0
    mixup: bool = False
    label_smoothing: bool = False
    focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
    fmin: int = 0
    fmax: int = 15000
    overlap: float = 0.0
    use_validation: bool = False
    
    # Axes (values to sweep over)
    seeds: list[int] = field(default_factory=lambda: [123])
    use_validation_axis: list[bool] = field(default_factory=lambda: [])
    hidden_units: list[int] = field(default_factory=lambda: [0, 128, 512])
    dropout_axis: list[float] = field(default_factory=lambda: [0.0, 0.25])
    learning_rate_axis: list[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.001])
    quality_combos: list[list[str]] = field(default_factory=lambda: [["high"], ["medium"], ["low"], ["high", "medium"]])
    balance_opts: list[bool] = field(default_factory=lambda: [True])
    upsampling_mode_opts: list[str] = field(default_factory=lambda: ["repeat"])
    upsampling_ratio_axis: list[float] = field(default_factory=lambda: [0.0, 0.25])
    
    # Data composition axes (for curated subsets)
    positive_subset_opts: list[list[str]] = field(default_factory=lambda: [[]])
    negative_subset_opts: list[list[str]] = field(default_factory=lambda: [[]])
    
    def get_base_params_dict(self) -> dict[str, Any]:
        """Convert base parameters to dictionary for sweep generation."""
        return {
            "epochs": self.epochs,
            "upsampling_ratio": self.upsampling_ratio,
            "mixup": self.mixup,
            "label_smoothing": self.label_smoothing,
            "focal-loss": self.focal_loss,
            "focal-loss-gamma": self.focal_loss_gamma,
            "focal-loss-alpha": self.focal_loss_alpha,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "overlap": self.overlap,
            "use_validation": self.use_validation,
        }
    
    def get_axes_dict(self) -> dict[str, Any]:
        """Convert axes to dictionary for sweep generation."""
        axes = {}
        if self.seeds:
            axes["seed"] = self.seeds
        if self.hidden_units:
            axes["hidden_units"] = self.hidden_units
        if self.dropout_axis:
            axes["dropout"] = self.dropout_axis
        if self.learning_rate_axis:
            axes["learning_rate"] = self.learning_rate_axis
        if self.quality_combos:
            axes["quality"] = self.quality_combos
        if self.balance_opts:
            axes["balance"] = self.balance_opts
        if self.upsampling_mode_opts:
            axes["upsampling_mode"] = self.upsampling_mode_opts
        if self.upsampling_ratio_axis:
            axes["upsampling_ratio"] = self.upsampling_ratio_axis
        if self.use_validation_axis:
            axes["use_validation"] = self.use_validation_axis
        if self.positive_subset_opts and self.positive_subset_opts != [[]]:
            axes["positive_subsets"] = self.positive_subset_opts
        if self.negative_subset_opts and self.negative_subset_opts != [[]]:
            axes["negative_subsets"] = self.negative_subset_opts
        return axes
