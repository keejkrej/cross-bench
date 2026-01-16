"""Configuration management for cross-bench."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import yaml


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        confidence_threshold: Minimum confidence for detections
        prompt_types: List of prompt types to benchmark
        max_samples: Maximum samples to process (None for all)
        output_dir: Directory for saving results
        save_visualizations: Whether to save visualization figures
        visualization_dpi: DPI for saved figures
        device: Device to use ('cuda', 'cpu', or 'auto')
        verbose: Whether to print progress
    """
    confidence_threshold: float = 0.5
    prompt_types: list[str] = field(default_factory=lambda: ["mask", "box", "point"])
    max_samples: Optional[int] = None
    output_dir: Optional[Path] = None
    save_visualizations: bool = True
    visualization_dpi: int = 150
    device: str = "auto"
    verbose: bool = True

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

    @classmethod
    def from_file(cls, config_path: Path | str) -> "BenchmarkConfig":
        """Load configuration from a JSON or YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            BenchmarkConfig instance
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            if config_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(**data)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "prompt_types": self.prompt_types,
            "max_samples": self.max_samples,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "save_visualizations": self.save_visualizations,
            "visualization_dpi": self.visualization_dpi,
            "device": self.device,
            "verbose": self.verbose,
        }

    def save(self, config_path: Path | str) -> None:
        """Save configuration to file.

        Args:
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                json.dump(self.to_dict(), f, indent=2)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.

    Attributes:
        path: Path to dataset (directory or manifest)
        name: Optional name override
        reference_dir: Subdirectory name for reference images
        mask_dir: Subdirectory name for masks
        target_dir: Subdirectory name for target images
        categories: Optional list of categories to include
    """
    path: Path
    name: Optional[str] = None
    reference_dir: str = "reference"
    mask_dir: str = "masks"
    target_dir: str = "target"
    categories: Optional[list[str]] = None

    def __post_init__(self):
        self.path = Path(self.path)

    @classmethod
    def from_file(cls, config_path: Path | str) -> "DatasetConfig":
        """Load from configuration file."""
        config_path = Path(config_path)
        with open(config_path) as f:
            if config_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(**data)
