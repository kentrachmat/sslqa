#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration loader with environment variable substitution.

This module provides a centralized configuration system that:
- Loads YAML configuration files
- Substitutes environment variables (${VAR_NAME})
- Provides easy access to configuration values
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Central configuration manager for SSL Augmentation project."""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration loader.

        Args:
            config_dir: Directory containing YAML config files
        """
        # Get project root (3 levels up from this file)
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / config_dir

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        # Load all configuration files
        self.models = self._load("models.yaml")
        self.training = self._load("training.yaml")
        self.augmentation = self._load("augmentation.yaml")
        self.evaluation = self._load("evaluation.yaml")
        self.paths = self._load("paths.yaml")

        # Substitute environment variables
        self._substitute_env_vars()

    def _load(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, "r") as f:
            return yaml.safe_load(f)

    def _substitute_env_vars(self):
        """Recursively substitute environment variables in all configs."""
        self.models = self._substitute_dict(self.models)
        self.training = self._substitute_dict(self.training)
        self.augmentation = self._substitute_dict(self.augmentation)
        self.evaluation = self._substitute_dict(self.evaluation)
        self.paths = self._substitute_dict(self.paths)

    def _substitute_dict(self, obj: Any) -> Any:
        """
        Recursively substitute ${VAR} with environment variable values.

        Args:
            obj: Object to process (dict, list, str, or other)

        Returns:
            Object with substituted values
        """
        if isinstance(obj, dict):
            return {k: self._substitute_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_dict(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_string(obj)
        else:
            return obj

    def _substitute_string(self, text: str) -> str:
        """
        Substitute environment variables in a string.

        Supports ${VAR_NAME} syntax. If VAR_NAME not found in environment,
        uses defaults from config or returns the original ${VAR_NAME}.

        Args:
            text: String potentially containing ${VAR} patterns

        Returns:
            String with substituted values
        """
        pattern = re.compile(r'\$\{([^}]+)\}')

        def replacer(match):
            var_name = match.group(1)
            # Try environment variable first
            if var_name in os.environ:
                return os.environ[var_name]
            # Try defaults
            if var_name.lower().endswith('_dir'):
                default_key = var_name.lower()
                if 'defaults' in self.models and default_key in self.models['defaults']:
                    return str(self.project_root / self.models['defaults'][default_key])
            if var_name == 'data_root':
                return str(self.project_root / self.paths.get('data_root', './data'))
            if var_name == 'experiments_root':
                return str(self.project_root / self.paths.get('experiments_root', './experiments'))
            # Return original if not found
            return match.group(0)

        return pattern.sub(replacer, text)

    def get_model_path(self, model_key: str) -> Path:
        """
        Get the path to a model by key.

        Args:
            model_key: Model identifier (e.g., 'qwen', 'llama', 'llama_70b')

        Returns:
            Path to the model directory

        Raises:
            KeyError: If model_key not found in configuration
        """
        if model_key not in self.models['models']:
            raise KeyError(f"Model '{model_key}' not found in configuration")

        path_str = self.models['models'][model_key]['path']
        return Path(path_str)

    def get_dataset_path(self, dataset: str, split: str = "train", processed: bool = True) -> Path:
        """
        Get path to a dataset file.

        Args:
            dataset: Dataset name ('squad' or 'pubmed')
            split: Data split ('train', 'dev', 'test')
            processed: Whether to use processed or raw dataset

        Returns:
            Path to the dataset CSV file
        """
        if dataset not in self.paths['datasets']:
            raise KeyError(f"Dataset '{dataset}' not found in configuration")

        dir_key = 'processed' if processed else 'raw'
        base_path = Path(self.paths['datasets'][dataset][dir_key])

        return base_path / f"{split}.csv"

    def get_augmented_path(self, dataset: str, filename: str = "train.csv") -> Path:
        """
        Get path to augmented dataset file.

        Args:
            dataset: Dataset name ('squad' or 'pubmed')
            filename: Filename (default: 'train.csv')

        Returns:
            Path to the augmented data file
        """
        if dataset not in self.paths['datasets']:
            raise KeyError(f"Dataset '{dataset}' not found in configuration")

        base_path = Path(self.paths['datasets'][dataset]['augmented'])
        return base_path / filename

    def get_training_run_dir(self, dataset: str, model: str, config: str, num_samples: int, run_id: str) -> Path:
        """
        Get directory for a training run.

        Args:
            dataset: Dataset name
            model: Model name
            config: Configuration type (baseline, semantic, etc.)
            num_samples: Number of training samples
            run_id: Unique run identifier

        Returns:
            Path to the training run directory
        """
        base_path = Path(self.paths['training_runs'])
        run_name = f"{run_id}_{model}_{config}_N{num_samples}_gpus2"
        return base_path / dataset / model / run_name

    def get_evaluation_results_dir(self, dataset: str) -> Path:
        """
        Get directory for evaluation results.

        Args:
            dataset: Dataset name

        Returns:
            Path to the evaluation results directory
        """
        base_path = Path(self.paths['evaluation_results'])
        return base_path / dataset


# Global configuration instance (lazy loaded)
_config_instance = None


def get_config() -> Config:
    """Get the global configuration instance (singleton pattern)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
