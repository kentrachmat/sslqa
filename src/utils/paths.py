#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Path management utilities for SSL Augmentation project.

This module provides helper functions for working with file paths
throughout the codebase.
"""

from pathlib import Path
from typing import Optional
import os


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the project root (2 levels up from this file)
    """
    return Path(__file__).parent.parent.parent


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The path (for chaining)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """
    Get the models directory.

    Uses MODELS_DIR environment variable if set, otherwise ./models
    """
    if "MODELS_DIR" in os.environ:
        return Path(os.environ["MODELS_DIR"])
    return get_project_root() / "models"


def get_experiments_dir() -> Path:
    """Get the experiments directory."""
    return get_project_root() / "experiments"


def get_results_dir() -> Path:
    """Get the results directory."""
    return get_project_root() / "results"


def resolve_path(path: str, relative_to: Optional[Path] = None) -> Path:
    """
    Resolve a path that may be relative or absolute.

    Args:
        path: Path string (may contain ~ or be relative)
        relative_to: Base path for relative paths (default: project root)

    Returns:
        Resolved absolute path
    """
    path = Path(path).expanduser()

    if path.is_absolute():
        return path

    if relative_to is None:
        relative_to = get_project_root()

    return (relative_to / path).resolve()


def list_run_directories(dataset: str, pattern: Optional[str] = None) -> list[Path]:
    """
    List training run directories for a dataset.

    Args:
        dataset: Dataset name ('squad' or 'pubmed')
        pattern: Optional glob pattern to filter runs

    Returns:
        List of run directory paths
    """
    base_dir = get_experiments_dir() / "training_runs" / dataset
    if not base_dir.exists():
        return []

    if pattern:
        return sorted(base_dir.glob(f"**/{pattern}"))
    else:
        return sorted([d for d in base_dir.rglob("*") if d.is_dir()])
