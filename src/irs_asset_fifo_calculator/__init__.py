# __init__.py
"""Top-level package for irs-asset-fifo-calculator."""
from importlib.metadata import PackageNotFoundError, version as _version
from .calculate_taxes import (
    run_fifo_pipeline,
    main,
    AssetData,
    FifoLot,
)

try:
    # Use the distribution name from your pyproject.toml
    __version__ = _version("irs-asset-fifo-calculator")
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed (e.g., running from a source checkout without `pip install -e .`)
    __version__ = "0.0.0"

__all__ = ["run_fifo_pipeline", "main", "AssetData", "FifoLot"]

