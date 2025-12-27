# __init__.py
"""Top-level package for irs-asset-fifo-calculator."""
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .calculate_taxes import (
    AssetData,
    FifoLot,
    main,
    run_fifo_pipeline,
)

try:
    # Use the distribution name from your pyproject.toml
    __version__ = _version("irs-asset-fifo-calculator")
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed (e.g., running from a source checkout without `pip install -e .`)
    __version__ = "0.0.0"

__all__ = ["AssetData", "FifoLot", "main", "run_fifo_pipeline"]
