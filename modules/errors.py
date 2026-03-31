"""User-facing error types for data loading and validation."""


class DatasetLoadError(ValueError):
    """Raised when a dataset cannot be loaded or is not usable as CSV."""
