# anomaly_engine/datasets/__init__.py
"""Datasets module"""

# Import available dataset classes
try:
    # Try to import from sam_dataset_loader first
    from .dataset_loader import VideoAnomalyDataset
    __all__ = ['VideoAnomalyDataset']
except ImportError:
    try:
        # Fallback to dataset_loader if sam_dataset_loader doesn't exist
        from .dataset_loader import VideoAnomalyDataset
        __all__ = ['VideoAnomalyDataset']
    except ImportError:
        __all__ = []

# Note: DatasetSplit was not found in dataset_loader.py, so we're not importing it
# If you need DatasetSplit, you'll need to define it in one of your dataset files