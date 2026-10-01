"""
SolarCast ML Package

Import submodules directly rather than using top-level eager imports
to avoid circular dependencies and heavy import chains at package load time.

Usage:
    from ml.config import Config, get_device
    from ml.models.solar_cnn import SolarCNN
    from ml.datasets.solar_dataset import SolarDataset
    from ml.training.trainer import SolarTrainer
    from ml.inference.predictor import SolarPredictor
"""

__version__ = "1.0.0"
