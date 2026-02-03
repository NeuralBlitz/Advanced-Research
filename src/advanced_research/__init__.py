"""Advanced Research Framework."""

__version__ = "0.1.0"
__author__ = "Advanced Research Team"

from .core.context import ContextInjector
from .core.integrations import (
    LRSIntegration,
    OpencodeIntegration,
    NeuralBlitzIntegration,
)

__all__ = [
    "ContextInjector",
    "LRSIntegration",
    "OpencodeIntegration",
    "NeuralBlitzIntegration",
]
