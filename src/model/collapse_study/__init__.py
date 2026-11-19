"""Regularizers and attention-backend routing used by shared-BN model."""

from .regularizers import CanonicalCapacityMatching, GaussianQQRegularizer, VISRegularizer
from .sdpa import LayerwiseSDPARouter, configure_training_sdpa, resolve_layer_backends

__all__ = ["CanonicalCapacityMatching", "GaussianQQRegularizer", "VISRegularizer",
           "LayerwiseSDPARouter", "configure_training_sdpa", "resolve_layer_backends"]
