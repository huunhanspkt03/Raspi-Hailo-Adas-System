"""
Multi-model inference wrapper for ADAS pipeline.
Auto-selects Hailo-8L or ONNX Runtime fallback.
"""
from __future__ import annotations

import os

from loguru import logger

from pipeline.hailo_backend import HailoInference
from pipeline.onnx_backend import OnnxFallbackInference

__all__ = ["HailoInference", "OnnxFallbackInference", "create_inference"]


def create_inference(models: dict, force_onnx: bool = False):
    """Factory: auto-selects Hailo or ONNX fallback based on /dev/hailo0."""
    if force_onnx or not os.path.exists("/dev/hailo0"):
        logger.warning("Hailo device not found — using ONNX Runtime fallback")
        return OnnxFallbackInference(models)
    return HailoInference(models)
