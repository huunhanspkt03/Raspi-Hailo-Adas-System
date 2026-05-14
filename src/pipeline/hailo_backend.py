"""HailoRT inference backend for ADAS pipeline (Hailo-8L AI Hat+)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from loguru import logger


class HailoInference:
    """
    Loads and runs multiple HEF models on the Hailo-8L AI Hat+.

    Usage:
        models = {
            "yolo26":    "src/models/yolo26_adas_2404.hef",
            "lane_segmentation": "src/models/yolo26s_seg.hef",
        }
        with HailoInference(models) as h:
            det_out  = h.infer("yolo26",    yolo_tensor)
            lane_out = h.infer("lane_segmentation", lane_tensor)
    """

    def __init__(self, models: dict):
        for name, path in models.items():
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"HEF not found for '{name}': {path}\n"
                    "Run scripts/compile_hailo.sh on x86 host first,\n"
                    "then scp .hef files to src/models/ on the Pi."
                )
        self.model_paths = models
        self._target = None
        self._network_groups = {}
        self._input_infos = {}
        self._input_names = {}
        self._output_infos = {}
        self._input_vstream_params = {}
        self._output_vstream_params = {}
        self._pipelines = {}
        self._pipeline_contexts = {}
        self._logged_debug_outputs = set()

    def __enter__(self):
        self._init_device()
        return self

    def __exit__(self, *_):
        self._cleanup()

    def _init_device(self):
        try:
            from hailo_platform import (
                VDevice, HEF, ConfigureParams,
                HailoStreamInterface, HailoSchedulingAlgorithm,
                InputVStreamParams, OutputVStreamParams, InferVStreams,
                FormatType,
            )
        except ImportError:
            raise ImportError(
                "hailo_platform not found. Install hailo-all on Raspberry Pi:\n"
                "  sudo apt install hailo-all && sudo reboot"
            )

        try:
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self._target = VDevice(params)
        except Exception as e:
            if "No device" in str(e) or "hailo0" in str(e).lower():
                raise RuntimeError(
                    "Hailo AI Hat+ not detected!\n"
                    "  Check: ls /dev/hailo*\n"
                    "  Check: lspci | grep -i hailo"
                )
            raise

        for name, hef_path in self.model_paths.items():
            hef = HEF(hef_path)
            cfg = ConfigureParams.create_from_hef(
                hef=hef, interface=HailoStreamInterface.PCIe
            )
            ng = self._target.configure(hef, cfg)[0]
            self._network_groups[name] = ng
            input_infos = hef.get_input_vstream_infos()
            output_infos = hef.get_output_vstream_infos()
            self._input_infos[name] = input_infos
            self._output_infos[name] = output_infos
            self._input_names[name] = input_infos[0].name
            self._input_vstream_params[name] = InputVStreamParams.make(ng)
            try:
                self._output_vstream_params[name] = OutputVStreamParams.make(
                    ng,
                    quantized=False,
                    format_type=FormatType.FLOAT32,
                )
            except Exception as exc:
                logger.warning(
                    "Model '{}' could not request FLOAT32 Hailo outputs ({}). Falling back to default stream params.",
                    name,
                    exc,
                )
                self._output_vstream_params[name] = OutputVStreamParams.make(ng)
            pipeline_ctx = InferVStreams(
                ng,
                self._input_vstream_params[name],
                self._output_vstream_params[name],
            )
            self._pipeline_contexts[name] = pipeline_ctx
            self._pipelines[name] = pipeline_ctx.__enter__()
            logger.info(f"Model '{name}' loaded | input: {tuple(input_infos[0].shape)}")

    def infer(self, model_name: str, input_tensor: np.ndarray) -> dict:
        """Run inference. input_tensor must be uint8 (H,W,C)."""
        if model_name not in self._pipelines:
            raise RuntimeError(
                f"[Hailo] model '{model_name}' was not loaded. "
                f"Loaded: {list(self._pipelines)}. "
                f"Check features config and HEF paths."
            )
        input_data = {self._input_names[model_name]: input_tensor[None, ...]}
        outputs = self._pipelines[model_name].infer(input_data)
        if os.getenv("ADAS_DEBUG_HAILO_OUTPUTS") == "1" and model_name not in self._logged_debug_outputs:
            self._logged_debug_outputs.add(model_name)
            for out_name, out_value in outputs.items():
                arr = np.asarray(out_value)
                logger.info(
                    "[Hailo debug] model={} output={} shape={} dtype={} min={:.5f} max={:.5f} mean={:.5f}",
                    model_name,
                    out_name,
                    tuple(arr.shape),
                    arr.dtype,
                    float(arr.min()),
                    float(arr.max()),
                    float(arr.mean()),
                )
        return outputs

    def get_input_shape(self, model_name: str) -> tuple[int, ...]:
        if model_name not in self._input_infos:
            raise KeyError(f"Unknown model: {model_name}")
        return tuple(self._input_infos[model_name][0].shape)

    def get_output_shapes(self, model_name: str) -> list[tuple[int, ...]]:
        if model_name not in self._output_infos:
            raise KeyError(f"Unknown model: {model_name}")
        return [tuple(info.shape) for info in self._output_infos[model_name]]

    def _cleanup(self):
        for name, pipeline_ctx in list(self._pipeline_contexts.items()):
            try:
                pipeline_ctx.__exit__(None, None, None)
            except Exception as exc:
                logger.warning(f"Failed to close Hailo pipeline '{name}': {exc}")
        self._pipelines.clear()
        self._pipeline_contexts.clear()
        if self._target:
            del self._target
            self._target = None
        logger.info("Hailo VDevice released.")
