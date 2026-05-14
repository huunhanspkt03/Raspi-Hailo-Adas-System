"""ONNX Runtime CPU fallback inference backend for x86 dev machines."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from loguru import logger

YOLO26_ONNX_CANDIDATES = (
    "src/models/yolo26_adas_2404.onnx",
    "checkpoints/Object_Detection/yolo2404.onnx",
    "src/models/yolo26_adas_1404.onnx",
    "checkpoints/Object_Detection/yolo26_1404.onnx",
)

LANE_SEGMENTATION_ONNX_CANDIDATES = (
    "src/models/yolo26s_seg.onnx",
    "checkpoints/YOLO26s_seg/yolo26s_seg.onnx",
    "src/models/yolo26n_seg.onnx",
)


class OnnxFallbackInference:
    """
    ONNX Runtime CPU fallback for x86 dev machines.
    Same interface as HailoInference — swap with zero code changes.
    """

    def __init__(self, models: dict):
        self.model_paths = {
            name: self._resolve_model_path(name, path)
            for name, path in models.items()
        }
        self._sessions = {}
        self._dll_dirs = []

    def _add_nvidia_dll_dirs(self) -> None:
        try:
            import nvidia
        except ImportError:
            return

        nvidia_root = Path(nvidia.__file__).resolve().parent
        dll_dirs = [
            nvidia_root / package / "bin"
            for package in (
                "cublas",
                "cuda_nvrtc",
                "cuda_runtime",
                "cudnn",
                "cufft",
                "curand",
                "nvjitlink",
            )
        ]
        existing = os.environ.get("PATH", "")
        for dll_dir in dll_dirs:
            if not dll_dir.exists():
                continue
            dll_dir_str = str(dll_dir)
            if hasattr(os, "add_dll_directory"):
                self._dll_dirs.append(os.add_dll_directory(dll_dir_str))
            if dll_dir_str not in existing:
                os.environ["PATH"] = dll_dir_str + os.pathsep + os.environ.get("PATH", "")

    @staticmethod
    def _resolve_candidate_path(configured_path: Path, candidates: tuple[str, ...]) -> Path:
        if configured_path.exists():
            return configured_path

        candidate_keys = {Path(candidate).as_posix() for candidate in candidates}
        if configured_path.as_posix() not in candidate_keys:
            return configured_path

        for candidate in candidates:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path
        return configured_path

    @staticmethod
    def _resolve_model_path(model_name: str, configured_path: str) -> str:
        """Resolve ONNX fallback path for a model.

        For lane segmentation we resolve a YOLO26s_seg ONNX from a fixed
        candidate list, while still honoring explicit --lane-onnx overrides.
        """
        raw_path = Path(configured_path)
        if raw_path.suffix.lower() == ".onnx":
            if model_name == "yolo26":
                return str(
                    OnnxFallbackInference._resolve_candidate_path(
                        raw_path, YOLO26_ONNX_CANDIDATES,
                    )
                )
            if model_name == "lane_segmentation":
                return str(
                    OnnxFallbackInference._resolve_candidate_path(
                        raw_path, LANE_SEGMENTATION_ONNX_CANDIDATES,
                    )
                )
            return str(raw_path)

        default_onnx = raw_path.with_suffix(".onnx")
        if model_name == "yolo26":
            return str(
                OnnxFallbackInference._resolve_candidate_path(
                    default_onnx, YOLO26_ONNX_CANDIDATES,
                )
            )
        if model_name != "lane_segmentation":
            return str(default_onnx)

        return str(
            OnnxFallbackInference._resolve_candidate_path(
                default_onnx, LANE_SEGMENTATION_ONNX_CANDIDATES,
            )
        )

    def __enter__(self):
        import onnxruntime as ort
        self._add_nvidia_dll_dirs()
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        for name, path in self.model_paths.items():
            if Path(path).exists():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                available = ort.get_available_providers()
                active_providers = [provider for provider in providers if provider in available]
                self._sessions[name] = ort.InferenceSession(path, providers=active_providers)
                session_providers = self._sessions[name].get_providers()
                logger.info(f"[ONNX] '{name}' loaded from {path} providers={session_providers}")
            else:
                logger.warning(f"[ONNX] model not found: {path}")
        return self

    def __exit__(self, *_):
        self._sessions.clear()

    @staticmethod
    def _prepare_input(sess, input_tensor: np.ndarray) -> tuple[str, np.ndarray]:
        inp = input_tensor.astype(np.float32) / 255.0
        if inp.ndim == 3:
            inp = inp[None]

        meta = sess.get_inputs()[0]
        shape = tuple(int(dim) if isinstance(dim, int) else -1 for dim in meta.shape)
        expects_nchw = len(shape) == 4 and shape[1] in (1, 3)
        expects_nhwc = len(shape) == 4 and shape[-1] in (1, 3)

        if expects_nchw and inp.ndim == 4 and inp.shape[-1] in (1, 3):
            inp = inp.transpose(0, 3, 1, 2)
        elif expects_nhwc:
            # Common host/dev path: preprocess already returns HWC RGB uint8.
            pass

        return meta.name, inp

    def infer(self, model_name: str, input_tensor: np.ndarray) -> dict:
        if model_name not in self._sessions:
            raise RuntimeError(
                f"[ONNX] model '{model_name}' was not loaded. "
                f"Loaded: {list(self._sessions)}. "
                f"Check features config and model file paths."
            )
        sess = self._sessions[model_name]
        input_name, inp = self._prepare_input(sess, input_tensor)
        outs = sess.run(None, {input_name: inp})
        return dict(zip([o.name for o in sess.get_outputs()], outs))

    def get_input_shape(self, model_name: str) -> tuple[int, ...]:
        if model_name not in self._sessions:
            raise KeyError(f"Unknown model: {model_name}")
        shape = self._sessions[model_name].get_inputs()[0].shape
        return tuple(int(dim) if isinstance(dim, int) else -1 for dim in shape)

    def get_output_shapes(self, model_name: str) -> list[tuple[int, ...]]:
        if model_name not in self._sessions:
            raise KeyError(f"Unknown model: {model_name}")
        outputs = self._sessions[model_name].get_outputs()
        return [
            tuple(int(dim) if isinstance(dim, int) else -1 for dim in output.shape)
            for output in outputs
        ]
