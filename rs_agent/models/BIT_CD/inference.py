from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BITModelConfig:
    checkpoint_path: Path = (
        Path(__file__).resolve().parent / "checkpoints" / "BIT_LEVIR" / "best_ckpt.pt"
    )
    network_name: str = "base_transformer_pos_s4_dd8_dedim8"
    tile_size: int = 256
    overlap: int = 32
    batch_size: int = 4
    device: str = "auto"


class BITChangeDetector:
    """Package-safe inference adapter around the upstream BIT implementation."""

    def __init__(self, config: BITModelConfig | None = None) -> None:
        self.config = config or BITModelConfig()
        self._torch: Any = None
        self._model: Any = None
        self._device: Any = None

    def predict(self, image_t1: np.ndarray, image_t2: np.ndarray) -> np.ndarray:
        image_t1 = _validate_rgb(image_t1, "image_t1")
        image_t2 = _validate_rgb(image_t2, "image_t2")
        if image_t1.shape != image_t2.shape:
            raise ValueError(
                f"BIT inputs must have identical shapes: {image_t1.shape} != {image_t2.shape}"
            )
        self._ensure_model()
        height, width, _ = image_t1.shape
        tile_size = self.config.tile_size
        overlap = self.config.overlap
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if overlap < 0 or overlap >= tile_size:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size.")

        score_sum = np.zeros((2, height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        windows = list(_tile_windows(height, width, tile_size, overlap))
        for start in range(0, len(windows), self.config.batch_size):
            batch_windows = windows[start : start + self.config.batch_size]
            batch_t1 = []
            batch_t2 = []
            valid_shapes = []
            for row, col, tile_h, tile_w in batch_windows:
                batch_t1.append(_prepare_tile(image_t1[row : row + tile_h, col : col + tile_w], tile_size))
                batch_t2.append(_prepare_tile(image_t2[row : row + tile_h, col : col + tile_w], tile_size))
                valid_shapes.append((tile_h, tile_w))
            logits = self._predict_batch(np.stack(batch_t1), np.stack(batch_t2))
            for index, (row, col, tile_h, tile_w) in enumerate(batch_windows):
                scores = logits[index, :, :tile_h, :tile_w]
                weight = _blend_weight(tile_h, tile_w)
                score_sum[:, row : row + tile_h, col : col + tile_w] += scores * weight
                weight_sum[row : row + tile_h, col : col + tile_w] += weight
        score_sum /= np.maximum(weight_sum[None, :, :], 1e-6)
        return np.argmax(score_sum, axis=0).astype(np.uint8)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "BIT inference requires torch, torchvision and einops. "
                "Install the project ML dependencies first."
            ) from exc
        from .models.networks import define_G

        checkpoint_path = self.config.checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"BIT checkpoint not found: {checkpoint_path}")
        if self.config.device == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.config.device
        device = torch.device(device_name)
        args = type("BITArgs", (), {"net_G": self.config.network_name})()
        model = define_G(args=args, gpu_ids=[])
        # The trusted upstream 2021 checkpoint contains optimizer-era NumPy
        # scalar metadata and therefore cannot use PyTorch 2.6+ weights_only mode.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_G_state_dict"])
        model.to(device)
        model.eval()
        self._torch = torch
        self._model = model
        self._device = device

    def _predict_batch(self, batch_t1: np.ndarray, batch_t2: np.ndarray) -> np.ndarray:
        torch = self._torch
        tensor_t1 = torch.from_numpy(batch_t1).to(self._device)
        tensor_t2 = torch.from_numpy(batch_t2).to(self._device)
        with torch.inference_mode():
            logits = self._model(tensor_t1, tensor_t2)
        return logits.detach().cpu().numpy().astype(np.float32)


def _validate_rgb(image: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must be HxWx3 RGB data, got {array.shape}.")
    array = array.astype(np.float32)
    finite = array[np.isfinite(array)]
    if finite.size and float(finite.max()) > 1.5:
        dtype_max = 255.0 if float(finite.max()) <= 255 else float(finite.max())
        array = array / dtype_max
    return np.clip(np.nan_to_num(array), 0.0, 1.0)


def _prepare_tile(tile: np.ndarray, tile_size: int) -> np.ndarray:
    padded = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
    padded[: tile.shape[0], : tile.shape[1]] = tile
    normalized = (padded - 0.5) / 0.5
    return np.transpose(normalized, (2, 0, 1))


def _tile_windows(height: int, width: int, tile_size: int, overlap: int):
    stride = tile_size - overlap
    rows = _positions(height, tile_size, stride)
    cols = _positions(width, tile_size, stride)
    for row in rows:
        for col in cols:
            yield row, col, min(tile_size, height - row), min(tile_size, width - col)


def _positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _blend_weight(height: int, width: int) -> np.ndarray:
    y = np.hanning(max(height, 3))[:height]
    x = np.hanning(max(width, 3))[:width]
    weight = np.outer(y, x).astype(np.float32)
    return np.maximum(weight, 0.05)
