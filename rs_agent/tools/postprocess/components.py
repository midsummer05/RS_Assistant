from __future__ import annotations

from typing import Dict, List

import numpy as np


def component_stats(mask: np.ndarray) -> List[Dict[str, int]]:
    binary = np.asarray(mask).astype(bool)
    height, width = binary.shape[:2]
    visited = np.zeros((height, width), dtype=bool)
    components: List[Dict[str, int]] = []

    for row in range(height):
        for col in range(width):
            if not binary[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            count = 0
            min_row = max_row = row
            min_col = max_col = col
            while stack:
                r, c = stack.pop()
                count += 1
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if nr < 0 or nr >= height or nc < 0 or nc >= width:
                        continue
                    if binary[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                {
                    "pixel_count": count,
                    "min_row": min_row,
                    "max_row": max_row,
                    "min_col": min_col,
                    "max_col": max_col,
                }
            )
    return components


def filter_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    binary = np.asarray(mask).astype(bool)
    height, width = binary.shape[:2]
    visited = np.zeros((height, width), dtype=bool)
    output = np.zeros((height, width), dtype="uint8")

    for row in range(height):
        for col in range(width):
            if not binary[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            pixels = []
            visited[row, col] = True
            while stack:
                r, c = stack.pop()
                pixels.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if nr < 0 or nr >= height or nc < 0 or nc >= width:
                        continue
                    if binary[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if len(pixels) >= min_pixels:
                for r, c in pixels:
                    output[r, c] = 1
    return output

