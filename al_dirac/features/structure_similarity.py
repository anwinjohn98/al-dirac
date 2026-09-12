from __future__ import annotations

from typing import Any

import numpy as np
from dscribe.kernels import REMatchKernel


class StructureSimilarity:
    def __init__(
        self,
        method: str = "rematch",
        alpha: float = 1.0,
        threshold: float = 1e-6,
        metric: str = "linear",
        gamma: float | None = None,
        normalize_descriptors: bool = True,
    ) -> None:
        self.method = method
        self.normalize_descriptors = normalize_descriptors
        self.kernel = self._build_kernel(
            method=method,
            alpha=alpha,
            threshold=threshold,
            metric=metric,
            gamma=gamma,
        )

    def _build_kernel(
        self,
        method: str,
        **kwargs: Any,
    ) -> Any:
        if method == "rematch":
            return REMatchKernel(**kwargs)

        raise ValueError(f"Unsupported structure similarity method: {method}")

    def _normalize(self, descriptors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return descriptors / norms

    def similarity(
        self,
        descriptors_a: np.ndarray,
        descriptors_b: np.ndarray,
        **kwargs: Any,
    ) -> float:
        descriptors_a = self._prepare_descriptors(descriptors_a)
        descriptors_b = self._prepare_descriptors(descriptors_b)

        value = self.kernel.create([descriptors_a, descriptors_b])
        return float(value[0, 1])

    def distance(
        self,
        descriptors_a: np.ndarray,
        descriptors_b: np.ndarray,
        **kwargs: Any,
    ) -> float:
        similarity = self.similarity(descriptors_a, descriptors_b)
        return float(np.sqrt(max(0.0, 2.0 - 2.0 * similarity)))

    def similarity_matrix(
        self,
        descriptor_list: list[np.ndarray],
        **kwargs: Any,
    ) -> np.ndarray:
        descriptors = [
            self._prepare_descriptors(descriptor) for descriptor in descriptor_list
        ]
        return np.asarray(self.kernel.create(descriptors), dtype=float)

    def distance_matrix(
        self,
        descriptor_list: list[np.ndarray],
        **kwargs: Any,
    ) -> np.ndarray:
        similarity = self.similarity_matrix(descriptor_list, **kwargs)
        return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarity))

    def _prepare_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        descriptors = np.asarray(descriptors, dtype=float)
        if not self.normalize_descriptors:
            return descriptors
        return self._normalize(descriptors)
