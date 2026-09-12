from __future__ import annotations

from typing import Any

import numpy as np


class SimilarityGraph:
    def __init__(self, eps: float) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.eps = eps

    def graph_from_distance_matrix(
        self,
        distance_matrix: np.ndarray,
        **kwargs: Any,
    ) -> dict[int, set[int]]:
        distance_matrix = np.asarray(distance_matrix, dtype=float)

        if distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2D.")
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square.")

        n = distance_matrix.shape[0]
        graph: dict[int, set[int]] = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] < self.eps:
                    graph[i].add(j)
                    graph[j].add(i)

        return graph

    def connected_components(
        self,
        graph: dict[int, set[int]],
        **kwargs: Any,
    ) -> list[list[int]]:
        visited: set[int] = set()
        components: list[list[int]] = []

        for start in graph:
            if start in visited:
                continue

            stack = [start]
            component: list[int] = []

            while stack:
                node = stack.pop()
                if node in visited:
                    continue

                visited.add(node)
                component.append(node)
                stack.extend(graph[node] - visited)

            components.append(sorted(component))

        return components

    def cluster_from_distance_matrix(
        self,
        distance_matrix: np.ndarray,
        **kwargs: Any,
    ) -> list[list[int]]:
        graph = self.graph_from_distance_matrix(distance_matrix)
        return self.connected_components(graph)
