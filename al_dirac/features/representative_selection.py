from __future__ import annotations

from typing import Any


class RepresentativeSelection:
    def __init__(
        self, priority: list[tuple[str, str]],
    ) -> None:
        if not priority:
            raise ValueError("priority must not be empty.")

        for field, direction in priority:
            if direction not in {"min", "max"}:
                raise ValueError(
                    f"Invalid direction for '{field}': {direction}. Use 'min' or 'max'."
                )

        self.priority = priority

    def select_representatives(
        self,
        clusters: list[list[int]],
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        representatives: list[dict[str, Any]] = []

        for cluster in clusters:
            cluster_records = [records[index] for index in cluster]
            representative = self._pick_representative(cluster_records)
            representatives.append(representative)

        return representatives

    def _pick_representative(
        self,
        cluster_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._validate_priority_fields(cluster_records)

        ordered = list(cluster_records)

        for field, direction in reversed(self.priority):
            ordered.sort(
                key=lambda record: record[field],
                reverse=(direction == "max"),
            )

        return ordered[0]

    def _validate_priority_fields(
        self,
        cluster_records: list[dict[str, Any]],
    ) -> None:
        for field, _ in self.priority:
            for record in cluster_records:
                if field not in record:
                    raise KeyError(
                        f"Missing priority field '{field}' in representative selection record."
                    )
                if record[field] is None:
                    raise ValueError(
                        f"Priority field '{field}' is None in representative selection record."
                    )
