from __future__ import annotations

from typing import Any

from al_dirac.selection.base import BaseSelector
from al_dirac.selection.score_expression import ScoreExpressionEvaluator


class EnsembleUncertaintySelector(BaseSelector):
    def __init__(
        self,
        score_expression: str = "force_max_uncertainty",
        larger_is_better: bool = True,
        min_score: float | None = None,
    ) -> None:
        super().__init__(selector_name="ensemble_uncertainty")
        self.score_expression = score_expression
        self.larger_is_better = larger_is_better
        self.min_score = min_score

    def _evaluate_expression(self, record: dict[str, Any]) -> float | None:
        return ScoreExpressionEvaluator(score_expression=self.score_expression).evaluate(record)

    def select(
        self,
        records: list[dict[str, Any]],
        k: int | None,
    ) -> list[dict[str, Any]]:
        if k is not None and k < 0:
            raise ValueError("k must be non-negative or None.")
        if k == 0:
            return []

        scored_records: list[dict[str, Any]] = []
        for record in records:
            score = self._evaluate_expression(record)
            if score is None:
                continue
            if self.min_score is not None and score < self.min_score:
                continue

            updated_record = dict(record)
            updated_record["selection_score"] = score
            scored_records.append(updated_record)

        selected_records = sorted(
            scored_records,
            key=lambda record: record["selection_score"],
            reverse=self.larger_is_better,
        )
        return selected_records if k is None else selected_records[:k]
