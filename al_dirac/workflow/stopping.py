from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from al_dirac.selection.score_expression import ScoreExpressionEvaluator


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class UncertaintyStoppingCriteria:
    key: str = "force_max_uncertainty"
    threshold: float | None = None
    statistic: str = "max"
    expression: str | None = None

    def check(self, records: list[dict[str, Any]]) -> StopDecision:
        if self.threshold is None:
            return StopDecision(should_stop=False)
        if self.statistic not in {"max", "mean", "median"}:
            raise ValueError("uncertainty_stop_statistic must be max, mean, or median.")

        evaluator = ScoreExpressionEvaluator(
            score_expression=self.expression or self.key,
        )
        values = np.asarray(
            [
                float(value)
                for record in records
                if (value := evaluator.evaluate(record)) is not None
            ],
            dtype=float,
        )
        if len(values) == 0:
            return StopDecision(should_stop=False)

        if self.statistic == "max":
            value = float(np.max(values))
        elif self.statistic == "mean":
            value = float(np.mean(values))
        else:
            value = float(np.median(values))

        if value <= self.threshold:
            score_name = self.expression or self.key
            return StopDecision(
                should_stop=True,
                reason=(
                    f"{self.statistic}_{score_name}={value:.6g} <= "
                    f"uncertainty_stop_threshold={self.threshold:.6g}"
                ),
                value=value,
            )

        return StopDecision(should_stop=False, value=value)
