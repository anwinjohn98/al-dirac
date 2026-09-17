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
    details: dict[str, float] | None = None


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


@dataclass(frozen=True)
class ModelErrorStoppingCriteria:
    # Checked against the worst (by default) ensemble member's own
    # validation-set error, not committee disagreement -- a direct measure
    # of model accuracy, matching how papers usually report energy/force
    # MAE, rather than a proxy like force_max_uncertainty.
    force_threshold: float | None = 0.05  # eV/A (50 meV/A)
    energy_threshold: float | None = 0.005  # eV/atom (5 meV/atom)
    statistic: str = "max"

    def _reduce(self, values: list[float]) -> float:
        array = np.asarray(values, dtype=float)
        if self.statistic == "max":
            return float(np.max(array))
        if self.statistic == "mean":
            return float(np.mean(array))
        if self.statistic == "median":
            return float(np.median(array))
        raise ValueError("statistic must be max, mean, or median.")

    def check(self, models: list[Any]) -> StopDecision:
        if self.force_threshold is None and self.energy_threshold is None:
            return StopDecision(should_stop=False)
        if self.statistic not in {"max", "mean", "median"}:
            raise ValueError("statistic must be max, mean, or median.")

        force_values: list[float] = []
        energy_values: list[float] = []
        for model in models:
            metrics_fn = getattr(model, "latest_validation_metrics", None)
            if metrics_fn is None:
                continue
            metrics = metrics_fn()
            if metrics is None:
                continue
            if metrics.get("mae_f") is not None:
                force_values.append(float(metrics["mae_f"]))
            if metrics.get("mae_e_per_atom") is not None:
                energy_values.append(float(metrics["mae_e_per_atom"]))

        force_value = self._reduce(force_values) if force_values else None
        energy_value = self._reduce(energy_values) if energy_values else None

        # A threshold that's set but has no data to check against is treated
        # as not-yet-satisfied (conservative), not as trivially passing.
        force_ok = self.force_threshold is None or (
            force_value is not None and force_value <= self.force_threshold
        )
        energy_ok = self.energy_threshold is None or (
            energy_value is not None and energy_value <= self.energy_threshold
        )
        details = {"force_mae": force_value, "energy_mae_per_atom": energy_value}

        if force_ok and energy_ok:
            reason_parts = []
            if self.force_threshold is not None:
                reason_parts.append(
                    f"{self.statistic}_force_mae={force_value:.6g} <= "
                    f"force_threshold={self.force_threshold:.6g} eV/A"
                )
            if self.energy_threshold is not None:
                reason_parts.append(
                    f"{self.statistic}_energy_mae_per_atom={energy_value:.6g} <= "
                    f"energy_threshold={self.energy_threshold:.6g} eV/atom"
                )
            return StopDecision(
                should_stop=True,
                reason=" and ".join(reason_parts),
                value=force_value,
                details=details,
            )

        return StopDecision(should_stop=False, value=force_value, details=details)
