from __future__ import annotations

import random
from typing import Any

from ase import Atoms

from al_dirac.curator.base import (
    BaseStructureCuration,
    CurationResult,
    CurationStageReport,
)


class RandomCurator(BaseStructureCuration):
    def __init__(
        self,
        fraction: float | None = None,
        k: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(method_name="random_curate")
        if (fraction is None) == (k is None):
            raise ValueError("Exactly one of fraction or k must be provided.")
        if fraction is not None and not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must satisfy 0 < fraction <= 1.")
        if k is not None and k < 0:
            raise ValueError("k must be non-negative.")

        self.fraction = fraction
        self.k = k
        self.seed = seed

    def _n_keep(self, n: int) -> int:
        if self.k is not None:
            return min(self.k, n)
        return min(max(1, round(n * self.fraction)), n)

    def curate(self, structures: list[Atoms], **kwargs: Any) -> list[Atoms]:
        return random.Random(self.seed).sample(structures, self._n_keep(len(structures)))

    def curate_records(
        self,
        records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CurationResult:
        n_keep = self._n_keep(len(records))
        kept_indices = set(random.Random(self.seed).sample(range(len(records)), n_keep))

        kept_records: list[dict[str, Any]] = []
        removed_records: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if index in kept_indices:
                kept_records.append(
                    self._with_curation_decision(record, {"decision": "kept"})
                )
            else:
                removed_records.append(
                    self._with_curation_decision(
                        record,
                        {"decision": "removed", "reason": "random_subsample"},
                    )
                )

        return CurationResult(
            kept_records=kept_records,
            removed_records=removed_records,
            stage_reports=[
                CurationStageReport(
                    stage=self.method_name,
                    input_count=len(records),
                    kept_count=len(kept_records),
                    removed_count=len(removed_records),
                    details={
                        "fraction": self.fraction,
                        "k": self.k,
                        "seed": self.seed,
                    },
                )
            ],
        )
