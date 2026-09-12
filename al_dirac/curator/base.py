from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ase import Atoms


@dataclass
class CurationStageReport:
    stage: str
    input_count: int
    kept_count: int
    removed_count: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CurationResult:
    kept_records: list[dict[str, Any]]
    removed_records: list[dict[str, Any]]
    stage_reports: list[CurationStageReport]


class BaseStructureCuration(ABC):
    def __init__(self, method_name: str) -> None:
        self.method_name = method_name

    def _with_curation_decision(
        self,
        record: dict[str, Any],
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        updated = dict(record)
        workflow = dict(updated.get("workflow", {}))
        curation = dict(workflow.get("curation") or {})
        decisions = list(curation.get("decisions", []))
        decisions.append({"stage": self.method_name, **decision})
        curation["decisions"] = decisions
        workflow["curation"] = curation
        updated["workflow"] = workflow
        return updated

    def _filter_records(
        self,
        records: list[dict[str, Any]],
        kept_structures: list[Atoms],
        *,
        reason: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        kept_ids = {id(atoms) for atoms in kept_structures}
        kept_records: list[dict[str, Any]] = []
        removed_records: list[dict[str, Any]] = []

        for record in records:
            if id(record["atoms"]) in kept_ids:
                kept = self._with_curation_decision(
                    record,
                    {"decision": "kept"},
                )
                kept_records.append(kept)
                continue

            removed = self._with_curation_decision(
                record,
                {
                    "decision": "removed",
                    "reason": reason,
                },
            )
            removed_records.append(removed)

        return kept_records, removed_records

    def curate_records(
        self,
        records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CurationResult:
        structures = [record["atoms"] for record in records]
        kept_structures = self.curate(structures, records=records, **kwargs)
        kept_records, removed_records = self._filter_records(
            records,
            kept_structures,
            reason=f"removed_by_{self.method_name}",
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
                )
            ],
        )

    @abstractmethod
    def curate(
        self,
        structures: list[Atoms],
        **kwargs: Any,
    ) -> list[Atoms]:
        pass
