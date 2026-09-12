from __future__ import annotations

from typing import Any

from ase import Atoms

from al_dirac.curator.base import CurationResult


WORKFLOW_STAGE_KEYS = (
    "parser",
    "sampler",
    "seed_selection",
    "curation",
    "uncertainty",
    "selection",
    "dft_labeling",
)


def workflow_template(workflow: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = {key: None for key in WORKFLOW_STAGE_KEYS}
    if workflow is not None:
        normalized.update(workflow)
    return normalized


def with_workflow_stage(
    record: dict[str, Any],
    stage: str,
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(record)
    workflow = workflow_template(updated.get("workflow"))
    workflow[stage] = value
    updated["workflow"] = workflow
    return updated


def build_records(
    candidate_structures: list[Atoms],
    *,
    iteration: int = 0,
    common_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            **(common_data or {}),
            "candidate_index": index,
            "structure_id": f"candidate_{iteration}_{index}",
            "atoms": atoms,
            "iteration": iteration,
            "workflow": workflow_template(
                {
                    **((common_data or {}).get("workflow", {})),
                    "sampler": None,
                }
            ),
        }
        for index, atoms in enumerate(candidate_structures)
    ]


def get_iteration_value(values: list[Any] | None, index: int) -> Any:
    if values is None or index >= len(values):
        return None
    return values[index]


def split_labeled_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labeled_records: list[dict[str, Any]] = []
    unlabeled_records: list[dict[str, Any]] = []
    for record in records:
        target = labeled_records if record.get("is_labeled", False) else unlabeled_records
        target.append(record)
    return labeled_records, unlabeled_records


def records_from_curation_result(result: CurationResult) -> list[dict[str, Any]]:
    stage_reports = [
        {
            "stage": report.stage,
            "input_count": report.input_count,
            "kept_count": report.kept_count,
            "removed_count": report.removed_count,
            "details": report.details,
        }
        for report in result.stage_reports
    ]

    curated_records: list[dict[str, Any]] = []
    for record in result.kept_records:
        updated = dict(record)
        workflow = workflow_template(updated.get("workflow"))
        curation = dict(workflow.get("curation") or {})
        curation["reports"] = stage_reports
        workflow["curation"] = curation
        updated["workflow"] = workflow
        curated_records.append(updated)
    return curated_records
