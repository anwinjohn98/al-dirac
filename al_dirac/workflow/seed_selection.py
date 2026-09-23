from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from al_dirac.curator.structure_curation import StructureCurationPipeline
from al_dirac.uncertainty.base import BaseUncertainty
from al_dirac.workflow.records import with_workflow_stage


def record_energy(record: dict[str, Any]) -> float:
    if record.get("energy") is not None:
        return float(record["energy"])
    atoms = record["atoms"]
    for key in ("energy", "energy_dft"):
        if key in atoms.info:
            return float(atoms.info[key])
    return float("inf")


def record_identity(record: dict[str, Any]) -> int:
    return id(record["atoms"])


def select_seed_records(
    records: list[dict[str, Any]],
    *,
    mode: str = "all",
    k: int | Callable[[int], int] | None = None,
    curation_pipeline: StructureCurationPipeline | None = None,
    curation_kwargs: dict[str, Any] | None = None,
    uncertainty: BaseUncertainty | None = None,
    score_key: str = "force_max_uncertainty",
    random_seed: int | None = None,
    score_records: Callable[..., list[dict[str, Any]]],
    curate_records: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    # k may be a plain int/None, or a pure function of the input pool size
    # (called once here, at most a second time by the caller for logging --
    # see ActiveLearningWorkflow.run_iteration()) that lets seed_k scale
    # with however much data actually exists instead of a fixed constant.
    if callable(k):
        k = k(len(records))
    if k is not None and k < 0:
        raise ValueError("seed_k must be non-negative or None.")
    if k == 0 or not records:
        return []
    if mode not in {
        "all",
        "random",
        "lowest_energy",
        "highest_uncertainty",
        "diverse",
        "diverse_low_energy",
    }:
        raise ValueError(f"Unsupported seed selection mode: {mode}")

    selected_records = list(records)
    if mode == "random":
        rng = random.Random(random_seed)
        selected_records = rng.sample(
            selected_records,
            len(selected_records) if k is None else min(k, len(selected_records)),
        )
    elif mode == "lowest_energy":
        selected_records = sorted(selected_records, key=record_energy)
    elif mode == "highest_uncertainty":
        if uncertainty is None:
            raise ValueError("seed_selection_uncertainty is required for highest_uncertainty.")
        selected_records = score_records(selected_records, uncertainty=uncertainty)
        selected_records = sorted(
            [record for record in selected_records if record.get(score_key) is not None],
            key=lambda record: float(record[score_key]),
            reverse=True,
        )
    elif mode in {"diverse", "diverse_low_energy"}:
        selected_records = curate_records(
            selected_records,
            curation_pipeline=curation_pipeline,
            curation_kwargs=curation_kwargs,
        )
        if mode == "diverse_low_energy":
            selected_records = sorted(selected_records, key=record_energy)

    if mode != "random" and k is not None:
        selected_records = selected_records[:k]

    return [
        with_workflow_stage(
            record,
            "seed_selection",
            {
                "mode": mode,
                "k": k,
                "score_key": score_key if mode == "highest_uncertainty" else None,
                "curation_pipeline": (
                    None if curation_pipeline is None else curation_pipeline.method_name
                ),
            },
        )
        for record in selected_records
    ]
