from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import write

from al_dirac.curator.base import CurationResult
from al_dirac.workflow.state import WorkflowState
from al_dirac.workflow.workflow_logger import WorkflowLogger


EXTXYZ_INFO_KEYS = (
    "structure_id",
    "iteration",
    "parent_structure_id",
    "energy_std",
    "energy_mean",
    "force_mean_uncertainty",
    "force_max_uncertainty",
    "selection_score",
    "energy_dft",
)


def write_iteration_plots(
    records: list[dict[str, Any]],
    *,
    state: WorkflowState,
    plot_dir: str | Path | None,
    selected_records: list[dict[str, Any]] | None = None,
    curation_result: CurationResult | None = None,
    logger: WorkflowLogger | None = None,
) -> None:
    if plot_dir is None:
        return

    from al_dirac.plotting.plots import plot_iteration_summary

    paths = plot_iteration_summary(
        records,
        plot_dir,
        iteration=state.iteration,
        selected_records=selected_records,
        curation_result=curation_result,
    )
    if logger is not None:
        logger.log_event(
            "plots_completed",
            state=state,
            plots=[str(path) for path in paths],
        )


def write_records_extxyz(records: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return path

    atoms_list = []
    for record in records:
        atoms = record["atoms"].copy()
        for key in EXTXYZ_INFO_KEYS:
            if record.get(key) is not None:
                atoms.info[key] = record[key]
        atoms_list.append(atoms)
    write(path, atoms_list, format="extxyz")
    return path


def write_seed_selection_plot(
    seed_records: list[dict[str, Any]],
    *,
    state: WorkflowState,
    plot_dir: str | Path | None,
    logger: WorkflowLogger | None = None,
) -> None:
    if plot_dir is None:
        return

    from al_dirac.plotting.plots import plot_seed_selection_values

    value_key = (
        "selection_score"
        if any(record.get("selection_score") is not None for record in seed_records)
        else "energy"
    )
    paths = [
        plot_seed_selection_values(
            seed_records,
            plot_dir,
            iteration=state.iteration,
            value_key=value_key,
        )
    ]
    if logger is not None:
        logger.log_event(
            "seed_selection_plots_completed",
            state=state,
            plots=[str(path) for path in paths],
        )


def write_iteration_extxyz_outputs(
    pool_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    *,
    state: WorkflowState,
    plot_dir: str | Path | None,
) -> None:
    if plot_dir is None:
        return

    iteration_dir = Path(plot_dir) / f"iteration_{state.iteration:04d}"
    write_records_extxyz(pool_records, iteration_dir / "scored_pool.extxyz")
    write_records_extxyz(selected_records, iteration_dir / "selected_records.extxyz")


def annotate_dft_labels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated_records = []
    for record in records:
        updated = dict(record)
        results = dict(getattr(updated["atoms"].calc, "results", {}) or {})
        energy = results.get("energy", updated["atoms"].info.get("energy"))
        if energy is not None:
            updated["energy_dft"] = float(energy)
        annotated_records.append(updated)
    return annotated_records


def write_labeling_plots(
    labeled_records: list[dict[str, Any]],
    *,
    state: WorkflowState,
    plot_dir: str | Path | None,
    logger: WorkflowLogger | None = None,
) -> None:
    if plot_dir is None:
        return

    from al_dirac.plotting.plots import plot_parity

    paths = []
    if any(
        record.get("energy_dft") is not None and record.get("energy_mean") is not None
        for record in labeled_records
    ):
        paths.append(
            plot_parity(
                labeled_records,
                plot_dir,
                iteration=state.iteration,
                true_key="energy_dft",
                pred_key="energy_mean",
                name="energy_dft_vs_energy_mean",
            )
        )
    if logger is not None and paths:
        logger.log_event(
            "labeling_plots_completed",
            state=state,
            plots=[str(path) for path in paths],
        )
