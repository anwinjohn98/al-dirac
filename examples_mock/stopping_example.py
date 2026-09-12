from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ase import Atoms

from al_dirac.selection.ensemble_uncertainty_selector import EnsembleUncertaintySelector
from al_dirac.uncertainty.base import BaseUncertainty
from al_dirac.workflow.active_learning import ActiveLearningWorkflow
from al_dirac.workflow.state import WorkflowState
from al_dirac.workflow.workflow_logger import WorkflowLogger


class PresetUncertainty(BaseUncertainty):
    def __init__(self) -> None:
        super().__init__(method_name="preset_uncertainty")

    def predict(self, atoms: Atoms) -> dict[str, Any]:
        return {
            "energy_std": float(atoms.info["mock_energy_std"]),
            "force_mean_uncertainty": float(atoms.info["mock_force_mean_uncertainty"]),
            "force_max_uncertainty": float(atoms.info["mock_force_max_uncertainty"]),
        }


def make_candidates(values: list[tuple[float, float, float]]) -> list[Atoms]:
    candidates = []
    for index, (energy_std, force_mean, force_max) in enumerate(values):
        atoms = Atoms(
            "H2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.75 + 0.15 * index],
            ],
            cell=[8.0, 8.0, 8.0],
            pbc=False,
        )
        atoms.info["mock_energy_std"] = energy_std
        atoms.info["mock_force_mean_uncertainty"] = force_mean
        atoms.info["mock_force_max_uncertainty"] = force_max
        candidates.append(atoms)
    return candidates


def run_case(
    *,
    name: str,
    candidates: list[Atoms],
    uncertainty_stop_key: str = "force_max_uncertainty",
    uncertainty_stop_expression: str | None = None,
    uncertainty_stop_threshold: float,
    uncertainty_stop_statistic: str = "max",
) -> list[dict[str, Any]]:
    run_dir = Path("example_outputs/stopping") / name
    artifact_dir = run_dir / "artifacts"
    plot_dir = run_dir / "plots"

    state = WorkflowState(run_id=f"stopping_{name}", iteration=0)
    logger = WorkflowLogger(run_dir=run_dir)
    uncertainty = PresetUncertainty()
    selector = EnsembleUncertaintySelector(
        score_expression="force_max_uncertainty",
        larger_is_better=True,
        min_score=None,
    )
    workflow = ActiveLearningWorkflow(
        uncertainty=uncertainty,
        selector=selector,
        pre_uncertainty_curation_pipeline=None,
        parser_curation_pipeline=None,
        model=None,
        model_factory=None,
        sampler=None,
        dft_runner=None,
        coverage_curation_pipeline=None,
    )

    selected_records = workflow.run_iteration(
        state,
        uncertainty_k=2,
        selection_mode="uncertainty",
        cold_start_k=None,
        candidate_structures=candidates,
        model=None,
        train_path=None,
        seed_structures=None,
        dft_root_dir=None,
        valid_path=None,
        train_prediction_model=False,
        train_model_ensemble=False,
        train_kwargs=None,
        model_factory=None,
        model_factory_train_kwargs=None,
        load_model_factory_checkpoints=True,
        parse_base_dir=None,
        parse_kwargs=None,
        parser_curation_pipeline=None,
        parser_curation_kwargs=None,
        seed_selection_mode="all",
        seed_k=None,
        seed_selection_curation_pipeline=None,
        seed_selection_curation_kwargs=None,
        seed_selection_uncertainty=None,
        seed_selection_score_key="force_max_uncertainty",
        seed_selection_random_seed=None,
        sampler=None,
        sampler_kwargs=None,
        common_data={"example": "stopping", "case": name},
        pre_uncertainty_curation_pipeline=None,
        pre_uncertainty_curation_kwargs=None,
        uncertainty=uncertainty,
        uncertainty_stop_key=uncertainty_stop_key,
        uncertainty_stop_threshold=uncertainty_stop_threshold,
        uncertainty_stop_statistic=uncertainty_stop_statistic,
        uncertainty_stop_expression=uncertainty_stop_expression,
        selector=selector,
        coverage_k=0,
        coverage_curation_pipeline=None,
        coverage_curation_kwargs=None,
        dft_runner=None,
        labeler_kwargs=None,
        template_replacements=None,
        logger=logger,
        artifact_dir=artifact_dir,
        plot_dir=plot_dir,
        restart_from_stage=None,
    )

    print(name)
    print(f"  stopped: {state.stop_reason is not None}")
    print(f"  stop reason: {state.stop_reason}")
    print(f"  selected records: {len(selected_records)}")
    print(f"  state status: {state.status}")
    print(f"  events: {logger.events_path}")
    print(f"  plots: {plot_dir / 'iteration_0000'}")
    return selected_records


def main() -> None:
    output_dir = Path("example_outputs/stopping")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    high_uncertainty = make_candidates(
        [
            (0.020, 0.040, 0.080),
            (0.018, 0.035, 0.070),
            (0.010, 0.020, 0.040),
        ]
    )
    low_uncertainty = make_candidates(
        [
            (0.004, 0.006, 0.010),
            (0.005, 0.007, 0.012),
            (0.003, 0.005, 0.009),
        ]
    )

    run_case(
        name="high_uncertainty_selects",
        candidates=high_uncertainty,
        uncertainty_stop_key="force_max_uncertainty",
        uncertainty_stop_threshold=0.050,
        uncertainty_stop_statistic="max",
    )
    run_case(
        name="low_uncertainty_stops_by_key",
        candidates=low_uncertainty,
        uncertainty_stop_key="force_max_uncertainty",
        uncertainty_stop_threshold=0.050,
        uncertainty_stop_statistic="max",
    )
    run_case(
        name="low_uncertainty_stops_by_expression",
        candidates=low_uncertainty,
        uncertainty_stop_key="force_max_uncertainty",
        uncertainty_stop_expression="0.7 * force_max_uncertainty + 0.3 * energy_std",
        uncertainty_stop_threshold=0.020,
        uncertainty_stop_statistic="max",
    )


if __name__ == "__main__":
    main()
