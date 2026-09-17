from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write

from al_dirac.models.model_factory import ModelEnsembleFactory
from al_dirac.selection.ensemble_uncertainty_selector import EnsembleUncertaintySelector
from al_dirac.uncertainty.ensemble import EnsembleUncertainty
from al_dirac.workflow.active_learning import ActiveLearningWorkflow
from al_dirac.workflow.state import WorkflowState
from al_dirac.workflow.workflow_logger import WorkflowLogger


class FakeMACEModel:
    """Stands in for MACEModel -- skips real training/inference so
    ModelErrorStoppingCriteria can be demonstrated without a GPU or a real
    MACE run. latest_validation_metrics() returns preset values instead of
    reading a results/*.txt file."""

    def __init__(self, mae_f: float, mae_e_per_atom: float, **kwargs: Any) -> None:
        self._metrics = {"mae_f": mae_f, "mae_e_per_atom": mae_e_per_atom}

    def train(self, **kwargs: Any) -> None:
        pass

    def latest_validation_metrics(self) -> dict[str, float]:
        return self._metrics

    def predict(self, atoms: Atoms) -> dict[str, Any]:
        n_atoms = len(atoms)
        return {
            "energy": 0.0,
            "forces": np.zeros((n_atoms, 3)),
            "stress": np.zeros(6),
        }


def make_candidates(n: int) -> list[Atoms]:
    return [
        Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75 + 0.05 * index]],
            cell=[8.0, 8.0, 8.0],
            pbc=False,
        )
        for index in range(n)
    ]


def run_case(
    *,
    name: str,
    mae_f: float,
    mae_e_per_atom: float,
    force_threshold: float = 0.05,
    energy_threshold: float = 0.005,
) -> None:
    run_dir = Path("example_outputs/stopping") / name
    artifact_dir = run_dir / "artifacts"
    plot_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_path = run_dir / "train.extxyz"
    write(train_path, make_candidates(1))

    state = WorkflowState(run_id=f"stopping_{name}", iteration=0)
    logger = WorkflowLogger(run_dir=run_dir)

    factory = ModelEnsembleFactory(
        FakeMACEModel,
        n_models=2,
        name_prefix="mock",
        training_method="train",
        base_kwargs={"mae_f": mae_f, "mae_e_per_atom": mae_e_per_atom},
    )
    # Placeholder model only to satisfy EnsembleUncertainty's non-empty
    # constructor check -- run_iteration() replaces .models with the freshly
    # "trained" committee once training completes.
    uncertainty = EnsembleUncertainty(models=[FakeMACEModel(mae_f=0.0, mae_e_per_atom=0.0)])
    selector = EnsembleUncertaintySelector(
        score_expression="force_max_uncertainty",
        larger_is_better=True,
        min_score=None,
    )
    workflow = ActiveLearningWorkflow(
        uncertainty=uncertainty,
        selector=selector,
        model_factory=factory,
    )

    selected_records = workflow.run_iteration(
        state,
        uncertainty_k=2,
        selection_mode="uncertainty",
        candidate_structures=make_candidates(3),
        train_path=str(train_path),
        train_prediction_model=False,
        train_model_ensemble=True,
        model_factory=factory,
        uncertainty=uncertainty,
        model_error_stop_force_threshold=force_threshold,
        model_error_stop_energy_threshold=energy_threshold,
        selector=selector,
        coverage_k=0,
        logger=logger,
        artifact_dir=artifact_dir,
        plot_dir=plot_dir,
    )

    print(name)
    print(f"  stopped: {state.stop_reason is not None}")
    print(f"  stop reason: {state.stop_reason}")
    print(f"  selected records: {len(selected_records)}")
    print(f"  events: {logger.events_path}")


def main() -> None:
    output_dir = Path("example_outputs/stopping")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Both metrics well above threshold -- loop continues (samples, scores,
    # selects normally).
    run_case(name="high_error_continues", mae_f=0.20, mae_e_per_atom=0.05)

    # Force error alone is fine, but energy error isn't -- AND logic means
    # this still doesn't stop.
    run_case(name="energy_error_blocks_stop", mae_f=0.03, mae_e_per_atom=0.05)

    # Both metrics below threshold -- loop stops right after training,
    # before sampling/scoring/selecting ever run.
    run_case(name="both_low_stops", mae_f=0.03, mae_e_per_atom=0.002)


if __name__ == "__main__":
    main()
