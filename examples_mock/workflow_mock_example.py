from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write

from al_dirac.curator.cheap_curate import CheapCurate
from al_dirac.curator.structure_curation import StructureCurationPipeline
from al_dirac.models.base import BaseModel
from al_dirac.samplers.rattle import RattleSampler
from al_dirac.selection.ensemble_uncertainty_selector import EnsembleUncertaintySelector
from al_dirac.uncertainty.ensemble import EnsembleUncertainty
from al_dirac.workflow.active_learning import ActiveLearningWorkflow
from al_dirac.workflow.state import WorkflowState
from al_dirac.workflow.workflow_logger import WorkflowLogger


class MockModel(BaseModel):
    def __init__(
        self,
        model_index: int,
        force_scale: float,
        energy_offset: float,
        stress_offset: float,
    ) -> None:
        super().__init__(model_name=f"mock_model_{model_index}")
        self.model_index = model_index
        self.force_scale = force_scale
        self.energy_offset = energy_offset
        self.stress_offset = stress_offset

    def train(
        self,
        train_path: str | Path,
        valid_path: str | Path | None = None,
    ) -> None:
        return None

    def predict(self, atoms: Atoms) -> dict[str, Any]:
        positions = atoms.get_positions()
        spread = float(np.linalg.norm(positions - positions.mean(axis=0)))

        return {
            "energy": float(0.1 * len(atoms) + self.energy_offset * spread),
            "forces": -self.force_scale * positions,
            "stress": np.full(6, self.stress_offset * spread),
        }

    def save(self, checkpoint_path: str | Path) -> None:
        Path(checkpoint_path).write_text(self.model_name + "\n")

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> "MockModel":
        return cls(0, 1.0, 0.0, 0.0)


def make_seed_structures() -> list[Atoms]:
    seeds = [
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=False),
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.20]], cell=[8, 8, 8], pbc=False),
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.80]], cell=[8, 8, 8], pbc=False),
    ]

    for index, atoms in enumerate(seeds):
        atoms.info["energy"] = [-1.0, -0.7, -0.2][index]

    return seeds


def write_records_extxyz(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    atoms_list = []
    for record in records:
        atoms = record["atoms"].copy()
        atoms.info["structure_id"] = record["structure_id"]
        atoms.info["iteration"] = record["iteration"]

        for key in (
            "parent_structure_id",
            "energy_std",
            "force_mean_uncertainty",
            "force_max_uncertainty",
            "selection_score",
        ):
            if record.get(key) is not None:
                atoms.info[key] = record[key]

        atoms_list.append(atoms)

    write(path, atoms_list, format="extxyz")
    return path


def main() -> None:
    run_dir = Path("example_outputs/workflow_mock")
    artifact_dir = run_dir / "artifacts"
    plot_dir = run_dir / "plots"
    selected_path = run_dir / "selected_records.extxyz"

    state = WorkflowState(
        run_id="workflow_mock_example",
        iteration=0,
    )
    logger = WorkflowLogger(
        run_dir=run_dir,
    )

    models = [
        MockModel(0, force_scale=0.8, energy_offset=0.01, stress_offset=0.001),
        MockModel(1, force_scale=1.0, energy_offset=0.03, stress_offset=0.002),
        MockModel(2, force_scale=1.2, energy_offset=0.06, stress_offset=0.003),
    ]
    uncertainty = EnsembleUncertainty(
        models=models,
    )
    selector = EnsembleUncertaintySelector(
        score_expression="force_max_uncertainty",
        larger_is_better=True,
        min_score=None,
    )

    sampler = RattleSampler(
        stdev=0.08,
        n_samples=4,
        seed=7,
        min_distance_scale=0.7,
        max_attempts_per_sample=20,
    )

    cheap_curate = CheapCurate(
        position_decimals=1,
        cell_decimals=6,
        energy_decimals=None,
        energy_tolerance=None,
    )
    curation_pipeline = StructureCurationPipeline(
        cheap_curate=cheap_curate,
        descriptor_curate=None,
        cluster_curate=None,
        use_cheap=True,
        use_descriptor=False,
        use_cluster=False,
    )

    workflow = ActiveLearningWorkflow(
        uncertainty=uncertainty,
        selector=selector,
        pre_uncertainty_curation_pipeline=curation_pipeline,
        parser_curation_pipeline=None,
        model=None,
        model_factory=None,
        sampler=sampler,
        dft_runner=None,
        coverage_curation_pipeline=None,
    )

    selected_records = workflow.run_iteration(
        state,
        uncertainty_k=3,
        selection_mode="uncertainty",
        cold_start_k=None,
        candidate_structures=None,
        model=None,
        train_path=None,
        seed_structures=make_seed_structures(),
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
        seed_selection_mode="lowest_energy",
        seed_k=2,
        seed_selection_curation_pipeline=None,
        seed_selection_curation_kwargs=None,
        seed_selection_uncertainty=None,
        seed_selection_score_key="force_max_uncertainty",
        seed_selection_random_seed=11,
        sampler=sampler,
        sampler_kwargs=None,
        common_data={"example": "workflow_mock"},
        pre_uncertainty_curation_pipeline=curation_pipeline,
        pre_uncertainty_curation_kwargs=None,
        uncertainty=uncertainty,
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

    write_records_extxyz(selected_records, selected_path)

    print("workflow mock complete")
    print(f"selected records: {len(selected_records)}")
    print(f"state iteration after run: {state.iteration}")
    print(f"artifacts: {artifact_dir}")
    print(f"seed selection artifact: {artifact_dir / 'iteration_0000_seed_selecting.pkl'}")
    print(f"plots: {plot_dir / 'iteration_0000'}")
    print(f"selected extxyz: {selected_path}")
    print(f"events: {logger.events_path}")
    print(f"metrics: {logger.metrics_path}")


if __name__ == "__main__":
    main()
