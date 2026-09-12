from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write

from al_dirac.models.base import BaseModel
from al_dirac.plotting.plots import (
    plot_selection_scores,
    plot_uncertainty_histogram,
    plot_uncertainty_scatter,
)
from al_dirac.selection.ensemble_uncertainty_selector import EnsembleUncertaintySelector
from al_dirac.uncertainty.ensemble import EnsembleUncertainty
from al_dirac.workflow.active_learning import ActiveLearningWorkflow


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

        energy = float(0.1 * len(atoms) + self.energy_offset * spread)
        forces = -self.force_scale * positions
        stress = np.full(6, self.stress_offset * spread, dtype=float)

        return {
            "energy": energy,
            "forces": forces,
            "stress": stress,
        }

    def save(self, checkpoint_path: str | Path) -> None:
        Path(checkpoint_path).write_text(self.model_name + "\n")

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> "MockModel":
        return cls(
            model_index=0,
            force_scale=1.0,
            energy_offset=0.0,
            stress_offset=0.0,
        )


def make_candidate_records() -> list[dict[str, Any]]:
    records = []
    bond_lengths = [0.75, 0.85, 1.00, 1.25, 1.55, 1.90, 2.30]

    for index, bond_length in enumerate(bond_lengths):
        atoms = Atoms(
            "H2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, bond_length],
            ],
            cell=[8, 8, 8],
            pbc=False,
        )
        records.append(
            {
                "atoms": atoms,
                "candidate_index": index,
                "structure_id": f"h2_candidate_{index}",
                "iteration": 0,
                "parent_structure_id": "h2_seed",
            }
        )

    return records


def write_records_extxyz(records: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    atoms_list = []
    for record in records:
        atoms = record["atoms"].copy()
        atoms.info["structure_id"] = record["structure_id"]
        atoms.info["force_max_uncertainty"] = record["force_max_uncertainty"]
        atoms.info["force_mean_uncertainty"] = record["force_mean_uncertainty"]
        atoms.info["energy_std"] = record["energy_std"]
        atoms.info["selection_score"] = record.get("selection_score")
        atoms_list.append(atoms)

    write(output_path, atoms_list, format="extxyz")
    return output_path


def print_records(title: str, records: list[dict[str, Any]]) -> None:
    print(title)
    for record in records:
        print(
            f"{record['structure_id']}: "
            f"energy_std={record.get('energy_std'):.6f}, "
            f"force_mean_uncertainty={record.get('force_mean_uncertainty'):.6f}, "
            f"force_max_uncertainty={record.get('force_max_uncertainty'):.6f}, "
            f"selection_score={record.get('selection_score')}"
        )


def main() -> None:
    output_dir = Path("example_outputs/uncertainty_selection")
    plot_dir = output_dir / "plots"
    pool_path = output_dir / "scored_pool.extxyz"
    selected_path = output_dir / "selected_candidates.extxyz"

    models = [
        MockModel(
            model_index=0,
            force_scale=0.8,
            energy_offset=0.01,
            stress_offset=0.001,
        ),
        MockModel(
            model_index=1,
            force_scale=1.0,
            energy_offset=0.03,
            stress_offset=0.002,
        ),
        MockModel(
            model_index=2,
            force_scale=1.2,
            energy_offset=0.06,
            stress_offset=0.003,
        ),
    ]

    uncertainty = EnsembleUncertainty(
        models=models,
    )
    selector = EnsembleUncertaintySelector(
        score_expression="force_max_uncertainty",
        larger_is_better=True,
        min_score=None,
    )
    expression_selector = EnsembleUncertaintySelector(
        score_expression="force_max_uncertainty + 0.5 * energy_std",
        larger_is_better=True,
        min_score=0.01,
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

    candidate_records = make_candidate_records()
    scored_records = workflow.score_records(
        candidate_records,
        uncertainty=uncertainty,
    )

    selected_records = workflow.select_for_labeling(
        scored_records,
        uncertainty_k=3,
        selector=selector,
        coverage_k=0,
        coverage_curation_pipeline=None,
        coverage_curation_kwargs=None,
    )

    expression_selected_records = expression_selector.select(
        scored_records,
        k=3,
    )

    write_records_extxyz(scored_records, pool_path)
    write_records_extxyz(selected_records, selected_path)

    plot_paths = [
        plot_uncertainty_histogram(
            scored_records,
            plot_dir,
            iteration=0,
            key="force_max_uncertainty",
            selected_records=selected_records,
        ),
        plot_uncertainty_histogram(
            scored_records,
            plot_dir,
            iteration=0,
            key="force_mean_uncertainty",
            selected_records=selected_records,
        ),
        plot_uncertainty_scatter(
            scored_records,
            plot_dir,
            iteration=0,
            x_key="energy_std",
            y_key="force_max_uncertainty",
            selected_records=selected_records,
        ),
        plot_selection_scores(
            selected_records,
            plot_dir,
            iteration=0,
            score_key="selection_score",
        ),
    ]

    print_records("scored records", scored_records)
    print_records("selected by force_max_uncertainty", selected_records)
    print_records("selected by score_expression", expression_selected_records)

    print("outputs:")
    print(f"scored pool extxyz: {pool_path}")
    print(f"selected extxyz: {selected_path}")
    for path in plot_paths:
        print(f"plot: {path}")


if __name__ == "__main__":
    main()
