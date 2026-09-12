from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.db import connect
from ase.io import write

from al_dirac.models.base import BaseModel
from al_dirac.models.model_factory import (
    HyperparameterStrategy,
    ModelFactoryStrategy,
    ModelEnsembleFactory,
    RandomDataSplitStrategy,
    SeedStrategy,
)
from al_dirac.selection.ensemble_uncertainty_selector import EnsembleUncertaintySelector
from al_dirac.uncertainty.base import BaseUncertainty
from al_dirac.uncertainty.ensemble import EnsembleUncertainty
from al_dirac.workflow.active_learning import ActiveLearningWorkflow


class MockTrainableModel(BaseModel):
    def __init__(
        self,
        name: str,
        checkpoint_dir: str | Path,
        force_scale: float = 1.0,
        energy_offset: float = 0.0,
        training_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_name=name)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.force_scale = force_scale
        self.energy_offset = energy_offset
        self.training_kwargs = training_kwargs or {}

    def expected_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "model.json"

    def train(
        self,
        train_path: str | Path,
        valid_path: str | Path | None = None,
        **train_kwargs: Any,
    ) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "train_path": str(train_path),
            "valid_path": None if valid_path is None else str(valid_path),
            "force_scale": self.force_scale,
            "energy_offset": self.energy_offset,
            "training_kwargs": {
                **self.training_kwargs,
                **train_kwargs,
            },
        }
        self.expected_checkpoint_path().write_text(json.dumps(payload, indent=2) + "\n")

    def predict(self, atoms: Atoms) -> dict[str, Any]:
        positions = atoms.get_positions()
        spread = float(np.linalg.norm(positions - positions.mean(axis=0)))
        return {
            "energy": float(0.1 * len(atoms) + self.energy_offset * spread),
            "forces": -self.force_scale * positions,
            "stress": np.zeros(6),
        }

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint_path.exists():
            checkpoint_path.write_text(
                json.dumps({"model_name": self.model_name}, indent=2) + "\n"
            )

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> "MockTrainableModel":
        data = json.loads(Path(checkpoint_path).read_text())
        return cls(
            name=data["model_name"],
            checkpoint_dir=Path(checkpoint_path).parent,
            force_scale=data.get("force_scale", 1.0),
            energy_offset=data.get("energy_offset", 0.0),
            training_kwargs=data.get("training_kwargs", {}),
        )


class ModelParameterStrategy(ModelFactoryStrategy):
    def __init__(self, parameters: list[dict[str, Any]]) -> None:
        self.parameters = parameters

    def apply(self, index: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        if index >= len(self.parameters):
            raise ValueError(f"Missing model parameters for model index {index}.")
        kwargs.update(self.parameters[index])
        return kwargs


class PlaceholderUncertainty(BaseUncertainty):
    def __init__(self) -> None:
        self.models = []
        self.method_name = "placeholder_uncertainty"

    def predict(self, atoms: Atoms) -> dict[str, Any]:
        raise RuntimeError("PlaceholderUncertainty is not used for prediction.")


def evaluate(atoms: Atoms) -> Atoms:
    atoms = atoms.copy()
    atoms.calc = EMT()
    atoms.info["energy"] = float(atoms.get_potential_energy())
    atoms.arrays["forces"] = atoms.get_forces()
    atoms.calc = None
    return atoms


def make_labeled_structures() -> list[Atoms]:
    structures = []
    lattice_constants = [3.50, 3.55, 3.60, 3.65, 3.70, 3.75]

    for index, a in enumerate(lattice_constants):
        atoms = Atoms(
            "Cu4",
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.5 * a, 0.5 * a],
                [0.5 * a, 0.0, 0.5 * a],
                [0.5 * a, 0.5 * a, 0.0],
            ],
            cell=[a, a, a],
            pbc=True,
        )
        atoms.info["structure_id"] = f"cu4_train_{index}"
        structures.append(evaluate(atoms))

    return structures


def write_train_db(
    structures: list[Atoms],
    train_path: Path,
    workflow: ActiveLearningWorkflow,
) -> None:
    db = connect(train_path)
    records = [
        {
            "atoms": atoms,
            "structure_id": atoms.info["structure_id"],
            "iteration": 0,
            "is_labeled": True,
            "is_selected": False,
            "source": "training_example",
        }
        for atoms in structures
    ]
    workflow.append_records_to_db(
        db,
        records,
        split="train",
        is_labeled=True,
        is_selected=False,
        source="training_example",
    )


def main() -> None:
    output_dir = Path("example_outputs/training")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_path = output_dir / "train.aselmdb"
    train_extxyz = output_dir / "train.extxyz"
    checkpoint_root = output_dir / "checkpoints"
    split_root = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    selector = EnsembleUncertaintySelector(
        score_expression="force_max_uncertainty",
        larger_is_better=True,
        min_score=None,
    )
    placeholder_uncertainty = PlaceholderUncertainty()
    workflow = ActiveLearningWorkflow(
        uncertainty=placeholder_uncertainty,
        selector=selector,
        pre_uncertainty_curation_pipeline=None,
        parser_curation_pipeline=None,
        model=None,
        model_factory=None,
        sampler=None,
        dft_runner=None,
        coverage_curation_pipeline=None,
    )

    structures = make_labeled_structures()
    write(train_extxyz, structures, format="extxyz")
    write_train_db(structures, train_path, workflow)

    single_model = MockTrainableModel(
        name="single_mock_model",
        checkpoint_dir=checkpoint_root / "single_mock_model",
        force_scale=1.0,
        energy_offset=0.01,
        training_kwargs={"seed": 1, "max_epochs": 5},
    )
    single_model.train(
        train_path=train_path,
        valid_path=None,
        batch_size=2,
        learning_rate=1e-3,
    )

    factory = ModelEnsembleFactory(
        model_cls=MockTrainableModel,
        n_models=3,
        name_prefix="mock_ensemble",
        training_method="train",
        checkpoint_root=checkpoint_root,
        base_kwargs={
            "force_scale": 1.0,
            "energy_offset": 0.0,
            "training_kwargs": {"max_epochs": 5},
        },
        name_key="name",
        checkpoint_dir_key="checkpoint_dir",
        checkpoint_dir_extra_arg=None,
        training_kwargs_key="training_kwargs",
        extra_args_key="extra_args",
        train_key="train_path",
        valid_key="valid_path",
        start_index=0,
        width=3,
    )
    factory.add_strategy(
        SeedStrategy(
            seed_start=100,
            seed_key="seed",
            training_kwargs_key="training_kwargs",
            overwrite=True,
        )
    )
    factory.add_strategy(
        ModelParameterStrategy(
            parameters=[
                {"force_scale": 0.8, "energy_offset": 0.01},
                {"force_scale": 1.0, "energy_offset": 0.02},
                {"force_scale": 1.2, "energy_offset": 0.03},
            ],
        )
    )
    factory.add_strategy(
        HyperparameterStrategy(
            hyperparameters=[
                {"learning_rate": 1e-3},
                {"learning_rate": 5e-4},
                {"learning_rate": 2e-4},
            ],
            training_kwargs_key="training_kwargs",
            overwrite=True,
        )
    )
    factory.add_strategy(
        RandomDataSplitStrategy(
            split_root=split_root,
            train_fraction=0.67,
            valid_fraction=0.33,
            seed_start=200,
            file_format="extxyz",
            dirname_prefix="model_split",
            train_filename="train.extxyz",
            valid_filename="valid.extxyz",
            start_index=0,
            width=3,
            train_key="train_path",
            valid_key="valid_path",
        )
    )

    ensemble_models = factory.train(
        train_path=train_extxyz,
        valid_path=None,
        train_kwargs={
            "batch_size": 2,
        },
        load_checkpoints=True,
    )

    uncertainty = EnsembleUncertainty(models=ensemble_models)
    prediction = uncertainty.predict(structures[0])

    print("training example complete")
    print(f"training structures: {len(structures)}")
    print(f"train db: {train_path}")
    print(f"train extxyz: {train_extxyz}")
    print(f"single checkpoint: {single_model.expected_checkpoint_path()}")
    print(f"ensemble models: {len(ensemble_models)}")
    print(f"ensemble checkpoints: {checkpoint_root}")
    print(f"data splits: {split_root}")
    print(f"energy_std: {prediction['energy_std']:.6f}")
    print(f"force_max_uncertainty: {prediction['force_max_uncertainty']:.6f}")


if __name__ == "__main__":
    main()
