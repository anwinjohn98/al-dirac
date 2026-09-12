from __future__ import annotations

import random
import shutil
import time
from pathlib import Path

from ase.io import read, write

from al_dirac.models.mace_model import MACEModel
from al_dirac.models.model_factory import (
    HyperparameterStrategy,
    ModelEnsembleFactory,
    SeedStrategy,
)

INPUT_FILE = Path("real_example_outputs/parser/train.extxyz")
FOUNDATION_MODEL = "medium"

OUTPUT_DIR = Path("real_example_outputs/model_factory")
DATA_DIR = OUTPUT_DIR / "data"

DEVICE = "cuda"
N_MODELS = 3
EPOCHS = 5
BATCH_SIZE = 2
SEED_START = 7
LORA_RANKS = [2, 4, 8]
LORA_ALPHA = 8


def split_structures(structures, seed):
    shuffled = list(structures)
    random.Random(seed).shuffle(shuffled)
    test = shuffled[:1]
    valid = shuffled[1:2]
    train = shuffled[2:]
    if not train or not valid or not test:
        raise ValueError("Need at least 3 structures for train/valid/test split.")
    return train, valid, test


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Run examples/parser_example.py first.")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    structures = read(INPUT_FILE, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    for atoms in structures:
        atoms.info.setdefault("config_type", "Pt_surface")

    train_structures, valid_structures, test_structures = split_structures(
        structures, SEED_START
    )

    train_file = DATA_DIR / "train.extxyz"
    valid_file = DATA_DIR / "valid.extxyz"
    test_file = DATA_DIR / "test.extxyz"
    write(train_file, train_structures, format="extxyz")
    write(valid_file, valid_structures, format="extxyz")
    write(test_file, test_structures, format="extxyz")

    if len(LORA_RANKS) != N_MODELS:
        raise ValueError("LORA_RANKS must have one entry per model.")

    factory = ModelEnsembleFactory(
        MACEModel,
        n_models=N_MODELS,
        name_prefix="pt_surface_lora_ensemble",
        training_method="lora_finetune",
        base_kwargs={
            "model": "MACE",
            "device": DEVICE,
            "default_dtype": "float32",
            "checkpoint_dir": OUTPUT_DIR,
        },
    )
    factory.add_strategy(SeedStrategy(seed_start=SEED_START))
    factory.add_strategy(
        HyperparameterStrategy(
            hyperparameters=[{"lora_rank": rank} for rank in LORA_RANKS]
        )
    )

    start_time = time.perf_counter()
    models = factory.train(
        train_file,
        valid_path=valid_file,
        train_kwargs={
            "test_file": test_file,
            "foundation_model": FOUNDATION_MODEL,
            "lora_alpha": LORA_ALPHA,
            "energy_key": "REF_energy",
            "forces_key": "REF_forces",
            "stress_key": "REF_stress",
            "E0s": "average",
            "batch_size": BATCH_SIZE,
            "valid_batch_size": BATCH_SIZE,
            "max_num_epochs": EPOCHS,
            "lr": 0.0005,
            "ema": True,
            "ema_decay": 0.99,
            "loss": "weighted",
            "num_workers": 0,
            "enable_cueq": True,
            "work_dir": str(OUTPUT_DIR),
            "extra_args": [
                "--device", DEVICE,
                "--default_dtype", "float32",
                "--error_table", "PerAtomMAE",
                "--plot", "False",
            ],
        },
        load_checkpoints=True,
    )
    elapsed = time.perf_counter() - start_time

    probe_structure = test_structures[0]
    print(f"trained {len(models)} ensemble members in {elapsed:.1f}s")
    for lora_rank, model in zip(LORA_RANKS, models):
        checkpoint_path = model.expected_checkpoint_path()
        prediction = model.predict(probe_structure.copy())
        print(
            f"{model.name} (lora_rank={lora_rank}): checkpoint={checkpoint_path} "
            f"(exists={checkpoint_path.exists() if checkpoint_path else False}), "
            f"predicted_energy={prediction['energy']:.6f}"
        )


if __name__ == "__main__":
    main()
