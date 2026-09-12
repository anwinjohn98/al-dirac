from __future__ import annotations

import random
import shutil
import time
from pathlib import Path

from ase.io import read, write

from al_dirac.models.mace_model import MACEModel

INPUT_FILE = Path("real_example_outputs/parser/train.extxyz")
# "medium" is MACE-MP-0, auto-downloaded by mace_run_train, and is one of the
# officially supported foundation models for multihead fine-tuning's "mp"
# replay method. It also avoids the remove_pt_head bug hit by our local
# multihead checkpoint (foundational_models/MACE/mace-mh-1.model).
FOUNDATION_MODEL = "medium"

OUTPUT_DIR = Path("real_example_outputs/mace_multihead_finetune")
DATA_DIR = OUTPUT_DIR / "data"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
LOG_DIR = OUTPUT_DIR / "logs"

TRAIN_FILE = DATA_DIR / "train.extxyz"
VALID_FILE = DATA_DIR / "valid.extxyz"
TEST_FILE = DATA_DIR / "test.extxyz"

DEVICE = "cuda"
EPOCHS = 5
BATCH_SIZE = 2
RANDOM_SEED = 7
LEARNING_RATE = 0.0005


def ensure_config_type(structures):
    for atoms in structures:
        atoms.info.setdefault("config_type", "Pt_surface")
    return structures


def split_structures(structures):
    shuffled = list(structures)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    test = shuffled[:1]
    valid = shuffled[1:2]
    train = shuffled[2:]

    if not train or not valid or not test:
        raise ValueError("Need at least 3 structures for train/valid/test split.")

    return train, valid, test


def list_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(file_path for file_path in path.rglob("*") if file_path.is_file())


def main() -> None:
    start_time = time.perf_counter()

    if not INPUT_FILE.exists():
        raise FileNotFoundError("Run examples/parser_example.py first.")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    structures = read(INPUT_FILE, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    structures = ensure_config_type(structures)

    train_structures, valid_structures, test_structures = split_structures(structures)

    write(TRAIN_FILE, train_structures, format="extxyz")
    write(VALID_FILE, valid_structures, format="extxyz")
    write(TEST_FILE, test_structures, format="extxyz")

    model = MACEModel(
        name="pt_surface_multihead_finetune",
        model="MACE",
        checkpoint_dir=MODEL_DIR,
        device=DEVICE,
        default_dtype="float32",
    )

    model.multihead_finetune(
        train_file=TRAIN_FILE,
        valid_file=VALID_FILE,
        test_file=TEST_FILE,
        foundation_model=FOUNDATION_MODEL,
        method="mp",
        statistics_file=None,
        work_dir=str(OUTPUT_DIR),
        energy_key="REF_energy",
        forces_key="REF_forces",
        stress_key="REF_stress",
        E0s="foundation",
        batch_size=BATCH_SIZE,
        valid_batch_size=BATCH_SIZE,
        max_num_epochs=EPOCHS,
        lr=LEARNING_RATE,
        ema=True,
        ema_decay=0.99,
        loss="weighted",
        num_workers=0,
        enable_cueq=True,
        distributed=False,
        launch_mode="single",
        nproc_per_node=1,
        seed=RANDOM_SEED,
        extra_args=[
            "--device", DEVICE,
            "--default_dtype", "float32",
            "--checkpoints_dir", str(CHECKPOINT_DIR),
            "--model_dir", str(MODEL_DIR),
            "--results_dir", str(RESULTS_DIR),
            "--log_dir", str(LOG_DIR),
            "--error_table", "PerAtomMAE",
            "--plot", "False",
        ],
    )

    elapsed = time.perf_counter() - start_time
    checkpoint_path = model.expected_checkpoint_path()
    checkpoint_files = list_files(CHECKPOINT_DIR)
    model_files = list_files(MODEL_DIR)
    result_files = list_files(RESULTS_DIR)
    log_files = list_files(LOG_DIR)

    print("MACE multihead finetuning complete")
    print(f"foundation model: {FOUNDATION_MODEL}")
    print(f"input structures: {len(structures)}")
    print(f"train structures: {len(train_structures)}")
    print(f"valid structures: {len(valid_structures)}")
    print(f"test structures: {len(test_structures)}")
    print(f"train file: {TRAIN_FILE}")
    print(f"valid file: {VALID_FILE}")
    print(f"test file: {TEST_FILE}")
    print(f"checkpoint expected: {checkpoint_path}")
    print(f"checkpoint exists: {checkpoint_path.exists() if checkpoint_path else False}")
    print(f"output dir: {OUTPUT_DIR}")
    print("checkpoint files:")
    for file_path in checkpoint_files:
        print(f"  {file_path}")
    print("model files:")
    for file_path in model_files:
        print(f"  {file_path}")
    print("result files:")
    for file_path in result_files:
        print(f"  {file_path}")
    print("log files:")
    for file_path in log_files:
        print(f"  {file_path}")
    print(f"elapsed seconds: {elapsed:.1f}")
    print(f"elapsed minutes: {elapsed / 60:.2f}")


if __name__ == "__main__":
    main()
