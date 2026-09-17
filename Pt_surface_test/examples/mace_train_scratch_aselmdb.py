from __future__ import annotations

import shutil
import time
from pathlib import Path

from al_dirac.models.mace_model import MACEModel

INPUT_FILE = Path("Pt_surface_test/real_example_outputs/parser/train.aselmdb")

OUTPUT_DIR = Path("Pt_surface_test/real_example_outputs/mace_scratch_aselmdb")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
LOG_DIR = OUTPUT_DIR / "logs"

DEVICE = "cuda"
EPOCHS = 5
BATCH_SIZE = 2
RANDOM_SEED = 7


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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    model = MACEModel(
        name="pt_surface_scratch_aselmdb",
        model="MACE",
        checkpoint_dir=MODEL_DIR,
        device=DEVICE,
        default_dtype="float32",
    )

    # .aselmdb (unlike extxyz) requires atomic_numbers explicitly -- mace_run_train
    # does not auto-infer the element set for .aselmdb/.h5 inputs.
    # No valid_file given -- MACEModel auto-splits INPUT_FILE (.aselmdb)
    # into a train/valid pair (mace_run_train's own valid_fraction
    # auto-split does not support .aselmdb train files).
    model.train(
        train_file=INPUT_FILE,
        statistics_file=None,
        work_dir=str(OUTPUT_DIR),
        hidden_irreps="32x0e",
        atomic_numbers=[78],
        energy_key="REF_energy",
        forces_key="REF_forces",
        stress_key="REF_stress",
        # Real isolated-atom Pt energy from VASP (ISPIN=2, converged) --
        # "average" is not supported for .aselmdb/.h5 train_file input.
        E0s="{78: -0.60591152}",
        batch_size=BATCH_SIZE,
        valid_batch_size=BATCH_SIZE,
        max_num_epochs=EPOCHS,
        valid_fraction=0.2,
        r_max=5.0,
        lr=0.01,
        ema=True,
        ema_decay=0.99,
        loss="weighted",
        num_channels=32,
        max_L=0,
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

    print("MACE scratch training (.aselmdb) complete")
    print(f"input file: {INPUT_FILE}")
    print(f"checkpoint expected: {checkpoint_path}")
    print(f"checkpoint exists: {checkpoint_path.exists() if checkpoint_path else False}")
    print(f"output dir: {OUTPUT_DIR}")
    print("checkpoint files:")
    for file_path in list_files(CHECKPOINT_DIR):
        print(f"  {file_path}")
    print("model files:")
    for file_path in list_files(MODEL_DIR):
        print(f"  {file_path}")
    print("result files:")
    for file_path in list_files(RESULTS_DIR):
        print(f"  {file_path}")
    print("log files:")
    for file_path in list_files(LOG_DIR):
        print(f"  {file_path}")
    print(f"elapsed seconds: {elapsed:.1f}")
    print(f"elapsed minutes: {elapsed / 60:.2f}")


if __name__ == "__main__":
    main()
