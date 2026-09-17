from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from al_dirac.models.mace_model import MACEModel
from al_dirac.parser.structure_parser import write_structure_records
from al_dirac.workflow.train_store import load_labeled_records_from_train_path

INPUT_ASELMDB = Path("real_example_outputs/parser/train.aselmdb")
OUTPUT_DIR = Path("real_example_outputs/format_comparison")

DEVICE = "cuda"
EPOCHS = 5
BATCH_SIZE = 2
RANDOM_SEED = 7
VALID_FRACTION = 0.2
E0S = "{78: -0.60591152}"

COMMON_KWARGS = dict(
    statistics_file=None,
    hidden_irreps="32x0e",
    atomic_numbers=[78],
    energy_key="REF_energy",
    forces_key="REF_forces",
    stress_key="REF_stress",
    E0s=E0S,
    batch_size=BATCH_SIZE,
    valid_batch_size=BATCH_SIZE,
    max_num_epochs=EPOCHS,
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
)


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Same shuffled train/valid split, written to both formats, so the only
    # difference between the two runs below is the train_file/valid_file
    # format itself.
    records = load_labeled_records_from_train_path(INPUT_ASELMDB, iteration=0)
    shuffled = list(records)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * VALID_FRACTION))
    valid_records = shuffled[:n_valid]
    train_records = shuffled[n_valid:]

    train_aselmdb = OUTPUT_DIR / "train.aselmdb"
    valid_aselmdb = OUTPUT_DIR / "valid.aselmdb"
    train_extxyz = OUTPUT_DIR / "train.extxyz"
    valid_extxyz = OUTPUT_DIR / "valid.extxyz"

    write_structure_records(train_records, train_aselmdb, output_format="aselmdb", split="train")
    write_structure_records(valid_records, valid_aselmdb, output_format="aselmdb", split="valid")
    write_structure_records(train_records, train_extxyz, output_format="extxyz")
    write_structure_records(valid_records, valid_extxyz, output_format="extxyz")

    results = {}
    for fmt, train_file, valid_file in (
        ("aselmdb", train_aselmdb, valid_aselmdb),
        ("extxyz", train_extxyz, valid_extxyz),
    ):
        run_dir = OUTPUT_DIR / fmt
        checkpoint_dir = run_dir / "checkpoints"
        model_dir = run_dir / "models"
        results_dir = run_dir / "results"
        log_dir = run_dir / "logs"
        for d in (checkpoint_dir, model_dir, results_dir, log_dir):
            d.mkdir(parents=True, exist_ok=True)

        model = MACEModel(
            name=f"pt_surface_compare_{fmt}",
            model="MACE",
            checkpoint_dir=model_dir,
            device=DEVICE,
            default_dtype="float32",
        )

        import time
        start = time.perf_counter()
        model.train(
            train_file=train_file,
            valid_file=valid_file,
            work_dir=str(run_dir),
            **COMMON_KWARGS,
            extra_args=[
                "--device", DEVICE,
                "--default_dtype", "float32",
                "--checkpoints_dir", str(checkpoint_dir),
                "--model_dir", str(model_dir),
                "--results_dir", str(results_dir),
                "--log_dir", str(log_dir),
                "--error_table", "PerAtomMAE",
                "--plot", "False",
            ],
        )
        elapsed = time.perf_counter() - start

        result_file = results_dir / f"pt_surface_compare_{fmt}_run-{RANDOM_SEED}_train.txt"
        last_eval = None
        if result_file.exists():
            for line in result_file.read_text().splitlines():
                row = json.loads(line)
                if row.get("mode") == "eval":
                    last_eval = row
        results[fmt] = {"elapsed_seconds": elapsed, "final_eval": last_eval}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
