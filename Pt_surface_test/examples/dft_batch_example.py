from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read

from al_dirac.dft.batch import BatchDFTRunner
from al_dirac.dft.vasp import VASPLabeler

SELECTED_CANDIDATES_FILE = Path("Pt_surface_test/real_example_outputs/uncertainty_selection/selected_candidates.extxyz")
SUBMISSION_TEMPLATE = Path("Pt_surface_test/examples/vasp_batch.run")

OUTPUT_DIR = Path("Pt_surface_test/real_example_outputs/dft_batch")

BATCH_SIZE = 3

# Settings mirror the real INCAR/KPOINTS at Pt_surface_test/DFT_data/710/.
VASP_CALCULATOR_KWARGS = {
    "xc": "PBE",
    "encut": 400,
    "ediff": 1e-5,
    # Single-point labeling, not relaxation: the whole point of active
    # learning is to label the exact candidate geometry that was queried, not
    # a relaxed one -- ibrion/isif/potim/ediffg are irrelevant with nsw=0.
    "ibrion": -1,
    "nsw": 0,
    "ismear": 1,
    "sigma": 0.2,
    "prec": "Accurate",
    "lreal": False,
    "lwave": False,
    "algo": "VeryFast",
    "npar": 16,
    "nelm": 300,
    "kpts": (5, 1, 1),
    "gamma": True,
}

def load_selected_records() -> list[dict[str, Any]]:
    structures = read(SELECTED_CANDIDATES_FILE, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    records = []
    for atoms in structures:
        record = dict(atoms.info)
        record["atoms"] = atoms
        records.append(record)
    return records

def main() -> None:
    if not SELECTED_CANDIDATES_FILE.exists():
        raise FileNotFoundError("Run examples/uncertainty_selection_example.py first.")
    if not SUBMISSION_TEMPLATE.exists():
        raise FileNotFoundError(f"Missing submission template: {SUBMISSION_TEMPLATE}")

    records = load_selected_records()
    structures = [record["atoms"] for record in records]

    labeler = VASPLabeler(calculator_kwargs=VASP_CALCULATOR_KWARGS)
    dft_runner = BatchDFTRunner(
        labeler=labeler,
        batch_size=BATCH_SIZE,
        submission_template=SUBMISSION_TEMPLATE,
        run_command_template="mpirun -np $SLURM_NTASKS vasp_std",
    )

    prepared_batches = dft_runner.prepare_batches(
        structures=structures,
        root_dir=OUTPUT_DIR / "jobs",
        records=records,
    )

    print(f"selected candidates: {len(structures)}")
    print(f"batches prepared: {len(prepared_batches)}")
    for batch in prepared_batches:
        print(f"  batch {batch['batch_index']}: {batch['batch_dir']} ({len(batch['jobs'])} jobs)")
        print(f"    submit script: {batch['submit_script']}")
        for job in batch["jobs"]:
            job_dir = Path(job["job_dir"])
            files = sorted(p.name for p in job_dir.iterdir()) if job_dir.exists() else []
            print(f"    job: {job_dir} -> files: {files}")


if __name__ == "__main__":
    main()
