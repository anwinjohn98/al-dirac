from __future__ import annotations

import shutil
from pathlib import Path

from ase import Atoms
from ase.io import read

from al_dirac.curator.cheap_curate import CheapCurate
from al_dirac.workflow.outputs import write_records_extxyz

# Two real VASP relaxation trajectories with very different character:
# - CONVERGED_TAIL: last few ionic steps of a finished relaxation, energies
#   within ~0.002 eV of each other -- CheapCurate should treat these as
#   duplicates once position_tolerance handles the periodic-boundary jitter.
# - RELAXATION: an earlier, still-relaxing trajectory spanning ~1.24 eV --
#   CheapCurate should NOT collapse these into a single structure just
#   because they land in the same coarse energy bin.
TRAJECTORIES = {
    "converged_tail": Path(
        "/scratch/gautschi/john51/al_dirac/Pt_surface_test/DFT_data/710/OUTCAR"
    ),
    "relaxation": Path(
        "/scratch/gautschi/john51/al_dirac/Pt_surface_test/DFT_data/710/OLD_01/OUTCAR"
    ),
}

OUTPUT_DIR = Path("Pt_surface_test/real_example_outputs/cheap_curate_trajectory")

ENERGY_BIN_WIDTHS = [0.5, 1.0, 2.0]
POSITION_TOLERANCE = 0.05
POSITION_DECIMALS = 1


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_records(name: str, outcar_path: Path) -> list[dict]:
    structures = read(outcar_path, index=":")
    if isinstance(structures, Atoms):
        structures = [structures]

    records = []
    for index, atoms in enumerate(structures):
        atoms.info["energy"] = atoms.get_potential_energy()
        records.append(
            {
                "atoms": atoms,
                "structure_id": f"{name}_{index}",
                "candidate_index": index,
            }
        )
    return records


def print_stage_summary(result) -> None:
    for report in result.stage_reports:
        print(
            f"    {report.stage}: "
            f"input={report.input_count}, "
            f"kept={report.kept_count}, "
            f"removed={report.removed_count}"
        )


def main() -> None:
    clean_output_dir(OUTPUT_DIR)

    for traj_name, outcar_path in TRAJECTORIES.items():
        if not outcar_path.exists():
            raise FileNotFoundError(outcar_path)

        records = load_records(traj_name, outcar_path)
        energies = [record["atoms"].info["energy"] for record in records]

        print(f"trajectory: {traj_name} ({outcar_path})")
        print(f"  n_structures: {len(records)}")
        print(f"  energies: {[f'{e:.6f}' for e in energies]}")
        print(f"  energy range: {max(energies) - min(energies):.6f} eV")

        traj_output_dir = OUTPUT_DIR / traj_name
        traj_output_dir.mkdir(parents=True, exist_ok=True)
        write_records_extxyz(records, traj_output_dir / "input_structures.extxyz")

        for energy_bin_width in ENERGY_BIN_WIDTHS:
            for position_tolerance in (None, POSITION_TOLERANCE):
                curator = CheapCurate(
                    position_decimals=POSITION_DECIMALS,
                    energy_bin_width=energy_bin_width,
                    position_tolerance=position_tolerance,
                )
                result = curator.curate_records(records)

                tag = f"bin{energy_bin_width}_tol{position_tolerance}"
                write_records_extxyz(
                    result.kept_records, traj_output_dir / f"kept_{tag}.extxyz"
                )

                print(
                    f"  energy_bin_width={energy_bin_width}, "
                    f"position_tolerance={position_tolerance}"
                )
                print_stage_summary(result)

        print()

    print("cheap_curate_trajectory_example complete")
    print(f"outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
