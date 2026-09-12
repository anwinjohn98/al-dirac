from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.db import connect
from ase.io import write

from al_dirac.io.dataset_io import write_atoms_db
from al_dirac.parser.structure_parser import (
    find_and_store_structures,
    parse_structure_records,
)


def evaluate(atoms: Atoms) -> Atoms:
    evaluated = atoms.copy()
    evaluated.calc = atoms.calc
    evaluated.get_potential_energy()
    evaluated.get_forces()
    return evaluated


def make_example_inputs(base_dir: Path) -> None:
    emt_dir = base_dir / "finished_emt_cu4"
    lj_dir = base_dir / "finished_lj_ar_md"
    unlabeled_dir = base_dir / "unlabeled_co"
    ignored_dir = base_dir / "unfinished_emt_cu4"

    for path in [emt_dir, lj_dir, unlabeled_dir, ignored_dir]:
        path.mkdir(parents=True, exist_ok=True)

    cu4 = Atoms(
        "Cu4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, 2.5],
        ],
        cell=[8, 8, 8],
        pbc=False,
    )
    cu4.calc = EMT()
    write(emt_dir / "cu4_relaxed.extxyz", evaluate(cu4), format="extxyz")
    write(ignored_dir / "cu4_relaxed.extxyz", evaluate(cu4), format="extxyz")
    (emt_dir / "metadata.txt").write_text("finished\n")

    frames = []
    for shift in [0.0, 0.1, 0.2]:
        ar4 = Atoms(
            "Ar4",
            positions=[
                [0.0, 0.0, 0.0],
                [3.8 + shift, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 0.0, 3.8],
            ],
            cell=[12, 12, 12],
            pbc=False,
        )
        ar4.calc = LennardJones()
        frames.append(evaluate(ar4))
    write(lj_dir / "ar_md.extxyz", frames, format="extxyz")
    (lj_dir / "metadata.txt").write_text("finished\n")

    co = Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]],
        cell=[8, 8, 8],
        pbc=False,
    )
    write(unlabeled_dir / "co_initial.extxyz", co, format="extxyz")


def write_records_extxyz(records: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    atoms_list = []
    for record in records:
        atoms = record["atoms"].copy()
        atoms.info["structure_id"] = record["structure_id"]
        atoms.info["is_labeled"] = record["is_labeled"]
        atoms.info["label_properties"] = ",".join(record["label_properties"])
        atoms.info["source"] = record["source"]
        atoms_list.append(atoms)

    write(output_path, atoms_list, format="extxyz")
    return output_path


def write_records_aselmdb(records: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    db = connect(output_path)

    for record in records:
        data = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "atoms",
                "candidate_index",
                "structure_id",
                "iteration",
                "is_labeled",
                "source",
            }
        }

        write_atoms_db(
            db,
            record["atoms"],
            structure_id=record["structure_id"],
            split="pool",
            iteration=record["iteration"],
            is_labeled=record["is_labeled"],
            is_selected=False,
            source=record["source"],
            data=data,
        )

    return output_path


def write_db_rows_extxyz(db, row_ids: list[int], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atoms_list = []

    for row_id in row_ids:
        row = db.get(row_id)
        atoms = row.toatoms()
        atoms.info["structure_id"] = row.structure_id
        atoms.info["is_labeled"] = row.is_labeled
        atoms.info["source"] = row.source
        atoms_list.append(atoms)

    write(output_path, atoms_list, format="extxyz")
    return output_path


def print_records(title: str, records: list[dict]) -> None:
    print(title)
    print(f"records: {len(records)}")
    for record in records:
        print(
            f"{record['structure_id']}: "
            f"path={record['relative_path']}, "
            f"is_labeled={record['is_labeled']}, "
            f"label_properties={record['label_properties']}"
        )


def main() -> None:
    input_dir = Path("example_outputs/parser/inputs")
    output_dir = Path("example_outputs/parser/outputs")
    manual_extxyz_path = output_dir / "manual_parsed_records.extxyz"
    manual_db_path = output_dir / "manual_parsed_records.aselmdb"
    direct_db_path = output_dir / "direct_find_and_store.aselmdb"
    direct_extxyz_path = output_dir / "direct_find_and_store.extxyz"

    make_example_inputs(input_dir)

    records = parse_structure_records(
        base_dir=input_dir,
        filename_glob="*.extxyz",
        recursive=True,
        file_format="extxyz",
        read_index=":",
        iteration=0,
        source="parser_example",
        keyword_in_filename=None,
        keyword_in_file=None,
        require_sibling_filename=None,
        skip_existing_ids=None,
        common_data={"project": "al_dirac_parser_example"},
        is_labeled=None,
        raise_on_read_error=True,
    )
    print_records("all parsed records from parse_structure_records", records)

    write_records_extxyz(records, manual_extxyz_path)
    write_records_aselmdb(records, manual_db_path)

    finished_records = parse_structure_records(
        base_dir=input_dir,
        filename_glob="*.extxyz",
        recursive=True,
        file_format="extxyz",
        read_index=":",
        require_sibling_filename="metadata.txt",
    )
    print_records("only records with metadata.txt sibling", finished_records)

    relaxed_records = parse_structure_records(
        base_dir=input_dir,
        filename_glob="*.extxyz",
        recursive=True,
        file_format="extxyz",
        read_index=":",
        keyword_in_filename="relaxed",
    )
    print_records("only files with 'relaxed' in filename", relaxed_records)

    skipped_records = parse_structure_records(
        base_dir=input_dir,
        filename_glob="*.extxyz",
        recursive=True,
        file_format="extxyz",
        read_index=":",
        skip_existing_ids={records[0]["structure_id"]},
    )
    print_records("all records except first structure_id", skipped_records)

    if direct_db_path.exists():
        direct_db_path.unlink()

    db = connect(direct_db_path)
    row_ids = find_and_store_structures(
        base_dir=input_dir,
        db=db,
        filename_glob="*.extxyz",
        recursive=True,
        file_format="extxyz",
        read_index=":",
        split="pool",
        iteration=0,
        is_labeled=None,
        is_selected=False,
        source="parser_example_direct_db",
        keyword_in_filename=None,
        keyword_in_file=None,
        require_sibling_filename="metadata.txt",
        skip_existing_ids=None,
        common_data={"project": "al_dirac_parser_example"},
        raise_on_read_error=True,
    )
    write_db_rows_extxyz(db, row_ids, direct_extxyz_path)

    print("outputs:")
    print(f"manual extxyz: {manual_extxyz_path}")
    print(f"manual aselmdb: {manual_db_path}")
    print(f"direct extxyz: {direct_extxyz_path}")
    print(f"direct aselmdb: {direct_db_path}")
    print(f"direct db rows: {len(row_ids)}")


if __name__ == "__main__":
    main()
