from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from al_dirac.parser.structure_parser import (
    parse_and_write_structures,
)


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def print_record_summary(records: list[dict[str, Any]]) -> None:
    labeled = [record for record in records if record["is_labeled"]]
    unlabeled = [record for record in records if not record["is_labeled"]]

    print(f"records: {len(records)}")
    print(f"labeled: {len(labeled)}")
    print(f"unlabeled: {len(unlabeled)}")

    print("first records:")
    for record in records[:5]:
        print(
            f"  {record['structure_id']} | "
            f"labeled={record['is_labeled']} | "
            f"labels={record['label_properties']} | "
            f"file={record['relative_path']}"
        )


def main() -> None:
    input_dir = Path("Pt_surface_test")
    output_dir = Path("real_example_outputs/parser")
    train_db_path = output_dir / "train.aselmdb"
    train_extxyz_path = output_dir / "train.extxyz"

    clean_output_dir(output_dir)

    records, outputs = parse_and_write_structures(
        base_dir=input_dir,
        output_paths={
            "aselmdb": train_db_path,
            "extxyz": train_extxyz_path,
        },
        filename_glob="*/vasprun.xml",
        recursive=False,
        file_format="vasp-xml",
        read_index=-1,
        iteration=0,
        source="pt_o_surface_relaxation",
        keyword_in_filename=None,
        keyword_in_file=None,
        require_sibling_filename="CONTCAR",
        require_sibling_contains={"OUTCAR": "reached required accuracy"},
        skip_existing_ids=None,
        common_data={
            "system": "Pt surface",
            "config_type": "Pt_surface",
        },
        is_labeled=None,
        raise_on_read_error=False,
    )

    print("real parser example complete")
    print_record_summary(records)
    print("outputs:")
    print(f"  train extxyz: {outputs['extxyz']}")
    print(f"  train db: {train_db_path}")
    print(f"  db rows written: {len(outputs['aselmdb'])}")


if __name__ == "__main__":
    main()
