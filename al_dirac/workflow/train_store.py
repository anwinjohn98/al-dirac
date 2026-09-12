from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ase.db import connect
from ase.db.core import Database

from al_dirac.io.dataset_io import write_atoms_db
from al_dirac.parser.structure_parser import (
    infer_label_properties,
    reference_energy,
    write_structure_records,
)
from al_dirac.workflow.records import workflow_template


def append_records_to_db(
    db: Database,
    records: list[dict[str, Any]],
    *,
    split: str,
    is_labeled: bool | None = None,
    is_selected: bool | None = None,
    source: str = "active_learning",
) -> list[int]:
    row_ids: list[int] = []

    for record in records:
        atoms = record["atoms"]
        data = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "atoms",
                "structure_id",
                "iteration",
                "is_labeled",
                "is_selected",
                "source",
            }
        }
        row_id = write_atoms_db(
            db=db,
            atoms=atoms,
            structure_id=str(
                record.get(
                    "structure_id",
                    f"record_{record.get('candidate_index', 'unknown')}",
                )
            ),
            split=split,
            iteration=int(record.get("iteration", 0)),
            is_labeled=(
                bool(record.get("is_labeled", False))
                if is_labeled is None
                else is_labeled
            ),
            is_selected=(
                bool(record.get("is_selected", False))
                if is_selected is None
                else is_selected
            ),
            source=str(record.get("source", source)),
            data=data,
        )
        row_ids.append(row_id)

    return row_ids


def append_selected_to_db(
    db: Database,
    selected_records: list[dict[str, Any]],
    *,
    split: str = "selected",
    source: str = "active_learning",
) -> list[int]:
    return append_records_to_db(
        db,
        selected_records,
        split=split,
        is_selected=True,
        source=source,
    )


def append_labeled_records_to_train_path(
    train_path: str | Path,
    records: list[dict[str, Any]],
) -> int:
    labeled_records = [record for record in records if bool(record.get("is_labeled", False))]
    if not labeled_records:
        return 0

    train_path = Path(train_path)
    if train_path.suffix != ".aselmdb":
        raise ValueError(
            "Appending parsed or newly labeled records requires train_path "
            "to use the '.aselmdb' format."
        )

    db = connect(str(train_path))
    existing_ids = {
        str(row.structure_id)
        for row in db.select()
        if getattr(row, "structure_id", None) is not None
    }
    new_records = [
        record for record in labeled_records if str(record["structure_id"]) not in existing_ids
    ]
    append_records_to_db(
        db,
        new_records,
        split="train",
        is_labeled=True,
        is_selected=False,
        source="active_learning",
    )
    return len(new_records)


def load_labeled_records_from_train_path(
    train_path: str | Path,
    *,
    iteration: int,
    common_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    train_path = Path(train_path)
    if train_path.suffix != ".aselmdb":
        raise ValueError(
            "Loading seed records from train_path currently requires the '.aselmdb' format."
        )
    if not train_path.exists():
        return []

    records: list[dict[str, Any]] = []
    db = connect(str(train_path))
    for row in db.select():
        atoms = row.toatoms()
        inferred_labeled, label_properties = infer_label_properties(atoms)
        db_labeled = bool(getattr(row, "is_labeled", False))
        if not (db_labeled or inferred_labeled):
            continue

        structure_id = getattr(row, "structure_id", f"train_row_{row.id}")
        records.append(
            {
                **(common_data or {}),
                "candidate_index": len(records),
                "structure_id": structure_id,
                "atoms": atoms,
                "iteration": iteration,
                "source": "train_path",
                "file_path": str(train_path),
                "energy": reference_energy(atoms),
                "is_labeled": True,
                "label_properties": label_properties,
                "workflow": workflow_template(
                    {
                        **((common_data or {}).get("workflow", {})),
                        "parser": None,
                        "sampler": None,
                        "seed_selection": {
                            "source": "train_path",
                            "row_id": row.id,
                        },
                    }
                ),
            }
        )

    return records


def split_aselmdb_train_valid(
    train_path: str | Path,
    *,
    valid_fraction: float = 0.1,
    seed: int | None = None,
) -> tuple[Path, Path]:
    # mace_run_train's valid_fraction auto-split (get_dataset_from_xyz) only
    # supports ASE-readable formats -- for .aselmdb train files,
    # head_config.collections is never populated, so training crashes with
    # "AttributeError: 'NoneType' object has no attribute 'valid'" unless an
    # explicit valid_file is given. Split and write one ourselves instead.
    train_path = Path(train_path)
    records = load_labeled_records_from_train_path(train_path, iteration=0)
    if not records:
        raise ValueError(f"No labeled records found in {train_path}.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * valid_fraction))
    valid_records = shuffled[:n_valid]
    train_records = shuffled[n_valid:]

    split_train_path = train_path.with_name(f"{train_path.stem}_train_split.aselmdb")
    split_valid_path = train_path.with_name(f"{train_path.stem}_valid_split.aselmdb")
    write_structure_records(
        train_records, split_train_path, output_format="aselmdb", split="train"
    )
    write_structure_records(
        valid_records, split_valid_path, output_format="aselmdb", split="valid"
    )
    return split_train_path, split_valid_path


def train_path_has_data(train_path: str | Path | None) -> bool:
    if train_path is None:
        return False

    path = Path(train_path)
    if not path.exists():
        return False
    if path.suffix == ".aselmdb":
        return len(connect(str(path))) > 0
    return path.is_file() and path.stat().st_size > 0
