from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.db.core import Database
from ase.io import read, write

from al_dirac.io.dataset_io import write_atoms_db


def _reference_results(atoms: Atoms) -> dict[str, Any]:
    results = dict(getattr(atoms.calc, "results", {}) or {})

    for key in ("energy", "free_energy", "stress"):
        if key not in results and key in atoms.info:
            results[key] = atoms.info[key]
    if "forces" not in results and "forces" in atoms.arrays:
        results["forces"] = atoms.arrays["forces"]

    return results


def _is_finite(value: Any, expected_shape: tuple[int, ...] | None = None) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False

    if array.size == 0:
        return False
    if expected_shape is not None and array.shape != expected_shape:
        return False
    return bool(np.isfinite(array).all())


def infer_label_properties(atoms: Atoms) -> tuple[bool, tuple[str, ...]]:
    """Infer available reference labels without evaluating a calculator."""
    results = _reference_results(atoms)
    energy_key = next(
        (
            key
            for key in ("energy", "free_energy")
            if key in results
            and results[key] is not None
            and np.asarray(results[key]).size == 1
            and _is_finite(results[key])
        ),
        None,
    )

    forces = results.get("forces")
    has_forces = (
        forces is not None
        and _is_finite(forces, expected_shape=(len(atoms), 3))
    )

    properties: list[str] = []
    if energy_key is not None:
        properties.append(energy_key)
    if has_forces:
        properties.append("forces")
    if (
        results.get("stress") is not None
        and _is_finite(results["stress"])
    ):
        properties.append("stress")

    return energy_key is not None and has_forces, tuple(properties)


def reference_energy(atoms: Atoms) -> float | None:
    results = _reference_results(atoms)
    for key in ("energy", "free_energy"):
        value = results.get(key)
        if value is None:
            continue
        if np.asarray(value).size == 1 and _is_finite(value):
            return float(value)
    return None


def _build_structure_id(
    base_dir: str | Path,
    file_path: str | Path,
    frame_index: int | None = None,
) -> str:
    base_dir = Path(base_dir).resolve()
    file_path = Path(file_path).resolve()

    try:
        rel_path = file_path.relative_to(base_dir)
    except ValueError:
        rel_path = file_path

    parts = list(rel_path.parts)
    if parts:
        parts[-1] = Path(parts[-1]).stem

    structure_id = "__".join(parts)
    if frame_index is not None:
        structure_id = f"{structure_id}__frame_{frame_index}"

    return structure_id


def _read_file_contains(file_path: Path, keyword: str) -> bool:
    try:
        text = file_path.read_text(errors="ignore")
    except Exception:
        return False
    return keyword in text


def _records_to_atoms_list(records: list[dict[str, Any]]) -> list[Atoms]:
    atoms_list = []
    for record in records:
        source_atoms = record["atoms"]
        reference_results = _reference_results(source_atoms)
        atoms = source_atoms.copy()
        atoms.calc = None

        reference_energy_value = reference_energy(source_atoms)
        if reference_energy_value is not None:
            atoms.info["REF_energy"] = reference_energy_value

        reference_forces = reference_results.get("forces")
        if reference_forces is not None and _is_finite(
            reference_forces,
            expected_shape=(len(source_atoms), 3),
        ):
            atoms.arrays["REF_forces"] = np.asarray(reference_forces, dtype=float)

        reference_stress = reference_results.get("stress")
        if reference_stress is not None and _is_finite(reference_stress):
            atoms.info["REF_stress"] = np.asarray(reference_stress, dtype=float)

        for key in (
            "structure_id",
            "iteration",
            "source",
            "relative_path",
            "is_labeled",
            "system",
            "config_type",
            "energy",
        ):
            if record.get(key) is not None:
                atoms.info[key] = record[key]
        if record.get("label_properties") is not None:
            atoms.info["label_properties"] = ",".join(record["label_properties"])
        atoms_list.append(atoms)
    return atoms_list


def write_structure_records(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    output_format: str,
    split: str = "pool",
    iteration: int | None = None,
    is_labeled: bool | None = None,
    is_selected: bool = False,
    source: str | None = None,
    append: bool = False,
) -> list[int] | Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format in {"aselmdb", "db", "json"}:
        db = connect(output_path, type=output_format)
        row_ids: list[int] = []
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
            row_ids.append(
                write_atoms_db(
                    db=db,
                    atoms=record["atoms"],
                    structure_id=record["structure_id"],
                    split=split,
                    iteration=(
                        int(record.get("iteration", 0))
                        if iteration is None
                        else iteration
                    ),
                    is_labeled=(
                        bool(record.get("is_labeled", False))
                        if is_labeled is None
                        else is_labeled
                    ),
                    is_selected=is_selected,
                    source=str(record.get("source", source)),
                    data=data,
                )
            )
        return row_ids

    if output_format in {"extxyz", "traj"}:
        # aselmdb/db above are inherently append-only (each row is added, not
        # overwritten), so `append` only matters here -- ase.io.write()
        # overwrites an existing extxyz/traj file by default.
        write(
            output_path,
            _records_to_atoms_list(records),
            format=output_format,
            append=append,
        )
        return output_path

    raise ValueError(
        "output_format must be one of: aselmdb, db, json, extxyz, traj."
    )


def parse_structure_records(
    base_dir: str | Path,
    filename_glob: str = "*",
    *,
    recursive: bool = True,
    file_format: str | None = None,
    read_index: int | str = ":",
    iteration: int = 0,
    source: str | None = None,
    keyword_in_filename: str | None = None,
    keyword_in_file: str | None = None,
    require_sibling_filename: str | None = None,
    require_sibling_contains: Mapping[str, str] | None = None,
    skip_existing_ids: set[str] | None = None,
    common_data: dict[str, Any] | None = None,
    is_labeled: bool | None = None,
    raise_on_read_error: bool = False,
) -> list[dict[str, Any]]:
    base_path = Path(base_dir).resolve()
    candidates = (
        base_path.rglob(filename_glob)
        if recursive
        else base_path.glob(filename_glob)
    )
    seen_ids = set() if skip_existing_ids is None else set(skip_existing_ids)
    records: list[dict[str, Any]] = []

    for file_path in candidates:
        file_path = file_path.resolve()
        if not file_path.is_file():
            continue
        if keyword_in_filename is not None and keyword_in_filename not in file_path.name:
            continue
        if keyword_in_file is not None and not _read_file_contains(
            file_path, keyword_in_file
        ):
            continue
        if require_sibling_filename is not None:
            sibling_path = file_path.parent / require_sibling_filename
            if not sibling_path.is_file():
                continue
        if require_sibling_contains is not None:
            missing_required_text = False
            for sibling_filename, keyword in require_sibling_contains.items():
                sibling_path = file_path.parent / sibling_filename
                if not sibling_path.is_file() or not _read_file_contains(
                    sibling_path,
                    keyword,
                ):
                    missing_required_text = True
                    break
            if missing_required_text:
                continue

        try:
            atoms_or_frames = read(str(file_path), format=file_format, index=read_index)
        except Exception:
            if raise_on_read_error:
                raise
            continue

        frames = atoms_or_frames if isinstance(atoms_or_frames, list) else [atoms_or_frames]
        source_value = source or str(file_path.relative_to(base_path))
        relative_path = str(file_path.relative_to(base_path))

        for frame_index, atoms in enumerate(frames):
            inferred_is_labeled, label_properties = infer_label_properties(atoms)
            record_frame_index = frame_index if isinstance(atoms_or_frames, list) else None
            structure_id = _build_structure_id(
                base_dir=base_path,
                file_path=file_path,
                frame_index=record_frame_index,
            )
            if structure_id in seen_ids:
                continue

            record: dict[str, Any] = {
                **(common_data or {}),
                "candidate_index": len(records),
                "structure_id": structure_id,
                "atoms": atoms,
                "iteration": iteration,
                "source": source_value,
                "file_path": str(file_path),
                "relative_path": relative_path,
                "energy": reference_energy(atoms),
                "is_labeled": (
                    inferred_is_labeled if is_labeled is None else is_labeled
                ),
                "label_properties": label_properties,
                "workflow": {
                    **((common_data or {}).get("workflow", {})),
                    "parser": {
                        "base_dir": str(base_path),
                        "file_path": str(file_path),
                        "relative_path": relative_path,
                        "source": source_value,
                        "frame_index": record_frame_index,
                        "filename_glob": filename_glob,
                        "recursive": recursive,
                        "file_format": file_format,
                        "read_index": read_index,
                        "keyword_in_filename": keyword_in_filename,
                        "keyword_in_file": keyword_in_file,
                        "require_sibling_filename": require_sibling_filename,
                        "require_sibling_contains": dict(require_sibling_contains or {}),
                        "label_inferred": inferred_is_labeled,
                        "label_override": is_labeled,
                        "label_properties": label_properties,
                    },
                },
            }
            if record_frame_index is not None:
                record["frame_index"] = record_frame_index

            records.append(record)
            seen_ids.add(structure_id)

    return records


def find_and_store_structures(
    base_dir: str | Path,
    db: Database,
    filename_glob: str = "*",
    *,
    recursive: bool = True,
    file_format: str | None = None,
    read_index: int | str = ":",
    split: str = "pool",
    iteration: int = 0,
    is_labeled: bool | None = None,
    is_selected: bool = False,
    source: str | None = None,
    keyword_in_filename: str | None = None,
    keyword_in_file: str | None = None,
    require_sibling_filename: str | None = None,
    require_sibling_contains: Mapping[str, str] | None = None,
    skip_existing_ids: set[str] | None = None,
    common_data: dict[str, Any] | None = None,
    raise_on_read_error: bool = False,
) -> list[int]:
    row_ids: list[int] = []
    for record in parse_structure_records(
        base_dir=base_dir,
        filename_glob=filename_glob,
        recursive=recursive,
        file_format=file_format,
        read_index=read_index,
        iteration=iteration,
        source=source,
        keyword_in_filename=keyword_in_filename,
        keyword_in_file=keyword_in_file,
        require_sibling_filename=require_sibling_filename,
        require_sibling_contains=require_sibling_contains,
        skip_existing_ids=skip_existing_ids,
        common_data=common_data,
        is_labeled=is_labeled,
        raise_on_read_error=raise_on_read_error,
    ):
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
        row_ids.append(
            write_atoms_db(
                db=db,
                atoms=record["atoms"],
                structure_id=record["structure_id"],
                split=split,
                iteration=iteration,
                is_labeled=bool(record["is_labeled"]),
                is_selected=is_selected,
                source=record["source"],
                data=data,
            )
        )

    return row_ids


def parse_and_write_structures(
    base_dir: str | Path,
    output_paths: Mapping[str, str | Path],
    filename_glob: str = "*",
    *,
    recursive: bool = True,
    file_format: str | None = None,
    read_index: int | str = ":",
    split: str = "pool",
    iteration: int = 0,
    is_labeled: bool | None = None,
    is_selected: bool = False,
    source: str | None = None,
    keyword_in_filename: str | None = None,
    keyword_in_file: str | None = None,
    require_sibling_filename: str | None = None,
    require_sibling_contains: Mapping[str, str] | None = None,
    skip_existing_ids: set[str] | None = None,
    common_data: dict[str, Any] | None = None,
    raise_on_read_error: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, list[int] | Path]]:
    records = parse_structure_records(
        base_dir=base_dir,
        filename_glob=filename_glob,
        recursive=recursive,
        file_format=file_format,
        read_index=read_index,
        iteration=iteration,
        source=source,
        keyword_in_filename=keyword_in_filename,
        keyword_in_file=keyword_in_file,
        require_sibling_filename=require_sibling_filename,
        require_sibling_contains=require_sibling_contains,
        skip_existing_ids=skip_existing_ids,
        common_data=common_data,
        is_labeled=is_labeled,
        raise_on_read_error=raise_on_read_error,
    )

    outputs: dict[str, list[int] | Path] = {}
    for output_format, output_path in output_paths.items():
        outputs[output_format] = write_structure_records(
            records,
            output_path,
            output_format=output_format,
            split=split,
            iteration=iteration,
            is_labeled=is_labeled,
            is_selected=is_selected,
            source=source,
        )

    return records, outputs
