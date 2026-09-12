from __future__ import annotations

from typing import Any

from ase import Atoms
from ase.db.core import Database

def write_atoms_db(
    db: Database,
    atoms: Atoms,
    structure_id: str,
    split: str = "pool",
    iteration: int = 0,
    is_labeled: bool = False,
    is_selected: bool = False,
    source: str | None = None,
    data: dict[str, Any] | None = None,
) -> int:
    key_value_pairs = {
        "structure_id": structure_id,
        "split": split,
        "iteration": iteration,
        "is_labeled": is_labeled,
        "is_selected": is_selected,
    }

    if source is not None:
        key_value_pairs["source"] = source

    return db.write(
        atoms,
        key_value_pairs=key_value_pairs,
        data=data or {},
    )
