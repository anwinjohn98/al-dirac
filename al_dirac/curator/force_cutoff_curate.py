from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from al_dirac.curator.base import (
    BaseStructureCuration,
    CurationResult,
    CurationStageReport,
)


class ForceCutoffCurate(BaseStructureCuration):
    def __init__(self, max_force: float) -> None:
        super().__init__(method_name="force_cutoff_curate")
        self.max_force = max_force

    def _max_force(self, atoms: Atoms) -> float | None:
        # Constrained atoms (e.g. FixAtoms on frozen slab layers) can report
        # large raw forces that are physically irrelevant since they can't
        # move -- apply the same constraint-adjustment ase.Atoms.get_forces()
        # uses, so a frozen substrate atom never triggers a false positive.
        if atoms.calc is not None and "forces" in atoms.calc.results:
            forces = atoms.get_forces(apply_constraint=True)
        else:
            forces = None
            if "REF_forces" in atoms.arrays:
                forces = atoms.arrays["REF_forces"]
            elif "forces" in atoms.arrays:
                forces = atoms.arrays["forces"]
            if forces is None:
                return None
            forces = np.array(forces, dtype=float)
            for constraint in atoms.constraints:
                constraint.adjust_forces(atoms, forces)
        return float(np.max(np.linalg.norm(np.asarray(forces), axis=1)))

    def _passes(self, atoms: Atoms) -> bool:
        max_force = self._max_force(atoms)
        if max_force is None:
            return True
        return max_force <= self.max_force

    def curate(self, structures: list[Atoms], **kwargs: Any) -> list[Atoms]:
        return [atoms for atoms in structures if self._passes(atoms)]

    def curate_records(
        self,
        records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CurationResult:
        kept_records: list[dict[str, Any]] = []
        removed_records: list[dict[str, Any]] = []

        for record in records:
            atoms = record["atoms"]
            if self._passes(atoms):
                kept_records.append(
                    self._with_curation_decision(record, {"decision": "kept"})
                )
                continue

            removed_records.append(
                self._with_curation_decision(
                    record,
                    {
                        "decision": "removed",
                        "reason": "max_force_exceeded",
                        "max_force": self._max_force(atoms),
                    },
                )
            )

        return CurationResult(
            kept_records=kept_records,
            removed_records=removed_records,
            stage_reports=[
                CurationStageReport(
                    stage=self.method_name,
                    input_count=len(records),
                    kept_count=len(kept_records),
                    removed_count=len(removed_records),
                    details={"max_force": self.max_force},
                )
            ],
        )
