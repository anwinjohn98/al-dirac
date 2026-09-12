from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from al_dirac.curator.base import (
    BaseStructureCuration,
    CurationResult,
    CurationStageReport,
)


class CheapCurate(BaseStructureCuration):
    def __init__(
        self,
        position_decimals: int = 6,
        cell_decimals: int = 6,
        energy_decimals: int | None = None,
        energy_bin_width: float | None = None,
        energy_tolerance: float | None = None,
        position_tolerance: float | None = None,
    ) -> None:
        super().__init__(method_name="cheap_curate")
        self.position_decimals = position_decimals
        self.cell_decimals = cell_decimals
        self.energy_decimals = energy_decimals
        # energy_decimals only ever gives power-of-10 bin widths (round(x, n)).
        # energy_bin_width allows an arbitrary width (e.g. 2 eV) via
        # divide-round-multiply, and takes priority over energy_decimals when set.
        self.energy_bin_width = energy_bin_width
        self.energy_tolerance = energy_tolerance
        # position_decimals is exact-match (rounded-tuple) and breaks when an
        # atom sits at/near a periodic cell boundary (a valid-but-different
        # wrapped coordinate reads as a large spurious displacement).
        # position_tolerance instead compares corresponding atoms via the true
        # minimum-image distance (ase.geometry.find_mic), so a same-bucket
        # pair only counts as a position match when every atom's minimum-image
        # displacement is within this tolerance.
        self.position_tolerance = position_tolerance

    def _composition_signature(self, atoms: Atoms) -> tuple[tuple[int, int], ...]:
        counts = Counter(atoms.get_atomic_numbers().tolist())
        return tuple(sorted(counts.items()))

    def _wrapped_positions(self, atoms: Atoms) -> np.ndarray:
        # Wrap a copy (not the caller's atoms) into the primary cell first, so
        # an atom that merely crossed a periodic boundary between frames
        # doesn't register as a large spurious displacement.
        wrapped = atoms.copy()
        wrapped.wrap()
        return wrapped.get_positions()

    def _canonical_order(self, atoms: Atoms) -> np.ndarray:
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        positions = np.round(self._wrapped_positions(atoms), self.position_decimals)

        return np.lexsort(
            (
                positions[:, 2],
                positions[:, 1],
                positions[:, 0],
                numbers,
            )
        )

    def _get_energy(self, atoms: Atoms) -> float | None:
        if "energy" in atoms.info:
            return float(atoms.info["energy"])
        if "energy_dft" in atoms.info:
            return float(atoms.info["energy_dft"])
        return None

    def _cell_signature(self, atoms: Atoms) -> tuple[tuple[float, ...], ...]:
        cell = np.round(atoms.get_cell().array, self.cell_decimals)
        return tuple(map(tuple, cell))

    def _rounded_energy(self, atoms: Atoms) -> float | None:
        energy = self._get_energy(atoms)
        if energy is None:
            return None
        if self.energy_bin_width is not None:
            return round(energy / self.energy_bin_width) * self.energy_bin_width
        if self.energy_decimals is None:
            return energy
        return round(energy, self.energy_decimals)

    def _position_signature(self, atoms: Atoms) -> tuple[tuple[float, ...], ...]:
        order = self._canonical_order(atoms)
        positions = np.round(self._wrapped_positions(atoms), self.position_decimals)
        return tuple(map(tuple, positions[order]))

    def _min_image_max_displacement(self, atoms_a: Atoms, atoms_b: Atoms) -> float:
        # Compare atoms by their raw index -- same-trajectory frames keep a
        # consistent atom order, and resorting independently per frame (via
        # _canonical_order) is unstable when nearly-degenerate positions
        # round to ties, which flips the sort order between frames and
        # produces spurious large "displacements" for atoms that barely
        # moved. Use the true minimum-image convention so a coordinate that
        # merely sits on the opposite side of a periodic boundary isn't
        # mistaken for a real displacement.
        diffs = atoms_a.get_positions() - atoms_b.get_positions()
        _, distances = find_mic(diffs, atoms_a.get_cell(), atoms_a.get_pbc())
        return float(np.max(distances)) if len(distances) else 0.0

    def _positions_match(self, existing: Atoms, atoms: Atoms) -> bool:
        if self.position_tolerance is not None:
            return self._min_image_max_displacement(existing, atoms) <= self.position_tolerance
        return self._position_signature(existing) == self._position_signature(atoms)

    def _same_bucket(self, atoms: Atoms) -> tuple[Any, ...]:
        return (
            len(atoms),
            self._composition_signature(atoms),
            tuple(bool(x) for x in atoms.get_pbc()),
            self._cell_signature(atoms),
            self._rounded_energy(atoms),
        )

    def curate(
        self,
        structures: list[Atoms],
        **kwargs: Any,
    ) -> list[Atoms]:
        curated_structures: list[Atoms] = []
        seen_by_bucket: dict[tuple[Any, ...], list[Atoms]] = {}

        for atoms in structures:
            bucket = self._same_bucket(atoms)
            existing_structures = seen_by_bucket.setdefault(bucket, [])

            energy = self._get_energy(atoms)

            is_duplicate = False
            for existing in existing_structures:
                if not self._positions_match(existing, atoms):
                    continue

                if self.energy_tolerance is not None:
                    existing_energy = self._get_energy(existing)
                    if energy is None or existing_energy is None:
                        continue
                    if abs(energy - existing_energy) > self.energy_tolerance:
                        continue

                is_duplicate = True
                break

            if is_duplicate:
                continue

            curated_structures.append(atoms)
            existing_structures.append(atoms)

        return curated_structures

    def curate_records(
        self,
        records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CurationResult:
        kept_records: list[dict[str, Any]] = []
        removed_records: list[dict[str, Any]] = []
        seen_by_bucket: dict[tuple[Any, ...], list[tuple[Atoms, dict[str, Any]]]] = {}

        for record in records:
            atoms = record["atoms"]
            bucket = self._same_bucket(atoms)
            existing_structures = seen_by_bucket.setdefault(bucket, [])

            energy = self._get_energy(atoms)

            duplicate_of: str | int | None = None
            for existing, existing_record in existing_structures:
                if not self._positions_match(existing, atoms):
                    continue

                if self.energy_tolerance is not None:
                    existing_energy = self._get_energy(existing)
                    if energy is None or existing_energy is None:
                        continue
                    if abs(energy - existing_energy) > self.energy_tolerance:
                        continue

                duplicate_of = existing_record.get(
                    "structure_id",
                    existing_record.get("candidate_index"),
                )
                break

            if duplicate_of is not None:
                updated = self._with_curation_decision(
                    record,
                    {
                        "decision": "removed",
                        "reason": "duplicate",
                        "duplicate_of": duplicate_of,
                    },
                )
                removed_records.append(updated)
                continue

            updated = self._with_curation_decision(
                record,
                {"decision": "kept"},
            )
            kept_records.append(updated)
            existing_structures.append((atoms, updated))

        return CurationResult(
            kept_records=kept_records,
            removed_records=removed_records,
            stage_reports=[
                CurationStageReport(
                    stage=self.method_name,
                    input_count=len(records),
                    kept_count=len(kept_records),
                    removed_count=len(removed_records),
                    details={
                        "position_decimals": self.position_decimals,
                        "position_tolerance": self.position_tolerance,
                        "cell_decimals": self.cell_decimals,
                        "energy_decimals": self.energy_decimals,
                        "energy_bin_width": self.energy_bin_width,
                        "energy_tolerance": self.energy_tolerance,
                    },
                )
            ],
        )
