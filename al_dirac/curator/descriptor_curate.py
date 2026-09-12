from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from al_dirac.curator.base import (
    BaseStructureCuration,
    CurationResult,
    CurationStageReport,
)
from al_dirac.features.soap_descriptor import SOAPDescriptor
from al_dirac.features.structure_similarity import StructureSimilarity


class DescriptorCurate(BaseStructureCuration):
    def __init__(
        self,
        soap_descriptor: SOAPDescriptor,
        structure_similarity: StructureSimilarity,
        similarity_threshold: float | None = None,
        distance_threshold: float | None = None,
        position_mode: str = "all",
        n_jobs: int = 1,
    ) -> None:
        super().__init__(method_name="descriptor_curate")
        if similarity_threshold is None and distance_threshold is None:
            raise ValueError(
                "At least one of similarity_threshold or distance_threshold must be provided."
            )

        self.soap_descriptor = soap_descriptor
        self.structure_similarity = structure_similarity
        self.similarity_threshold = similarity_threshold
        self.distance_threshold = distance_threshold
        self.position_mode = position_mode
        self.n_jobs = n_jobs

    def _is_redundant(
        self,
        descriptor: np.ndarray,
        kept_descriptors: list[np.ndarray],
    ) -> bool:
        for kept_descriptor in kept_descriptors:
            if self.similarity_threshold is not None:
                similarity = self.structure_similarity.similarity(
                    descriptor,
                    kept_descriptor,
                )
                if similarity >= self.similarity_threshold:
                    return True

            if self.distance_threshold is not None:
                distance = self.structure_similarity.distance(
                    descriptor,
                    kept_descriptor,
                )
                if distance <= self.distance_threshold:
                    return True

        return False

    def curate(
        self,
        structures: list[Atoms],
        **kwargs: Any,
    ) -> list[Atoms]:
        curated_structures: list[Atoms] = []
        kept_descriptors: list[np.ndarray] = []

        positions_list = kwargs.get("positions_list")

        descriptors = self.soap_descriptor.create_local_batch(
            structures,
            position_mode=self.position_mode,
            positions_list=positions_list,
            n_jobs=self.n_jobs,
        )

        for atoms, descriptor in zip(structures, descriptors):
            if self._is_redundant(descriptor, kept_descriptors):
                continue

            curated_structures.append(atoms)
            kept_descriptors.append(descriptor)

        return curated_structures

    def curate_records(
        self,
        records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CurationResult:
        kept_records: list[dict[str, Any]] = []
        removed_records: list[dict[str, Any]] = []
        kept_descriptors: list[np.ndarray] = []

        structures = [record["atoms"] for record in records]
        descriptors = self.soap_descriptor.create_local_batch(
            structures,
            position_mode=self.position_mode,
            positions_list=kwargs.get("positions_list"),
            n_jobs=self.n_jobs,
        )

        for record, descriptor in zip(records, descriptors):
            if self._is_redundant(descriptor, kept_descriptors):
                updated = self._with_curation_decision(
                    record,
                    {
                        "decision": "removed",
                        "reason": "descriptor_redundant",
                    },
                )
                removed_records.append(updated)
                continue

            updated = self._with_curation_decision(
                record,
                {"decision": "kept"},
            )
            kept_records.append(updated)
            kept_descriptors.append(descriptor)

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
                        "similarity_threshold": self.similarity_threshold,
                        "distance_threshold": self.distance_threshold,
                        "position_mode": self.position_mode,
                        "n_jobs": self.n_jobs,
                    },
                )
            ],
        )
