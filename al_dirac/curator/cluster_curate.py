from __future__ import annotations

from typing import Any

from ase import Atoms

from al_dirac.curator.base import (
    BaseStructureCuration,
    CurationResult,
    CurationStageReport,
)
from al_dirac.features.representative_selection import RepresentativeSelection
from al_dirac.features.similarity_graph import SimilarityGraph
from al_dirac.features.soap_descriptor import SOAPDescriptor
from al_dirac.features.structure_similarity import StructureSimilarity


class ClusterCurate(BaseStructureCuration):
    def __init__(
        self,
        soap_descriptor: SOAPDescriptor,
        structure_similarity: StructureSimilarity,
        similarity_graph: SimilarityGraph,
        representative_selection: RepresentativeSelection,
        position_mode: str = "all",
        n_jobs: int = 1,
    ) -> None:
        super().__init__(method_name="cluster_curate")
        self.soap_descriptor = soap_descriptor
        self.structure_similarity = structure_similarity
        self.similarity_graph = similarity_graph
        self.representative_selection = representative_selection
        self.position_mode = position_mode
        self.n_jobs = n_jobs

    def curate(
        self,
        structures: list[Atoms],
        records: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[Atoms]:
        if not structures:
            return []

        if records is None:
            records = [{"atoms": atoms, "candidate_index": i} for i, atoms in enumerate(structures)]

        if len(records) != len(structures):
            raise ValueError("records and structures must have the same length.")

        descriptors = self.soap_descriptor.create_local_batch(
            structures,
            position_mode=self.position_mode,
            positions_list=kwargs.get("positions_list"),
            n_jobs=self.n_jobs,
        )

        distance_matrix = self.structure_similarity.distance_matrix(descriptors)
        clusters = self.similarity_graph.cluster_from_distance_matrix(distance_matrix)
        representative_records = self.representative_selection.select_representatives(
            clusters,
            records,
        )

        return [record["atoms"] for record in representative_records]

    def curate_records(
        self,
        records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CurationResult:
        if not records:
            return CurationResult(
                kept_records=[],
                removed_records=[],
                stage_reports=[
                    CurationStageReport(
                        stage=self.method_name,
                        input_count=0,
                        kept_count=0,
                        removed_count=0,
                    )
                ],
            )

        structures = [record["atoms"] for record in records]
        descriptors = self.soap_descriptor.create_local_batch(
            structures,
            position_mode=self.position_mode,
            positions_list=kwargs.get("positions_list"),
            n_jobs=self.n_jobs,
        )

        distance_matrix = self.structure_similarity.distance_matrix(descriptors)
        clusters = self.similarity_graph.cluster_from_distance_matrix(distance_matrix)
        representative_records = self.representative_selection.select_representatives(
            clusters,
            records,
        )
        kept_ids = {id(record["atoms"]) for record in representative_records}

        kept_records: list[dict[str, Any]] = []
        removed_records: list[dict[str, Any]] = []
        for record in records:
            if id(record["atoms"]) in kept_ids:
                updated = self._with_curation_decision(
                    record,
                    {"decision": "kept"},
                )
                kept_records.append(updated)
                continue

            updated = self._with_curation_decision(
                record,
                {
                    "decision": "removed",
                    "reason": "non_representative_cluster_member",
                },
            )
            removed_records.append(updated)

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
                        "n_clusters": len(clusters),
                        "cluster_sizes": [len(cluster) for cluster in clusters],
                        "position_mode": self.position_mode,
                        "n_jobs": self.n_jobs,
                    },
                )
            ],
        )
