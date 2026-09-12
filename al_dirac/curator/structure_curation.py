from __future__ import annotations

from typing import Any

from ase import Atoms

from al_dirac.curator.base import (
    BaseStructureCuration,
    CurationResult,
    CurationStageReport,
)
from al_dirac.curator.cheap_curate import CheapCurate
from al_dirac.curator.cluster_curate import ClusterCurate
from al_dirac.curator.descriptor_curate import DescriptorCurate


class StructureCurationPipeline(BaseStructureCuration):
    def __init__(
        self,
        cheap_curate: CheapCurate | None = None,
        descriptor_curate: DescriptorCurate | None = None,
        cluster_curate: ClusterCurate | None = None,
        use_cheap: bool = True,
        use_descriptor: bool = False,
        use_cluster: bool = True,
    ) -> None:
        super().__init__(method_name="structure_curation")
        self.cheap_curate = cheap_curate
        self.descriptor_curate = descriptor_curate
        self.cluster_curate = cluster_curate
        self.use_cheap = use_cheap
        self.use_descriptor = use_descriptor
        self.use_cluster = use_cluster

    def _filter_records(
        self,
        records: list[dict[str, Any]],
        structures: list[Atoms],
    ) -> list[dict[str, Any]]:
        kept_ids = {id(atoms) for atoms in structures}
        return [record for record in records if id(record["atoms"]) in kept_ids]

    def curate(
        self,
        structures: list[Atoms],
        records: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[Atoms]:
        curated_structures = list(structures)
        if records is None:
            records = [{"atoms": atoms, "candidate_index": i} for i, atoms in enumerate(structures)]

        if len(records) != len(structures):
            raise ValueError("records and structures must have the same length.")

        curated_records = list(records)

        if self.use_cheap:
            if self.cheap_curate is None:
                raise ValueError("cheap_curate is enabled but no CheapCurate stage was provided.")
            curated_structures = self.cheap_curate.curate(curated_structures, **kwargs)
            curated_records = self._filter_records(curated_records, curated_structures)

        if self.use_descriptor:
            if self.descriptor_curate is None:
                raise ValueError(
                    "descriptor_curate is enabled but no DescriptorCurate stage was provided."
                )
            curated_structures = self.descriptor_curate.curate(curated_structures, **kwargs)
            curated_records = self._filter_records(curated_records, curated_structures)

        if self.use_cluster:
            if self.cluster_curate is None:
                raise ValueError(
                    "cluster_curate is enabled but no ClusterCurate stage was provided."
                )
            curated_structures = self.cluster_curate.curate(
                curated_structures,
                records=curated_records,
                **kwargs,
            )

        return curated_structures

    def curate_records(
        self,
        records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CurationResult:
        current_records = list(records)
        removed_records: list[dict[str, Any]] = []
        stage_reports: list[CurationStageReport] = []

        if self.use_cheap:
            if self.cheap_curate is None:
                raise ValueError("cheap_curate is enabled but no CheapCurate stage was provided.")
            result = self.cheap_curate.curate_records(current_records, **kwargs)
            current_records = result.kept_records
            removed_records.extend(result.removed_records)
            stage_reports.extend(result.stage_reports)

        if self.use_descriptor:
            if self.descriptor_curate is None:
                raise ValueError(
                    "descriptor_curate is enabled but no DescriptorCurate stage was provided."
                )
            result = self.descriptor_curate.curate_records(current_records, **kwargs)
            current_records = result.kept_records
            removed_records.extend(result.removed_records)
            stage_reports.extend(result.stage_reports)

        if self.use_cluster:
            if self.cluster_curate is None:
                raise ValueError(
                    "cluster_curate is enabled but no ClusterCurate stage was provided."
                )
            result = self.cluster_curate.curate_records(current_records, **kwargs)
            current_records = result.kept_records
            removed_records.extend(result.removed_records)
            stage_reports.extend(result.stage_reports)

        stage_reports.append(
            CurationStageReport(
                stage=self.method_name,
                input_count=len(records),
                kept_count=len(current_records),
                removed_count=len(removed_records),
            )
        )
        return CurationResult(
            kept_records=current_records,
            removed_records=removed_records,
            stage_reports=stage_reports,
        )
