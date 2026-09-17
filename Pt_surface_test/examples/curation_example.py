from __future__ import annotations

import shutil
from pathlib import Path

from al_dirac.curator.cheap_curate import CheapCurate
from al_dirac.curator.cluster_curate import ClusterCurate
from al_dirac.curator.descriptor_curate import DescriptorCurate
from al_dirac.curator.structure_curation import StructureCurationPipeline
from al_dirac.features.representative_selection import RepresentativeSelection
from al_dirac.features.similarity_graph import SimilarityGraph
from al_dirac.features.soap_descriptor import SOAPDescriptor
from al_dirac.features.structure_similarity import StructureSimilarity
from al_dirac.plotting.plots import (
    plot_cluster_size_distribution,
    plot_curation_removal_reasons,
    plot_curation_stage_counts,
)
from al_dirac.workflow.outputs import write_records_extxyz
from al_dirac.workflow.train_store import load_labeled_records_from_train_path


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def record_species(records: list[dict]) -> list[str]:
    return sorted(
        {
            symbol
            for record in records
            for symbol in record["atoms"].get_chemical_symbols()
        }
    )


def print_stage_summary(result) -> None:
    for report in result.stage_reports:
        print(
            f"{report.stage}: "
            f"input={report.input_count}, "
            f"kept={report.kept_count}, "
            f"removed={report.removed_count}"
        )


def main() -> None:
    input_db_path = Path("real_example_outputs/parser/train.aselmdb")
    output_dir = Path("real_example_outputs/curation")
    plot_dir = output_dir / "plots"

    clean_output_dir(output_dir)

    records = load_labeled_records_from_train_path(
        input_db_path,
        iteration=0,
        common_data={"system": "Pt surface"},
    )
    species = record_species(records)

    soap = SOAPDescriptor(
        species=species,
        r_cut=6.0,
        n_max=6,
        l_max=4,
        sigma=0.5,
        periodic=True,
        sparse=False,
    )
    similarity = StructureSimilarity(
        metric="cosine",
    )

    cheap_curate = CheapCurate(
        position_decimals=2,
        cell_decimals=2,
        energy_decimals=4,
        energy_tolerance=0.0001,
    )
    descriptor_curate = DescriptorCurate(
        soap_descriptor=soap,
        structure_similarity=similarity,
        similarity_threshold=0.9999,
        distance_threshold=None,
        position_mode="all",
        n_jobs=1,
    )
    cluster_curate = ClusterCurate(
        soap_descriptor=soap,
        structure_similarity=similarity,
        similarity_graph=SimilarityGraph(
            eps=0.005,
        ),
        representative_selection=RepresentativeSelection(
            priority=[("energy", "min")],
        ),
        position_mode="all",
        n_jobs=1,
    )

    pipeline = StructureCurationPipeline(
        cheap_curate=cheap_curate,
        descriptor_curate=descriptor_curate,
        cluster_curate=cluster_curate,
        use_cheap=True,
        use_descriptor=False,
        use_cluster=True,
    )

    result = pipeline.curate_records(records)

    write_records_extxyz(records, output_dir / "input_structures.extxyz")
    write_records_extxyz(result.kept_records, output_dir / "kept_structures.extxyz")
    write_records_extxyz(result.removed_records, output_dir / "removed_structures.extxyz")

    plot_curation_stage_counts(result, plot_dir, iteration=0)
    plot_curation_removal_reasons(result, plot_dir, iteration=0)
    plot_cluster_size_distribution(result, plot_dir, iteration=0)

    print("real curation example complete")
    print(f"species: {species}")
    print(f"input:   {len(records)}")
    print(f"kept:    {len(result.kept_records)}")
    print(f"removed: {len(result.removed_records)}")
    print_stage_summary(result)
    print("outputs:")
    print(f"  input extxyz: {output_dir / 'input_structures.extxyz'}")
    print(f"  kept extxyz: {output_dir / 'kept_structures.extxyz'}")
    print(f"  removed extxyz: {output_dir / 'removed_structures.extxyz'}")
    print(f"  plots: {plot_dir}")


if __name__ == "__main__":
    main()
