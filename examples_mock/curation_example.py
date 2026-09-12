from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write

from al_dirac.curator.cheap_curate import CheapCurate
from al_dirac.curator.cluster_curate import ClusterCurate
from al_dirac.curator.descriptor_curate import DescriptorCurate
from al_dirac.curator.structure_curation import StructureCurationPipeline
from al_dirac.features.representative_selection import RepresentativeSelection
from al_dirac.features.similarity_graph import SimilarityGraph
from al_dirac.features.soap_descriptor import SOAPDescriptor
from al_dirac.features.structure_similarity import StructureSimilarity
from al_dirac.plotting.plots import (
    plot_curation_removal_reasons,
    plot_curation_stage_counts,
)


def write_record_structures(
    records: list[dict],
    output_dir: Path,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    write(path, [record["atoms"] for record in records])
    return path


def curation_decisions(record: dict) -> list[dict]:
    workflow = record.get("workflow") or {}
    curation = workflow.get("curation") or {}
    return list(curation.get("decisions") or [])


def make_records() -> list[dict]:
    structures = [
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=False),
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=False),
        Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]], cell=[8, 8, 8], pbc=False),
        Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.14]], cell=[8, 8, 8], pbc=False),
        Atoms(
            "NH3",
            positions=[
                [0.00, 0.00, 0.00],
                [0.94, 0.00, 0.35],
                [-0.47, 0.81, 0.35],
                [-0.47, -0.81, 0.35],
            ],
            cell=[9, 9, 9],
            pbc=False,
        ),
        Atoms(
            "NH3",
            positions=[
                [0.00, 0.00, 0.00],
                [0.94, 0.00, 0.35],
                [-0.47, 0.81, 0.35],
                [-0.47, -0.81, 0.35],
            ],
            cell=[9, 9, 9],
            pbc=False,
        ),
        Atoms("Cu2", positions=[[0, 0, 0], [2.45, 0, 0]], cell=[8, 8, 8], pbc=True),
        Atoms("Cu2", positions=[[0, 0, 0], [2.55, 0, 0]], cell=[8, 8, 8], pbc=True),
        Atoms("Cu2", positions=[[0, 0, 0], [2.65, 0, 0]], cell=[8, 8, 8], pbc=True),
        Atoms(
            "PtCO",
            positions=[
                [0, 0, 0],
                [0, 0, 1.85],
                [0, 0, 3.00],
            ],
            cell=[10, 10, 12],
            pbc=True,
        ),
    ]
    energies = [
        -1.0,
        -1.0,
        -2.0,
        -2.1,
        -3.0,
        -3.1,
        -0.5,
        -0.8,
        -0.6,
        -4.0,
    ]
    records = []
    for index, (atoms, energy) in enumerate(zip(structures, energies)):
        atoms.info["energy"] = energy
        records.append(
            {
                "atoms": atoms,
                "candidate_index": index,
                "structure_id": f"structure_{index}",
                "priority": index,
                "energy": energy,
            }
        )
    return records

def main() -> None:
    output_dir = Path("example_outputs/curation/plots")
    structure_dir = Path("example_outputs/curation/structures")

    soap = SOAPDescriptor(
        species=["H", "C", "O", "N", "Cu", "Pt"],
        r_cut=3.0,
        n_max=4,
        l_max=3,
        periodic=True,
    )
    similarity = StructureSimilarity()
    graph = SimilarityGraph(eps=0.5)
    representative_selection = RepresentativeSelection(priority=[("energy", "min")])

    cheap_curate = CheapCurate(
        position_decimals=6,
        cell_decimals=6,
        energy_decimals=None,
        energy_tolerance=None,
    )
    descriptor_curate = DescriptorCurate(
        soap_descriptor=soap,
        structure_similarity=similarity,
        similarity_threshold=None,
        distance_threshold=1e-4,
        position_mode="all",
        n_jobs=1,
    )
    cluster_curate = ClusterCurate(
        soap_descriptor=soap,
        structure_similarity=similarity,
        similarity_graph=graph,
        representative_selection=representative_selection,
        position_mode="all",
        n_jobs=1,
    )

    pipeline = StructureCurationPipeline(
        cheap_curate=cheap_curate,
        descriptor_curate=descriptor_curate,
        cluster_curate=cluster_curate,
        use_cheap=True,
        use_descriptor=True,
        use_cluster=True,
    )

    records = make_records()
    result = pipeline.curate_records(records)
    structure_paths = [
        write_record_structures(records, structure_dir, "input_structures.xyz"),
        write_record_structures(result.kept_records, structure_dir, "kept_structures.xyz"),
        write_record_structures(
            result.removed_records,
            structure_dir,
            "removed_structures.xyz",
        ),
    ]

    print(f"input:   {len(records)}")
    print(f"kept:    {len(result.kept_records)}")
    print(f"removed: {len(result.removed_records)}")
    for report in result.stage_reports:
        print(
            f"{report.stage}: "
            f"input={report.input_count}, "
            f"kept={report.kept_count}, "
            f"removed={report.removed_count}"
        )

    print("removed records:")
    for record in result.removed_records:
        removed_decisions = [
            decision
            for decision in curation_decisions(record)
            if decision.get("decision") == "removed"
        ]
        for decision in removed_decisions:
            duplicate_of = decision.get("duplicate_of")
            duplicate_text = (
                "" if duplicate_of is None else f", duplicate_of={duplicate_of}"
            )
            print(
                f"{record['structure_id']}: "
                f"stage={decision.get('stage')}, "
                f"reason={decision.get('reason')}"
                f"{duplicate_text}"
            )

    paths = [
        plot_curation_stage_counts(result, output_dir, iteration=0),
        plot_curation_removal_reasons(result, output_dir, iteration=0),
    ]
    print("plots:")
    for path in paths:
        print(path)
    print("structures:")
    for path in structure_paths:
        print(path)


if __name__ == "__main__":
    main()
