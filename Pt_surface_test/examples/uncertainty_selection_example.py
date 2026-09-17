from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read

from al_dirac.models.mace_model import MACEModel
from al_dirac.plotting.plots import (
    plot_selection_scores,
    plot_uncertainty_histogram,
    plot_uncertainty_scatter,
)
from al_dirac.selection.ensemble_uncertainty_selector import EnsembleUncertaintySelector
from al_dirac.uncertainty.ensemble import EnsembleUncertainty
from al_dirac.workflow.outputs import write_records_extxyz

SAMPLER_DIR = Path("Pt_surface_test/real_example_outputs/sampler")
CANDIDATE_FILES = [
    SAMPLER_DIR / "rattle_samples.extxyz",
    SAMPLER_DIR / "md_samples.extxyz",
    SAMPLER_DIR / "dimer_samples.extxyz",
]

DEVICE = "cuda"

MODEL_FACTORY_DIR = Path("Pt_surface_test/real_example_outputs/model_factory")
MODEL_PATHS = [
    MODEL_FACTORY_DIR / "pt_surface_lora_ensemble_000.model",
    MODEL_FACTORY_DIR / "pt_surface_lora_ensemble_001.model",
    MODEL_FACTORY_DIR / "pt_surface_lora_ensemble_002.model",
]

OUTPUT_DIR = Path("Pt_surface_test/real_example_outputs/uncertainty_selection")


def load_candidate_records() -> list[dict[str, Any]]:
    records = []
    for source_path in CANDIDATE_FILES:
        if not source_path.exists():
            continue
        structures = read(source_path, index=":")
        if not isinstance(structures, list):
            structures = [structures]
        for index, atoms in enumerate(structures):
            records.append(
                {
                    "atoms": atoms,
                    "structure_id": f"{source_path.stem}_{index}",
                    "iteration": 0,
                }
            )
    return records


def load_ensemble_models() -> list[MACEModel]:
    return [
        MACEModel.load(model_path, device=DEVICE, default_dtype="float32", head="Default")
        for model_path in MODEL_PATHS
    ]


def score_records(
    records: list[dict[str, Any]], uncertainty: EnsembleUncertainty
) -> list[dict[str, Any]]:
    scored = []
    for record in records:
        result = uncertainty.predict(record["atoms"])
        updated = dict(record)
        updated.update(result)
        scored.append(updated)
    return scored


def print_records(title: str, records: list[dict[str, Any]]) -> None:
    print(title)
    for record in records:
        print(
            f"{record['structure_id']}: "
            f"energy_std={record.get('energy_std'):.6f}, "
            f"force_mean_uncertainty={record.get('force_mean_uncertainty'):.6f}, "
            f"force_max_uncertainty={record.get('force_max_uncertainty'):.6f}, "
            f"selection_score={record.get('selection_score')}"
        )


def main() -> None:
    for model_path in MODEL_PATHS:
        if not model_path.exists():
            raise FileNotFoundError("Run examples/model_factory_example.py first.")

    candidate_records = load_candidate_records()
    if not candidate_records:
        raise FileNotFoundError("Run examples/sampler_example.py first.")

    plot_dir = OUTPUT_DIR / "plots"
    pool_path = OUTPUT_DIR / "scored_pool.extxyz"
    selected_path = OUTPUT_DIR / "selected_candidates.extxyz"

    models = load_ensemble_models()
    uncertainty = EnsembleUncertainty(models=models)
    selector = EnsembleUncertaintySelector(
        score_expression="force_max_uncertainty",
        larger_is_better=True,
        min_score=None,
    )
    expression_selector = EnsembleUncertaintySelector(
        score_expression="force_max_uncertainty + 0.5 * energy_std",
        larger_is_better=True,
        min_score=None,
    )

    scored_records = score_records(candidate_records, uncertainty)

    selected_records = selector.select(scored_records, k=5)
    expression_selected_records = expression_selector.select(scored_records, k=5)

    write_records_extxyz(scored_records, pool_path)
    write_records_extxyz(selected_records, selected_path)

    plot_paths = [
        plot_uncertainty_histogram(
            scored_records, plot_dir, iteration=0,
            key="force_max_uncertainty", selected_records=selected_records,
        ),
        plot_uncertainty_histogram(
            scored_records, plot_dir, iteration=0,
            key="force_mean_uncertainty", selected_records=selected_records,
        ),
        plot_uncertainty_scatter(
            scored_records, plot_dir, iteration=0,
            x_key="energy_std", y_key="force_max_uncertainty",
            selected_records=selected_records,
        ),
        plot_selection_scores(
            selected_records, plot_dir, iteration=0, score_key="selection_score",
        ),
    ]

    print(f"candidate pool: {len(candidate_records)} structures")
    print_records("selected by force_max_uncertainty", selected_records)
    print_records("selected by score_expression", expression_selected_records)

    print("outputs:")
    print(f"scored pool extxyz: {pool_path}")
    print(f"selected extxyz: {selected_path}")
    for path in plot_paths:
        print(f"plot: {path}")


if __name__ == "__main__":
    main()
