from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read

from al_dirac.workflow.stopping import UncertaintyStoppingCriteria

SCORED_POOL_FILE = Path("Pt_surface_test/real_example_outputs/uncertainty_selection/scored_pool.extxyz")


def load_scored_records() -> list[dict[str, Any]]:
    structures = read(SCORED_POOL_FILE, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    return [dict(atoms.info) for atoms in structures]


def run_case(name: str, records: list[dict[str, Any]], criteria: UncertaintyStoppingCriteria) -> None:
    decision = criteria.check(records)
    print(f"{name}:")
    print(f"  key/expression: {criteria.expression or criteria.key}")
    print(f"  statistic: {criteria.statistic}")
    print(f"  threshold: {criteria.threshold}")
    print(f"  should_stop: {decision.should_stop}")
    print(f"  value: {decision.value}")
    print(f"  reason: {decision.reason}")
    print()


def main() -> None:
    if not SCORED_POOL_FILE.exists():
        raise FileNotFoundError("Run examples/uncertainty_selection_example.py first.")

    records = load_scored_records()
    print(f"scored pool: {len(records)} structures")
    print()

    run_case(
        "high threshold (expect stop)",
        records,
        UncertaintyStoppingCriteria(key="force_max_uncertainty", threshold=0.01, statistic="max"),
    )
    run_case(
        "low threshold (expect continue)",
        records,
        UncertaintyStoppingCriteria(key="force_max_uncertainty", threshold=0.005, statistic="max"),
    )
    run_case(
        "mean statistic instead of max",
        records,
        UncertaintyStoppingCriteria(key="force_max_uncertainty", threshold=0.007, statistic="mean"),
    )
    run_case(
        "composite expression, scaled to match real magnitudes",
        records,
        UncertaintyStoppingCriteria(
            expression="force_max_uncertainty + 0.001 * energy_std",
            threshold=0.01,
            statistic="max",
        ),
    )


if __name__ == "__main__":
    main()
