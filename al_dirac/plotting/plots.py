from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = "/tmp/al_dirac_matplotlib"
os.makedirs(_MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)
os.environ.setdefault("XDG_CACHE_HOME", _MPLCONFIGDIR)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def iteration_plot_dir(root_dir: str | Path, iteration: int) -> Path:
    plot_dir = Path(root_dir) / f"iteration_{iteration:04d}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _values(records: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [record.get(key) for record in records if record.get(key) is not None]
    return np.asarray(values, dtype=float)


def _paired_values(
    records: list[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (record.get(x_key), record.get(y_key))
        for record in records
        if record.get(x_key) is not None and record.get(y_key) is not None
    ]
    if not pairs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x, y = zip(*pairs)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _record_curation_decisions(record: dict[str, Any]) -> list[dict[str, Any]]:
    workflow = record.get("workflow") or {}
    curation = workflow.get("curation") or {}
    return list(curation.get("decisions") or [])


def plot_uncertainty_histogram(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    iteration: int,
    key: str = "force_max_uncertainty",
    selected_records: list[dict[str, Any]] | None = None,
) -> Path:
    output_dir = iteration_plot_dir(output_dir, iteration)
    values = _values(records, key)
    bins = np.histogram_bin_edges(values, bins=40)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, alpha=0.7, label="pool")

    if selected_records:
        ax.hist(_values(selected_records, key), bins=bins, alpha=0.7, label="selected")

    ax.set_xlabel(key)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()

    path = output_dir / f"{key}_histogram.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_uncertainty_scatter(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    iteration: int,
    x_key: str = "energy_std",
    y_key: str = "force_max_uncertainty",
    selected_records: list[dict[str, Any]] | None = None,
) -> Path:
    output_dir = iteration_plot_dir(output_dir, iteration)
    x, y = _paired_values(records, x_key, y_key)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=16, alpha=0.5, label="pool")

    if selected_records:
        sx, sy = _paired_values(selected_records, x_key, y_key)
        ax.scatter(sx, sy, s=30, alpha=0.9, label="selected")

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.legend()
    fig.tight_layout()

    path = output_dir / f"{x_key}_vs_{y_key}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_selection_scores(
    selected_records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    iteration: int,
    score_key: str = "selection_score",
) -> Path:
    output_dir = iteration_plot_dir(output_dir, iteration)
    scores = np.sort(_values(selected_records, score_key))[::-1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(scores)), scores, marker="o")
    ax.set_xlabel("selected rank")
    ax.set_ylabel(score_key)
    fig.tight_layout()

    path = output_dir / f"{score_key}_selected.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_seed_selection_values(
    seed_records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    iteration: int,
    value_key: str = "energy",
) -> Path:
    output_dir = iteration_plot_dir(output_dir, iteration)
    values = _values(seed_records, value_key)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(values)), values, marker="o")
    ax.set_xlabel("seed rank")
    ax.set_ylabel(value_key)
    fig.tight_layout()

    path = output_dir / f"seed_{value_key}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_active_learning_progress(
    events_path: str | Path,
    output_dir: str | Path,
) -> Path:
    events = [
        json.loads(line)
        for line in Path(events_path).read_text().splitlines()
        if line.strip()
    ]

    selecting = [
        event
        for event in events
        if event.get("event") == "stage_completed"
        and event.get("stage") == "selecting"
    ]

    iterations = [event.get("iteration", 0) for event in selecting]
    selected_counts = [event.get("record_count", np.nan) for event in selecting]
    pool_counts = [event.get("pool_size", np.nan) for event in selecting]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iterations, selected_counts, marker="o", label="selected")
    if not all(np.isnan(pool_counts)):
        ax.plot(iterations, pool_counts, marker="o", label="pool")

    ax.set_xlabel("iteration")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "active_learning_progress.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_parity(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    iteration: int,
    true_key: str,
    pred_key: str,
    name: str,
) -> Path:
    output_dir = iteration_plot_dir(output_dir, iteration)
    true, pred = _paired_values(records, true_key, pred_key)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true, pred, s=16, alpha=0.6)
    if len(true):
        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)

    ax.set_xlabel(true_key)
    ax.set_ylabel(pred_key)
    fig.tight_layout()

    path = output_dir / f"{name}_parity.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_curation_stage_counts(
    curation_result: Any,
    output_dir: str | Path,
    *,
    iteration: int,
) -> Path:
    output_dir = iteration_plot_dir(output_dir, iteration)
    reports = curation_result.stage_reports
    stages = [report.stage for report in reports]
    input_counts = [report.input_count for report in reports]
    kept_counts = [report.kept_count for report in reports]
    removed_counts = [report.removed_count for report in reports]

    x = np.arange(len(stages))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(7, len(stages) * 1.4), 4))
    ax.bar(x - width, input_counts, width, label="input")
    ax.bar(x, kept_counts, width, label="kept")
    ax.bar(x + width, removed_counts, width, label="removed")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=30, ha="right")
    ax.set_ylabel("records")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "curation_stage_counts.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_curation_removal_reasons(
    curation_result: Any,
    output_dir: str | Path,
    *,
    iteration: int,
) -> Path:
    output_dir = iteration_plot_dir(output_dir, iteration)
    reasons: Counter[str] = Counter()

    for record in curation_result.removed_records:
        for decision in _record_curation_decisions(record):
            if decision.get("decision") == "removed":
                reasons[str(decision.get("reason", "unknown"))] += 1

    labels = list(reasons) or ["none"]
    counts = [reasons[label] for label in labels] if reasons else [0]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 4))
    ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("removed records")
    fig.tight_layout()

    path = output_dir / "curation_removal_reasons.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_cluster_size_distribution(
    clusters: dict[Any, list[Any]] | list[list[Any]],
    output_dir: str | Path,
    *,
    iteration: int,
) -> Path:
    output_dir = iteration_plot_dir(output_dir, iteration)
    if hasattr(clusters, "stage_reports"):
        sizes = []
        for report in clusters.stage_reports:
            if report.stage == "cluster_curate":
                sizes = list((report.details or {}).get("cluster_sizes", []))
                break
    else:
        cluster_values = clusters.values() if isinstance(clusters, dict) else clusters
        sizes = [len(cluster) for cluster in cluster_values]

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(1, max(sizes, default=0) + 2) - 0.5
    ax.hist(sizes, bins=bins)
    ax.set_xlabel("cluster size")
    ax.set_ylabel("clusters")
    fig.tight_layout()

    path = output_dir / "cluster_size_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_iteration_summary(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    iteration: int,
    selected_records: list[dict[str, Any]] | None = None,
    curation_result: Any | None = None,
) -> list[Path]:
    paths = [
        plot_uncertainty_histogram(
            records,
            output_dir,
            iteration=iteration,
            key="force_max_uncertainty",
            selected_records=selected_records,
        ),
        plot_uncertainty_histogram(
            records,
            output_dir,
            iteration=iteration,
            key="force_mean_uncertainty",
            selected_records=selected_records,
        ),
        plot_uncertainty_scatter(
            records,
            output_dir,
            iteration=iteration,
            x_key="energy_std",
            y_key="force_max_uncertainty",
            selected_records=selected_records,
        ),
    ]

    if selected_records:
        paths.append(
            plot_selection_scores(
                selected_records,
                output_dir,
                iteration=iteration,
            )
        )

    if curation_result is not None:
        paths.extend(
            [
                plot_curation_stage_counts(
                    curation_result,
                    output_dir,
                    iteration=iteration,
                ),
                plot_curation_removal_reasons(
                    curation_result,
                    output_dir,
                    iteration=iteration,
                ),
            ]
        )

    return paths
