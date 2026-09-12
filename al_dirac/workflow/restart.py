from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from al_dirac.constants import DEFAULT_EVENTS_FILE, DEFAULT_METRICS_FILE
from al_dirac.dft.batch import BatchDFTRunner
from al_dirac.workflow.state import WorkflowState


DEFAULT_STAGE_ORDER = (
    "parsing",
    "training",
    "seed_selecting",
    "sampling",
    "curating",
    "scoring",
    "selecting",
    "labeling",
)
DEFAULT_ARTIFACTS_DIR = "artifacts"


@dataclass(frozen=True)
class RestartPlan:
    state: WorkflowState
    last_completed_stage: str | None
    last_completed_artifact: str | None
    next_stage: str | None
    last_event: dict[str, Any] | None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def stage_artifact_path(
    artifact_dir: str | Path,
    iteration: int,
    stage: str,
) -> Path:
    return Path(artifact_dir) / f"iteration_{iteration:04d}_{stage}.pkl"


def save_artifact(path: str | Path, value: Any) -> Path:
    path = Path(path)
    _atomic_write_bytes(path, pickle.dumps(value))
    return path


def load_artifact(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_state(path: str | Path) -> WorkflowState:
    data = json.loads(Path(path).read_text())
    return WorkflowState.from_dict(data)


def save_state(path: str | Path, state: WorkflowState) -> None:
    path = Path(path)
    _atomic_write_text(path, json.dumps(state.to_dict(), indent=2) + "\n")


def read_events(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def last_completed_event(
    events: list[dict[str, Any]],
    iteration: int | None = None,
) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event") != "stage_completed":
            continue
        if iteration is not None and event.get("iteration") != iteration:
            continue
        return event
    return None


def last_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return events[-1] if events else None


def next_stage(
    last_stage: str | None,
    stage_order: tuple[str, ...] = DEFAULT_STAGE_ORDER,
) -> str | None:
    if last_stage is None:
        return stage_order[0]
    if last_stage not in stage_order:
        return None

    index = stage_order.index(last_stage)
    if index + 1 >= len(stage_order):
        return None
    return stage_order[index + 1]


def build_restart_plan(
    run_dir: str | Path,
    *,
    state_filename: str = DEFAULT_METRICS_FILE,
    events_filename: str = DEFAULT_EVENTS_FILE,
    stage_order: tuple[str, ...] = DEFAULT_STAGE_ORDER,
) -> RestartPlan:
    run_dir = Path(run_dir)
    state = load_state(run_dir / state_filename)
    events = read_events(run_dir / events_filename)
    latest_event = last_event(events)
    if latest_event is not None and latest_event.get("event") == "iteration_end":
        return RestartPlan(
            state=state,
            last_completed_stage=None,
            last_completed_artifact=None,
            next_stage=stage_order[0],
            last_event=latest_event,
        )

    event = last_completed_event(events, iteration=state.iteration)

    last_stage = state.last_completed_stage
    last_artifact = state.last_completed_artifact

    if event is not None:
        last_stage = event.get("stage")
        last_artifact = event.get("artifact")

    return RestartPlan(
        state=state,
        last_completed_stage=last_stage,
        last_completed_artifact=last_artifact,
        next_stage=next_stage(last_stage, stage_order),
        last_event=event,
    )


def collect_completed_dft_from_batches(
    prepared_batches: list[dict[str, Any]],
    dft_runner: BatchDFTRunner,
) -> list[dict[str, Any]]:
    collectable_batches: list[dict[str, Any]] = []

    for batch in prepared_batches:
        jobs = []
        for job in batch.get("jobs", []):
            job_dir = Path(job["job_dir"])
            if dft_runner.labeler.job_succeeded(job_dir):
                jobs.append(job)

        if jobs:
            batch_copy = dict(batch)
            batch_copy["jobs"] = jobs
            collectable_batches.append(batch_copy)

    return dft_runner.collect_results(collectable_batches)


def incomplete_dft_batches(
    prepared_batches: list[dict[str, Any]],
    dft_runner: BatchDFTRunner,
) -> list[dict[str, Any]]:
    incomplete: list[dict[str, Any]] = []

    for batch in prepared_batches:
        jobs = batch.get("jobs", [])
        if not jobs:
            incomplete.append(batch)
            continue

        if any(
            not dft_runner.labeler.job_succeeded(Path(job["job_dir"]))
            for job in jobs
        ):
            incomplete.append(batch)

    return incomplete
