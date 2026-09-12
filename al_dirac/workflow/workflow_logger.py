from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from al_dirac.constants import (
    DEFAULT_EVENTS_FILE,
    DEFAULT_LOG_FILE,
    DEFAULT_METRICS_FILE,
)
from al_dirac.workflow.state import WorkflowState


class WorkflowLogger:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        events_filename: str = DEFAULT_EVENTS_FILE,
        metrics_filename: str = DEFAULT_METRICS_FILE,
        text_filename: str = DEFAULT_LOG_FILE,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.run_dir / events_filename
        self.metrics_path = self.run_dir / metrics_filename
        self.text_path = self.run_dir / text_filename
        self.start_time = time.perf_counter()
        self.iteration_start_time: float | None = None
        self.stage_start_time: float | None = None

    def _timestamp(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _elapsed_seconds(self, start_time: float | None) -> float | None:
        if start_time is None:
            return None
        return round(time.perf_counter() - start_time, 3)

    def _json_default(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a") as handle:
            handle.write(json.dumps(payload, default=self._json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("w") as handle:
            handle.write(
                json.dumps(payload, indent=2, default=self._json_default) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)

    def _append_text(self, message: str) -> None:
        with self.text_path.open("a") as handle:
            handle.write(message + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def compact_state(self, state: WorkflowState) -> dict[str, Any]:
        return {
            "run_id": state.run_id,
            "iteration": state.iteration,
            "status": state.status,
            "train_size": state.train_size,
            "valid_size": state.valid_size,
            "test_size": state.test_size,
            "pool_size": state.pool_size,
            "selected_size": state.selected_size,
            "last_checkpoint": state.last_checkpoint,
            "last_completed_stage": state.last_completed_stage,
            "last_completed_artifact": state.last_completed_artifact,
            "best_metric": state.best_metric,
            "stop_reason": state.stop_reason,
            "metadata": state.metadata,
        }

    def log_event(
        self,
        event: str,
        *,
        state: WorkflowState | None = None,
        **data: Any,
    ) -> None:
        payload = {
            "event": event,
            "timestamp": self._timestamp(),
            "run_elapsed_seconds": self._elapsed_seconds(self.start_time),
            **data,
        }

        if state is not None:
            payload.update(
                {
                    "iteration": state.iteration,
                    "status": state.status,
                }
            )

        self._append_jsonl(self.events_path, payload)

    def log_state(self, state: WorkflowState) -> None:
        self._write_json(self.metrics_path, self.compact_state(state))

    def log_iteration_start(self, state: WorkflowState) -> None:
        self.iteration_start_time = time.perf_counter()
        self.stage_start_time = self.iteration_start_time
        self.log_event("iteration_start", state=state)
        self._append_text(f"iteration {state.iteration} start")
        self.log_state(state)

    def log_iteration_end(self, state: WorkflowState) -> None:
        iteration_elapsed = self._elapsed_seconds(self.iteration_start_time)
        self.log_event(
            "iteration_end",
            state=state,
            train_size=state.train_size,
            pool_size=state.pool_size,
            selected_size=state.selected_size,
            iteration_elapsed_seconds=iteration_elapsed,
        )
        self._append_text(
            f"iteration {state.iteration} end: "
            f"train={state.train_size} "
            f"pool={state.pool_size} "
            f"selected={state.selected_size} "
            f"elapsed_seconds={iteration_elapsed}"
        )
        self.log_state(state)

    def log_stage_completed(
        self,
        state: WorkflowState,
        stage: str,
        *,
        artifact: str | Path | None = None,
        **data: Any,
    ) -> None:
        stage_elapsed = self._elapsed_seconds(self.stage_start_time)
        artifact_value = None if artifact is None else str(artifact)
        state.mark_stage_completed(stage, artifact_value)
        self.log_event(
            "stage_completed",
            state=state,
            stage=stage,
            artifact=artifact_value,
            stage_elapsed_seconds=stage_elapsed,
            **data,
        )

        summary = f"{stage}: completed"
        if artifact_value is not None:
            summary += f" artifact={artifact_value}"
        summary += f" stage_elapsed_seconds={stage_elapsed}"
        if data:
            summary += " " + " ".join(f"{key}={value}" for key, value in data.items())
        self._append_text(summary)
        self.stage_start_time = time.perf_counter()
        self.log_state(state)

    def log_error(
        self,
        state: WorkflowState,
        error: Exception,
        **data: Any,
    ) -> None:
        self.log_event(
            "error",
            state=state,
            error_type=type(error).__name__,
            error_message=str(error),
            **data,
        )
        self._append_text(f"error {type(error).__name__}: {error}")
        self.log_state(state)
