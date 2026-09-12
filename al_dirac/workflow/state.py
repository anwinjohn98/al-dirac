from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from al_dirac.constants import WORKFLOW_STATUS_INITIALIZED


@dataclass
class WorkflowState:
    run_id: str
    iteration: int = 0
    status: str = WORKFLOW_STATUS_INITIALIZED

    model_name: str | None = None
    train_size: int = 0
    valid_size: int = 0
    test_size: int = 0
    pool_size: int = 0
    selected_size: int = 0

    last_checkpoint: str | None = None
    last_completed_stage: str | None = None
    last_completed_artifact: str | None = None
    best_metric: float | None = None
    stop_reason: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowState":
        valid_fields = cls.__dataclass_fields__
        return cls(
            **{
                key: value
                for key, value in data.items()
                if key in valid_fields
            }
        )

    def mark_stage_completed(
        self,
        stage: str,
        artifact: str | None = None,
    ) -> None:
        self.last_completed_stage = stage
        self.last_completed_artifact = artifact
