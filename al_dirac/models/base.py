from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class BaseModel(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def train(self, train_path: str | Path, valid_path: str | Path | None = None) -> None:
        pass

    @abstractmethod
    def predict(self, atoms: Any) -> dict[str, Any]:
        pass

    @abstractmethod
    def save(self, checkpoint_path: str | Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, checkpoint_path: str | Path) -> "BaseModel":
        pass
