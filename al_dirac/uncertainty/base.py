from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ase import Atoms

class BaseUncertainty(ABC):
    def __init__(self, method_name: str) -> None:
        self.method_name = method_name

    @abstractmethod
    def predict(self, atoms: Atoms) -> dict[str, Any]:
        pass
