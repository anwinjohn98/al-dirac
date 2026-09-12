from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ase import Atoms

class BaseSampler(ABC):
    def __init__(self, sampler_name: str) -> None:
        self.sampler_name = sampler_name

    @abstractmethod
    def sample(self, atoms: Atoms, **kwargs: Any) -> list[Atoms]:
        pass
