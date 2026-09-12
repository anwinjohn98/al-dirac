from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

class BaseSelector(ABC):
    def __init__(self, selector_name: str) -> None:
        self.selector_name = selector_name

    @abstractmethod
    def select(
        self,
        records: list[dict[str, Any]],
        k: int | None,
    ) -> list[dict[str, Any]]:
        pass
