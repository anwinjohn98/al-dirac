from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ase import Atoms


class BaseDFTLabeler(ABC):
    def __init__(self, labeler_name: str) -> None:
        self.labeler_name = labeler_name

    @abstractmethod
    def prepare_job(
        self,
        atoms: Atoms,
        workdir: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Prepare one DFT labeling job and return its working directory."""
        raise NotImplementedError

    @abstractmethod
    def job_finished(
        self,
        workdir: str | Path,
        **kwargs: Any,
    ) -> bool:
        """Return whether the prepared job has finished."""
        raise NotImplementedError

    @abstractmethod
    def job_succeeded(
        self,
        workdir: str | Path,
        **kwargs: Any,
    ) -> bool:
        """Return whether the finished job produced a valid label."""
        raise NotImplementedError

    @abstractmethod
    def collect_result(
        self,
        workdir: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Collect labeled results from one finished successful job."""
        raise NotImplementedError
