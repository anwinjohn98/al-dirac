from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import read, write

from al_dirac.dft.base import BaseDFTLabeler


class VASPLabeler(BaseDFTLabeler):
    def __init__(
        self,
        calculator_kwargs: dict[str, Any] | None = None,
        write_input_kwargs: dict[str, Any] | None = None,
        output_filename: str = "OUTCAR",
        completion_marker: str = "General timing and accounting informations for this job",
        required_outputs: tuple[str, ...] = ("OUTCAR", "vasprun.xml"),
    ) -> None:
        super().__init__(labeler_name="vasp")
        self.calculator_kwargs = (
            {} if calculator_kwargs is None else dict(calculator_kwargs)
        )
        self.write_input_kwargs = (
            {} if write_input_kwargs is None else dict(write_input_kwargs)
        )
        self.output_filename = output_filename
        self.completion_marker = completion_marker
        self.required_outputs = required_outputs

    def prepare_job(
        self,
        atoms: Atoms,
        workdir: str | Path,
        **kwargs: Any,
    ) -> Path:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        calc_kwargs = dict(self.calculator_kwargs)
        calc_kwargs.update(kwargs.get("calculator_kwargs", {}))
        calc_kwargs["directory"] = str(workdir)

        calculator = Vasp(**calc_kwargs)
        atoms_to_write = atoms.copy()
        atoms_to_write.calc = calculator

        write_kwargs = dict(self.write_input_kwargs)
        write_kwargs.update(kwargs.get("write_input_kwargs", {}))
        calculator.write_input(atoms_to_write, **write_kwargs)

        write(workdir / "structure.extxyz", atoms)

        return workdir

    def _output_file(self, workdir: str | Path) -> Path:
        return Path(workdir) / self.output_filename

    def _required_outputs_present(self, workdir: str | Path) -> bool:
        workdir = Path(workdir)

        for filename in self.required_outputs:
            file_path = workdir / filename
            if not file_path.exists():
                return False
            if file_path.stat().st_size == 0:
                return False

        return True

    def job_finished(
        self,
        workdir: str | Path,
        **kwargs: Any,
    ) -> bool:
        if not self._required_outputs_present(workdir):
            return False

        output_file = self._output_file(workdir)
        text = output_file.read_text(errors="ignore")
        return self.completion_marker in text

    def job_succeeded(
        self,
        workdir: str | Path,
        **kwargs: Any,
    ) -> bool:
        if not self.job_finished(workdir, **kwargs):
            return False

        try:
            atoms = read(self._output_file(workdir))
            atoms.get_potential_energy()
            atoms.get_forces()
            atoms.get_stress()
        except Exception:
            return False

        return True

    def collect_result(
        self,
        workdir: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        workdir = Path(workdir)

        if not self.job_succeeded(workdir, **kwargs):
            raise RuntimeError(f"VASP job in '{workdir}' did not complete successfully.")

        atoms = read(self._output_file(workdir))

        return {
            "atoms": atoms,
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
            "stress": atoms.get_stress(),
            "workdir": str(workdir),
        }
