from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

from al_dirac.samplers.base import BaseSampler


class DimerSampler(BaseSampler):
    def __init__(
        self,
        control_kwargs: dict[str, Any] | None = None,
        displace_kwargs: dict[str, Any] | None = None,
        translate_kwargs: dict[str, Any] | None = None,
        run_kwargs: dict[str, Any] | None = None,
        eigenmodes: Any = None,
        random_seed: int | None = None,
        comm: Any = None,
        collect_trajectory: bool = False,
        sample_interval: int = 1,
        calculator: Any = None,
    ) -> None:
        super().__init__(sampler_name="dimer")

        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive.")

        self.control_kwargs = {} if control_kwargs is None else dict(control_kwargs)
        self.displace_kwargs = {} if displace_kwargs is None else dict(displace_kwargs)
        self.translate_kwargs = {} if translate_kwargs is None else dict(translate_kwargs)
        self.run_kwargs = {} if run_kwargs is None else dict(run_kwargs)
        self.eigenmodes = eigenmodes
        self.random_seed = random_seed
        self.comm = comm
        self.collect_trajectory = collect_trajectory
        self.sample_interval = sample_interval
        self.calculator = calculator

    def _build_default_displacement(self, atoms: Atoms) -> np.ndarray:
        displacement = np.zeros((len(atoms), 3), dtype=float)
        displacement[-1, 2] = 0.1
        return displacement

    def _prepare_displace_kwargs(
        self,
        atoms: Atoms,
        control_kwargs: dict[str, Any],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        displace_kwargs = dict(self.displace_kwargs)
        displace_kwargs.update(overrides)

        method = displace_kwargs.get("method", control_kwargs.get("displacement_method"))
        displacement_vector = displace_kwargs.get("displacement_vector")

        if method == "vector" and displacement_vector is None:
            displace_kwargs["displacement_vector"] = self._build_default_displacement(
                atoms
            )

        return displace_kwargs

    def sample(self, atoms: Atoms, **kwargs: Any) -> list[Atoms]:
        if self.calculator is None:
            raise ValueError("DimerSampler requires a calculator -- pass calculator=... .")

        trial = atoms.copy()
        trial.calc = self.calculator

        control_kwargs = dict(self.control_kwargs)
        control_kwargs.update(kwargs.get("control_kwargs", {}))

        translate_kwargs = dict(self.translate_kwargs)
        translate_kwargs.update(kwargs.get("translate_kwargs", {}))

        run_kwargs = dict(self.run_kwargs)
        run_kwargs.update(kwargs.get("run_kwargs", {}))

        displace_kwargs = self._prepare_displace_kwargs(
            trial,
            control_kwargs,
            kwargs.get("displace_kwargs", {}),
        )

        eigenmodes = kwargs.get("eigenmodes", self.eigenmodes)
        random_seed = kwargs.get("random_seed", self.random_seed)
        comm = kwargs.get("comm", self.comm)
        collect_trajectory = kwargs.get("collect_trajectory", self.collect_trajectory)
        sample_interval = kwargs.get("sample_interval", self.sample_interval)

        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive.")

        samples: list[Atoms] = []

        with DimerControl(**control_kwargs) as d_control:
            minmode_kwargs: dict[str, Any] = {"control": d_control}
            if eigenmodes is not None:
                minmode_kwargs["eigenmodes"] = eigenmodes
            if random_seed is not None:
                minmode_kwargs["random_seed"] = random_seed
            if comm is not None:
                minmode_kwargs["comm"] = comm

            d_atoms = MinModeAtoms(trial, **minmode_kwargs)
            d_atoms.displace(**displace_kwargs)

            if collect_trajectory:
                samples.append(d_atoms.atoms.copy())

                def _store_frame() -> None:
                    samples.append(d_atoms.atoms.copy())

            with MinModeTranslate(d_atoms, **translate_kwargs) as optimizer:
                if collect_trajectory:
                    optimizer.attach(_store_frame, interval=sample_interval)
                optimizer.run(**run_kwargs)

            if not collect_trajectory:
                samples.append(d_atoms.atoms.copy())

        return samples
