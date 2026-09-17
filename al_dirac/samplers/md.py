from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from al_dirac.samplers.base import BaseSampler

@dataclass(frozen=True)
class DynamicsSpec:
    module: str
    class_name: str
    
_DYNAMICS_SPECS: dict[str, DynamicsSpec] = {
    "VelocityVerlet": DynamicsSpec("ase.md.verlet", "VelocityVerlet"),
    "Langevin": DynamicsSpec("ase.md.langevin","Langevin"),
    "NoseHooverChainNVT": DynamicsSpec("ase.md.nose_hoover_chain","NoseHooverChainNVT"),
    "Bussi": DynamicsSpec("ase.md.bussi", "Bussi"),
    "Andersen": DynamicsSpec("ase.md.andersen", "Andersen"),
    "NVTBerendsen": DynamicsSpec("ase.md.nvtberendsen", "NVTBerendsen"),
    "NPTBerendsen": DynamicsSpec("ase.md.nptberendsen", "NPTBerendsen"),
    "IsotropicMTKNPT": DynamicsSpec("ase.md.nose_hoover_chain", "IsotropicMTKNPT"),
    "MTKNPT": DynamicsSpec("ase.md.nose_hoover_chain", "MTKNPT"),
    "MaskedMTKNPT": DynamicsSpec("ase.md.nose_hoover_chain", "MaskedMTKNPT"),
    "LangevinBAOAB": DynamicsSpec("ase.md.langevinbaoab", "LangevinBAOAB"),
    "MelchionnaNPT": DynamicsSpec("ase.md.melchionna", "MelchionnaNPT"),
    "ContourExploration": DynamicsSpec("ase.md.contour_exploration", "ContourExploration"),
}

class MDSampler(BaseSampler):
    def __init__(
        self,
        dynamics_name: str = "Langevin",
        timestep_fs: float = 1.0,
        steps: int = 1000,
        sample_interval: int = 10,
        initialize_velocities: bool = True,
        velocity_temperature_K: float | None = 300.0,
        zero_center_of_mass_momentum: bool = False,
        dynamics_kwargs: dict[str, Any] | None = None,
        log_progress: bool = False,
        log_interval: int | None = None,
        calculator: Any = None,
    ) -> None:
        super().__init__(sampler_name="md")

        if dynamics_name not in _DYNAMICS_SPECS:
            raise ValueError(f"Unsupported dynamics: {dynamics_name}")
        if timestep_fs <= 0:
            raise ValueError("timestep_fs must be positive")
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive.")
        if log_interval is not None and log_interval <= 0:
            raise ValueError("log_interval must be positive or None.")

        self.dynamics_name = dynamics_name
        self.timestep_fs = timestep_fs
        self.steps = steps
        self.sample_interval = sample_interval
        self.initialize_velocities = initialize_velocities
        self.velocity_temperature_K = velocity_temperature_K
        self.zero_center_of_mass_momentum = zero_center_of_mass_momentum
        self.dynamics_kwargs = {} if dynamics_kwargs is None else dict(dynamics_kwargs)
        self.log_progress = log_progress
        self.log_interval = log_interval
        self.calculator = calculator

    def _get_dynamics_class(self):
        spec = _DYNAMICS_SPECS[self.dynamics_name]
        module = importlib.import_module(spec.module)
        return getattr(module, spec.class_name)

    def _validate_dynamics_kwargs(self, dynamics_cls) -> None:
        signature = inspect.signature(dynamics_cls.__init__)
        parameters = signature.parameters

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            return

        allowed = set(parameters)
        allowed.discard("self")
        allowed.discard("atoms")
        allowed.discard("timestep")

        unknown = [key for key in self.dynamics_kwargs if key not in allowed]
        if unknown:
            raise ValueError(
                f"Unsupported dynamics_kwargs for {self.dynamics_name}: {unknown}."
                f"Allowed keys are: {sorted(allowed)}"
            )
    def _initialize_velocities(self, atoms: Atoms) -> None:
        if not self.initialize_velocities:
            return

        if self.velocity_temperature_K is None:
            raise ValueError(
                "velocity_temperature_K must be provided when initialize_velocities=True."
            )

        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=self.velocity_temperature_K,
        )

        if self.zero_center_of_mass_momentum:
            momenta = atoms.get_momenta()
            atoms.set_momenta(momenta - momenta.mean(axis=0))

    def _build_dynamics(self, atoms: Atoms):
        dynamics_cls = self._get_dynamics_class()
        self._validate_dynamics_kwargs(dynamics_cls)

        return dynamics_cls(
            atoms,
            timestep=self.timestep_fs * units.fs,
            **self.dynamics_kwargs,
        )

    def sample(self, atoms: Atoms, **kwargs: Any) -> list[Atoms]:
        if self.calculator is None:
            raise ValueError("MDSampler requires a calculator -- pass calculator=... .")

        trial = atoms.copy()
        trial.calc = self.calculator

        self._initialize_velocities(trial)
        dyn = self._build_dynamics(trial)

        samples: list[Atoms] = [trial.copy()]

        def _store_frame() -> None:
            samples.append(trial.copy())

        if self.log_progress:
            log_interval = self.log_interval or self.sample_interval

            def _log_progress() -> None:
                print(f"MD step {dyn.nsteps}/{self.steps}")

            print(f"MD step 0/{self.steps}")
            dyn.attach(_log_progress, interval=log_interval)

        dyn.attach(_store_frame, interval=self.sample_interval)
        dyn.run(self.steps)

        return samples
