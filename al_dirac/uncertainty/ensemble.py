from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from al_dirac.models.base import BaseModel
from al_dirac.uncertainty.base import BaseUncertainty


class EnsembleUncertainty(BaseUncertainty):
    def __init__(
        self,
        models: list[BaseModel],
    ) -> None:
        super().__init__(method_name="ensemble")
        if not models:
            raise ValueError("models must not be empty.")

        self.models = models

    def predict(self, atoms: Atoms) -> dict[str, Any]:
        predictions = [model.predict(atoms) for model in self.models]
        n_models = len(predictions)
        n_atoms = len(atoms)

        energy_values = [pred["energy"] for pred in predictions]
        force_values = [pred["forces"] for pred in predictions]
        stress_values = [pred["stress"] for pred in predictions]

        result: dict[str, Any] = {
            "n_models": n_models,
            "n_atoms": n_atoms,
            "energy_predictions": energy_values,
            "force_predictions": force_values,
            "stress_predictions": stress_values,
            "energy_mean": None,
            "energy_std": None,
            "energy_rho": None,
            "forces_mean": None,
            "forces_std": None,
            "force_atomwise_uncertainty": None,
            "force_mean_uncertainty": None,
            "force_max_uncertainty": None,
            "force_magnitude_per_atom": None,
            "force_rho": None,
            "stress_mean": None,
            "stress_std": None,
        }

        if all(value is not None for value in energy_values):
            energy_array = np.asarray(energy_values, dtype=float)
            energy_mean = float(np.mean(energy_array))
            energy_std = float(np.std(energy_array))

            result["energy_mean"] = energy_mean
            result["energy_std"] = energy_std
            result["energy_rho"] = float(
                np.sqrt(2.0 / (n_models * max(n_atoms, 1))) * energy_std
            )

        if all(value is not None for value in force_values):
            force_array = np.asarray(force_values, dtype=float)
            forces_mean = np.mean(force_array, axis=0)
            forces_std = np.std(force_array, axis=0)

            force_atomwise_uncertainty = np.linalg.norm(forces_std, axis=-1)
            force_mean_uncertainty = float(np.mean(force_atomwise_uncertainty))
            force_max_uncertainty = float(np.max(force_atomwise_uncertainty))

            result["forces_mean"] = forces_mean
            result["forces_std"] = forces_std
            result["force_atomwise_uncertainty"] = force_atomwise_uncertainty
            result["force_mean_uncertainty"] = force_mean_uncertainty
            result["force_max_uncertainty"] = force_max_uncertainty
            result["force_magnitude_per_atom"] = np.linalg.norm(forces_mean, axis=-1)
            result["force_rho"] = float(
                np.sqrt(2.0 / (n_models * max(n_atoms, 1)))
                * force_mean_uncertainty
            )

        if all(value is not None for value in stress_values):
            stress_array = np.asarray(stress_values, dtype=float)
            result["stress_mean"] = np.mean(stress_array, axis=0)
            result["stress_std"] = np.std(stress_array, axis=0)

        return result
