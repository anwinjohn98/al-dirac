from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP


class SOAPDescriptor:
    def __init__(
        self,
        species: list[str],
        r_cut: float = 5.0,
        n_max: int = 8,
        l_max: int = 6,
        sigma: float = 0.5,
        rbf: str = "gto",
        periodic: bool = False,
        crossover: bool = True,
        sparse: bool = False,
    ) -> None:
        self.species = species
        accepted_kwargs = inspect.signature(SOAP).parameters
        soap_kwargs: dict[str, Any] = {"species": species}

        if "r_cut" in accepted_kwargs:
            soap_kwargs["r_cut"] = r_cut
        elif "rcut" in accepted_kwargs:
            soap_kwargs["rcut"] = r_cut

        if "n_max" in accepted_kwargs:
            soap_kwargs["n_max"] = n_max
        elif "nmax" in accepted_kwargs:
            soap_kwargs["nmax"] = n_max

        if "l_max" in accepted_kwargs:
            soap_kwargs["l_max"] = l_max
        elif "lmax" in accepted_kwargs:
            soap_kwargs["lmax"] = l_max

        optional_kwargs = {
            "sigma": sigma,
            "rbf": rbf,
            "periodic": periodic,
            "crossover": crossover,
            "sparse": sparse,
            "average": "off",
        }
        soap_kwargs.update(
            {
                key: value
                for key, value in optional_kwargs.items()
                if key in accepted_kwargs
            }
        )
        self.soap = SOAP(**soap_kwargs)

    def _select_positions(
        self,
        atoms: Atoms,
        position_mode: str = "all",
        positions: list[int] | list[list[float]] | None = None,
    ) -> list[int] | list[list[float]] | None:
        if position_mode == "all":
            return None

        if position_mode == "indices":
            return positions

        raise ValueError(f"Unsupported position_mode: {position_mode}")

    def create_local(
        self,
        atoms: Atoms,
        position_mode: str = "all",
        positions: list[int] | list[list[float]] | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        selected_positions = self._select_positions(
            atoms,
            position_mode=position_mode,
            positions=positions,
        )

        create_kwargs: dict[str, Any] = {}
        accepted_kwargs = inspect.signature(self.soap.create).parameters
        if "positions" in accepted_kwargs:
            create_kwargs["positions"] = selected_positions
        elif "centers" in accepted_kwargs:
            create_kwargs["centers"] = selected_positions
        if "n_jobs" in accepted_kwargs:
            create_kwargs["n_jobs"] = n_jobs
        if "verbose" in accepted_kwargs:
            create_kwargs["verbose"] = verbose

        descriptors = self.soap.create(atoms, **create_kwargs)
        return np.asarray(descriptors, dtype=float)

    def create_local_batch(
        self,
        structures: list[Atoms],
        position_mode: str = "all",
        positions_list: list[list[int] | list[list[float]] | None] | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        if positions_list is None:
            positions_list = [None] * len(structures)

        return [
            self.create_local(
                atoms,
                position_mode=position_mode,
                positions=positions,
                n_jobs=n_jobs,
                verbose=verbose,
                **kwargs,
            )
            for atoms, positions in zip(structures, positions_list)
        ]
