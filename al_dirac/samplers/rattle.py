from __future__ import annotations

from ase import Atoms
from ase.geometry import get_distances
from ase.neighborlist import NeighborList, natural_cutoffs

from al_dirac.samplers.base import BaseSampler


class RattleSampler(BaseSampler):
    def __init__(
        self,
        stdev: float = 0.05,
        n_samples: int = 10,
        seed: int | None = None,
        min_distance_scale: float = 0.7,
        max_attempts_per_sample: int = 20,
    ) -> None:
        super().__init__(sampler_name="rattle")
        self.stdev = stdev
        self.n_samples = n_samples
        self.seed = seed
        self.min_distance_scale = min_distance_scale
        self.max_attempts_per_sample = max_attempts_per_sample

    def _is_valid_structure(self, atoms: Atoms) -> bool:
        base_cutoffs = natural_cutoffs(atoms)
        neighbor_list = NeighborList(
            cutoffs=base_cutoffs,
            self_interaction=False,
            bothways=True,
        )
        neighbor_list.update(atoms)

        for atom_index in range(len(atoms)):
            indices, _ = neighbor_list.get_neighbors(atom_index)
            if len(indices) == 0:
                continue

            distances = get_distances(
                atoms.positions[atom_index],
                atoms.positions[indices],
                cell=atoms.cell,
                pbc=atoms.pbc,
            )[1][0]

            for neighbor_index, distance in zip(indices, distances):
                if neighbor_index <= atom_index:
                    continue

                min_distance = self.min_distance_scale * (
                    base_cutoffs[atom_index] + base_cutoffs[neighbor_index]
                )

                if distance < min_distance:
                    return False
        return True

    def sample(self, atoms: Atoms, **kwargs) -> list[Atoms]:
        sampled_structures: list[Atoms] = []

        sample_index = 0
        while len(sampled_structures) < self.n_samples:
            accepted = False

            for attempt in range(self.max_attempts_per_sample):
                trial = atoms.copy()
                local_seed = None
                if self.seed is not None:
                    local_seed = (
                        self.seed
                        + sample_index * self.max_attempts_per_sample
                        + attempt
                    )
                trial.rattle(stdev=self.stdev, seed=local_seed)

                if self._is_valid_structure(trial):
                    sampled_structures.append(trial)
                    accepted = True
                    break

            if not accepted:
                break

            sample_index += 1
        
        return sampled_structures
