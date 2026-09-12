from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write

from al_dirac.samplers.dimer import DimerSampler
from al_dirac.samplers.md import MDSampler
from al_dirac.samplers.rattle import RattleSampler


def make_seed_structure() -> Atoms:
    atoms = Atoms(
        "Cu4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, 2.5],
        ],
        cell=[8, 8, 8],
        pbc=False,
    )
    atoms.calc = EMT()
    return atoms


def attach_calculator(samples: list[Atoms]) -> list[Atoms]:
    prepared = []
    for atoms in samples:
        atoms.calc = EMT()
        prepared.append(atoms)
    return prepared


def write_samples(samples: list[Atoms], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, samples, format="extxyz")
    return path


def print_samples(name: str, samples: list[Atoms]) -> None:
    print(f"{name}: {len(samples)} samples")
    for index, atoms in enumerate(samples):
        energy = atoms.get_potential_energy()
        print(f"  frame={index}, atoms={len(atoms)}, energy={energy:.6f}")


def main() -> None:
    output_dir = Path("example_outputs/sampler")
    seed = make_seed_structure()

    seed_path = write_samples([seed], output_dir / "seed.extxyz")

    rattle_sampler = RattleSampler(
        stdev=0.05,
        n_samples=5,
        seed=7,
        min_distance_scale=0.7,
        max_attempts_per_sample=20,
    )
    rattle_samples = attach_calculator(rattle_sampler.sample(seed))
    rattle_path = write_samples(rattle_samples, output_dir / "rattle_samples.extxyz")

    md_sampler = MDSampler(
        dynamics_name="Langevin",
        timestep_fs=1.0,
        steps=50,
        sample_interval=10,
        initialize_velocities=True,
        velocity_temperature_K=300.0,
        zero_center_of_mass_momentum=True,
        dynamics_kwargs={
            "temperature_K": 300.0,
            "friction": 0.02,
        },
        log_progress=True,
        log_interval=10,
    )
    md_samples = attach_calculator(md_sampler.sample(seed))
    md_path = write_samples(md_samples, output_dir / "md_samples.extxyz")

    dimer_sampler = DimerSampler(
        control_kwargs={
            "logfile": None,
            "eigenmode_logfile": None,
        },
        displace_kwargs={
            "method": "vector",
        },
        translate_kwargs={
            "logfile": None,
        },
        run_kwargs={
            "fmax": 0.05,
            "steps": 5,
        },
        eigenmodes=None,
        random_seed=11,
        comm=None,
        collect_trajectory=True,
        sample_interval=1,
    )
    dimer_samples = attach_calculator(dimer_sampler.sample(seed))
    dimer_path = write_samples(dimer_samples, output_dir / "dimer_samples.extxyz")

    print("outputs:")
    print(f"seed:   {seed_path}")
    print(f"rattle: {rattle_path}")
    print(f"md:     {md_path}")
    print(f"dimer:  {dimer_path}")

    print_samples("rattle", rattle_samples)
    print_samples("md", md_samples)
    print_samples("dimer", dimer_samples)


if __name__ == "__main__":
    main()
