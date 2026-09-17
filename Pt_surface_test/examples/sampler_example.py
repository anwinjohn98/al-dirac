from __future__ import annotations

from pathlib import Path

from ase.io import read

from al_dirac.models.mace_model import MACEModel
from al_dirac.samplers.dimer import DimerSampler
from al_dirac.samplers.md import MDSampler
from al_dirac.samplers.rattle import RattleSampler
from al_dirac.workflow.outputs import write_records_extxyz

INPUT_FILE = Path("real_example_outputs/parser/train.extxyz")
MODEL_PATH = Path(
    "real_example_outputs/mace_multihead_finetune/models/pt_surface_multihead_finetune.model"
)

OUTPUT_DIR = Path("real_example_outputs/sampler")

DEVICE = "cuda"


def load_seed_structure():
    structures = read(INPUT_FILE, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    return structures[0].copy()


def attach_calculator(samples, calculator):
    for atoms in samples:
        atoms.calc = calculator
    return samples


def print_samples(name: str, samples) -> None:
    print(f"{name}: {len(samples)} samples")
    for index, atoms in enumerate(samples):
        energy = atoms.get_potential_energy()
        print(f"  frame={index}, atoms={len(atoms)}, energy={energy:.6f}")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Run examples/parser_example.py first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run examples/mace_multihead_finetune.py first.")

    seed = load_seed_structure()
    model = MACEModel.load(MODEL_PATH, device=DEVICE, default_dtype="float32")
    calculator = model.calculator
    seed.calc = calculator

    seed_path = write_records_extxyz([{"atoms": seed}], OUTPUT_DIR / "seed.extxyz")

    rattle_sampler = RattleSampler(
        stdev=0.05,
        n_samples=5,
        seed=7,
        min_distance_scale=0.7,
        max_attempts_per_sample=20,
    )
    rattle_samples = attach_calculator(rattle_sampler.sample(seed), calculator)
    rattle_path = write_records_extxyz(
        [{"atoms": atoms} for atoms in rattle_samples],
        OUTPUT_DIR / "rattle_samples.extxyz",
    )

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
    md_samples = attach_calculator(md_sampler.sample(seed), calculator)
    md_path = write_records_extxyz(
        [{"atoms": atoms} for atoms in md_samples],
        OUTPUT_DIR / "md_samples.extxyz",
    )

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
    dimer_samples = attach_calculator(dimer_sampler.sample(seed), calculator)
    dimer_path = write_records_extxyz(
        [{"atoms": atoms} for atoms in dimer_samples],
        OUTPUT_DIR / "dimer_samples.extxyz",
    )

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
