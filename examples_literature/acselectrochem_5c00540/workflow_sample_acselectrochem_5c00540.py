"""Reference sample: parameter choices from a real published study, mapped
onto al_dirac's actual API.

Mathanker, A.; Guo, J.; Goldsmith, B. R.; Varley, J. B.; Govindarajan, N.
"Estimating Potential-Dependent Physicochemical Properties at Metal-Electrolyte
Interfaces Using Machine Learning Interatomic Potentials."
ACS Electrochem. 2026, 2, 1176-1189. https://doi.org/10.1021/acselectrochem.5c00540

The paper and its Supporting Information are included alongside this script
(ec5c00540.pdf, ec5c00540_si_001.pdf).

This is NOT turnkey-runnable: it does not include the paper's actual
Au/Cu/Rh(111)-electrolyte structures or their initial AIMD trajectories --
SEED_SOURCE_FILE below is a placeholder you must point at your own structure.
It exists to show how their reported hyperparameters map onto al_dirac's
MACEModel/ModelEnsembleFactory/EnsembleUncertaintySelector/BatchDFTRunner API,
scoped to the Cu(111) + pure water (0 uC/cm^2) system as the simplest case
the paper reports.

Known gaps/approximations versus the paper (called out inline below):
  - Their initial training set comes from literal ab initio MD (AIMD); this
    script uses MDSampler (MLIP-driven MD) instead, which is the correct
    analogue for their *later* MACE-MD-driven active-learning iterations, not
    their AIMD bootstrap stage.
  - Their acceptance rule keeps configurations with committee force
    disagreement (F_err) in the band [0.1, 0.3]. F_err is a genuine relative
    (dimensionless) quantity -- per-atom force disagreement divided by that
    atom's own mean force magnitude, reduced over atoms. UNCERTAINTY_SCORE_
    EXPRESSION below reproduces this exactly via score_expression
    ("max(force_atomwise_uncertainty / force_magnitude_per_atom)"), so
    min_score/max_score = 0.1/0.3 now carry the same physical meaning as in
    the paper, not just the same mechanism. Only 10-15% of what passes that
    band is then randomly kept for further SPCs -- reproduced via
    RandomCurator (RANDOM_CURATION_FRACTION), though the paper doesn't state
    an exact fraction within 10-15% or a random seed.
  - VASP smearing (ISMEAR/SIGMA) and PREC were not stated in the excerpted
    text/SI sections read for this sample -- left at ASE/VASP defaults rather
    than guessed.
  - Committee training batch size was not stated in the excerpted sections
    read for this sample -- a placeholder is used and flagged below.
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read

from al_dirac.curator.random_curate import RandomCurator
from al_dirac.dft.batch import BatchDFTRunner
from al_dirac.dft.vasp import VASPLabeler
from al_dirac.models.mace_model import MACEModel
from al_dirac.models.model_factory import ModelEnsembleFactory, SeedStrategy
from al_dirac.samplers.md import MDSampler
from al_dirac.selection.ensemble_uncertainty_selector import EnsembleUncertaintySelector
from al_dirac.uncertainty.ensemble import EnsembleUncertainty
from al_dirac.workflow.active_learning import ActiveLearningWorkflow
from al_dirac.workflow.state import WorkflowState
from al_dirac.workflow.workflow_logger import WorkflowLogger

DEVICE = "cuda"

# Placeholder -- point this at your own Cu(111) + water slab (96 water
# molecules, 0 uC/cm^2 case from Table S1) in extxyz format, energy_key
# "REF_energy" / forces_key "REF_forces".
SEED_SOURCE_FILE = Path("path/to/your/cu111_water_seed.extxyz")
TRAIN_PATH = Path("path/to/your/cu111_water_train.extxyz")
VALID_FILE = Path("path/to/your/cu111_water_valid.extxyz")
TEST_FILE = Path("path/to/your/cu111_water_test.extxyz")
# Any already-trained model (e.g. a prior iteration's committee member) used
# only to seed the MD sampler's calculator and EnsembleUncertainty's
# non-empty constructor check before the fresh committee is trained -- same
# pattern as workflow_loop.py's sampler_model.
SAMPLER_MODEL_PATH = Path("path/to/your/prior_or_foundation.model")

RUN_DIR = Path("Pt_surface_test/real_example_outputs/acselectrochem_5c00540")
ARTIFACT_DIR = RUN_DIR / "artifacts"
PLOT_DIR = RUN_DIR / "plots"
MODEL_FACTORY_CHECKPOINT_DIR = RUN_DIR / "model_factory"

# --- MACE architecture: paper section "MACE Architecture" + SI Figure S1 ---
# ScaleShiftMACE, two interaction layers, 64x0e+64x1o+64x2e hidden irreps,
# r_max=5 A, 8 radial Bessel basis functions, real spherical harmonics to
# l=2. num_radial_basis/max_ell aren't named MACEModel parameters, so they
# go through extra_args below.
MACE_MODEL_ARCHITECTURE = "ScaleShiftMACE"
HIDDEN_IRREPS = "64x0e + 64x1o + 64x2e"
NUM_INTERACTIONS = 2
R_MAX = 5.0
NUM_RADIAL_BASIS = 8
MAX_ELL = 2

# --- Training schedule: paper section "MACE Architecture" ---
# "trained for 1000 epochs using the Adam optimizer with an initial learning
# rate of 0.01. After 800 epochs, training entered a stochastic weight
# averaging phase in which the learning rate was reduced to 0.001 and the
# energy and force weights were set to 1000 and 100."
MAX_NUM_EPOCHS = 1000
LEARNING_RATE = 0.01
LOSS = "weighted"
ENERGY_WEIGHT = 1.0
FORCES_WEIGHT = 100.0
SWA_START_EPOCH = 800
SWA_LR = 0.001
SWA_ENERGY_WEIGHT = 1000.0
SWA_FORCES_WEIGHT = 100.0

# Not stated in the excerpted sections read for this sample -- placeholder.
BATCH_SIZE = 32

# "A committee of four MACE models" (Introduction / Active Learning Workflow)
N_MODELS = 4
SEED_START = 0

# --- DFT (VASP) settings for single-point labeling: "Density Functional
# Theory and Ab Initio Molecular Dynamics Simulations" ---
# revised PBE (RPBE) + Grimme D3 dispersion, 450 eV cutoff, 1e-6 eV SCF
# convergence, max 250 electronic steps, dipole correction perpendicular to
# the slab with a fixed reference point at (0.5, 0.5, 0.25), 2x3x1 k-points
# for the slab. ISMEAR/SIGMA/PREC were not stated for the slab SPCs in the
# excerpted sections -- left at VASP/ASE defaults rather than guessed.
VASP_CALCULATOR_KWARGS = {
    "xc": "RPBE",
    "ivdw": 11,  # Grimme D3, zero-damping (paper cites both zero- and
                 # BJ-damping D3 references; zero-damping used here as the
                 # more commonly reported default -- not explicitly
                 # disambiguated in the excerpted text).
    "encut": 450,
    "ediff": 1e-6,
    "nelm": 250,
    # Single-point labeling, matching the paper's DFT/SPC evaluations of
    # AIMD/MACE-MD-sampled snapshots -- not a relaxation.
    "ibrion": -1,
    "nsw": 0,
    "ldipol": True,
    "idipol": 3,
    "dipol": (0.5, 0.5, 0.25),
    "kpts": (2, 3, 1),
    "gamma": True,  # inferred, not explicitly stated for the slab mesh
}

DFT_BATCH_SIZE = 3
SUBMISSION_TEMPLATE = Path("Pt_surface_test/examples/vasp_batch.run")

# --- MD sampling: "Active Learning Workflow" + "Density Functional Theory
# and Ab Initio Molecular Dynamics Simulations" ---
# NVT ensemble, Nose-Hoover thermostat, 300 K, 1 fs timestep. This models
# their *MACE-MD*-driven active-learning iterations (MLIP forces), not their
# initial AIMD bootstrap stage (real DFT forces every step).
MD_TEMPERATURE_K = 300.0
MD_TIMESTEP_FS = 1.0
MD_TDAMP_FS = 100.0  # ~100x timestep, a standard Nose-Hoover damping choice;
                      # not stated in the paper.
MD_STEPS = 200
MD_SAMPLE_INTERVAL = 40

# --- Selection: "Active Learning Workflow" ---
# "configurations with F_err values between 0.1 and 0.3 were selected for
# additional SPCs... configurations with F_err below 0.1 were excluded as
# redundant, while those above 0.3 were omitted to avoid introducing
# configurations that may fall outside the physically meaningful region."
#
# F_err = per-atom force disagreement divided by that atom's own mean force
# magnitude, then reduced to a single number -- a genuine relative
# (dimensionless) quantity, not the same thing as al_dirac's raw
# force_max_uncertainty (which is absolute, in eV/A). The division has to
# happen per atom *before* reducing over atoms, so this is expressed
# directly as a score_expression over the two per-atom fields
# EnsembleUncertainty already provides, rather than a single named field:
UNCERTAINTY_SCORE_EXPRESSION = "max(force_atomwise_uncertainty / force_magnitude_per_atom)"
UNCERTAINTY_MIN_SCORE = 0.1
UNCERTAINTY_MAX_SCORE = 0.3

# "...only 10-15% of the configurations satisfying this criterion were
# randomly selected for further SPCs" (same paragraph as the F_err band
# above) -- band-passing candidates are further randomly thinned before
# being sent to DFT. 0.125 is the midpoint of the reported 10-15% range;
# the paper does not report an exact value or a random seed.
RANDOM_CURATION_FRACTION = 0.125
RANDOM_CURATION_SEED = 0


def load_seed_structures() -> list:
    structures = read(SEED_SOURCE_FILE, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    return [structures[0].copy()]


def main() -> None:
    state = WorkflowState(run_id="acselectrochem_5c00540", iteration=0)
    logger = WorkflowLogger(run_dir=RUN_DIR)

    # Scoped by iteration so a later iteration's retraining doesn't overwrite
    # this iteration's committee models/checkpoints/logs -- see
    # workflow_loop.py's iteration_model_factory_dir for the same pattern.
    iteration_model_factory_dir = (
        MODEL_FACTORY_CHECKPOINT_DIR / f"iteration_{state.iteration:04d}"
    )

    # Placeholder committee model used only to satisfy EnsembleUncertainty's
    # non-empty constructor check and to supply the MD sampler's calculator
    # before the freshly-trained committee is available -- see
    # workflow_loop.py's sampler_model for the same pattern.
    sampler_model = MACEModel.load(
        SAMPLER_MODEL_PATH, device=DEVICE, default_dtype="float32"
    )
    uncertainty = EnsembleUncertainty(models=[sampler_model])

    selector = EnsembleUncertaintySelector(
        score_expression=UNCERTAINTY_SCORE_EXPRESSION,
        larger_is_better=True,
        min_score=UNCERTAINTY_MIN_SCORE,
        max_score=UNCERTAINTY_MAX_SCORE,
    )

    # Randomly keeps only 10-15% of the band-passing candidates, matching the
    # paper's acceptance rule exactly (see RANDOM_CURATION_FRACTION above).
    random_curator = RandomCurator(
        fraction=RANDOM_CURATION_FRACTION,
        seed=RANDOM_CURATION_SEED,
    )

    # Committee of 4 same-architecture models trained from scratch, differing
    # by random seed -- not LoRA fine-tuning (the paper trains its own
    # dedicated potential, it does not fine-tune a foundation model).
    factory = ModelEnsembleFactory(
        MACEModel,
        n_models=N_MODELS,
        name_prefix="acselectrochem_committee",
        training_method="train",
        base_kwargs={
            "model": MACE_MODEL_ARCHITECTURE,
            "device": DEVICE,
            "default_dtype": "float32",
            "checkpoint_dir": iteration_model_factory_dir,
        },
    )
    factory.add_strategy(SeedStrategy(seed_start=SEED_START))

    sampler = MDSampler(
        dynamics_name="NoseHooverChainNVT",
        timestep_fs=MD_TIMESTEP_FS,
        steps=MD_STEPS,
        sample_interval=MD_SAMPLE_INTERVAL,
        initialize_velocities=True,
        velocity_temperature_K=MD_TEMPERATURE_K,
        dynamics_kwargs={
            "temperature_K": MD_TEMPERATURE_K,
            "tdamp": MD_TDAMP_FS * units.fs,
        },
        calculator=sampler_model.calculator,
    )

    dft_runner = BatchDFTRunner(
        labeler=VASPLabeler(calculator_kwargs=VASP_CALCULATOR_KWARGS),
        batch_size=DFT_BATCH_SIZE,
        submission_template=SUBMISSION_TEMPLATE,
        run_command_template="mpirun -np $SLURM_NTASKS vasp_std",
    )

    workflow = ActiveLearningWorkflow(
        uncertainty=uncertainty,
        selector=selector,
        pre_uncertainty_curation_pipeline=None,
        parser_curation_pipeline=None,
        model=None,
        model_factory=factory,
        sampler=sampler,
        dft_runner=dft_runner,
        coverage_curation_pipeline=None,
        uncertainty_curation_pipeline=random_curator,
    )

    selected_records = workflow.run_iteration(
        state,
        uncertainty_k=None,  # min_score/max_score band does the selecting
        selection_mode="auto",
        cold_start_k=None,
        candidate_structures=None,
        model=None,
        train_path=TRAIN_PATH,
        seed_structures=load_seed_structures(),
        dft_root_dir=None,  # stop before DFT -- no real structures to label
        valid_path=VALID_FILE,
        train_prediction_model=False,
        train_model_ensemble=True,
        train_kwargs=None,
        model_factory=factory,
        model_factory_train_kwargs={
            "test_file": TEST_FILE,
            "hidden_irreps": HIDDEN_IRREPS,
            "num_interactions": NUM_INTERACTIONS,
            "r_max": R_MAX,
            "energy_key": "REF_energy",
            "forces_key": "REF_forces",
            "E0s": "average",  # matches the paper's least-squares E0
                                # regression from the training data
            "batch_size": BATCH_SIZE,
            "valid_batch_size": BATCH_SIZE,
            "max_num_epochs": MAX_NUM_EPOCHS,
            "lr": LEARNING_RATE,
            "loss": LOSS,
            "num_workers": 0,
            "enable_cueq": True,
            "work_dir": str(iteration_model_factory_dir),
            "extra_args": [
                "--device", DEVICE,
                "--default_dtype", "float32",
                "--num_radial_basis", str(NUM_RADIAL_BASIS),
                "--max_ell", str(MAX_ELL),
                "--energy_weight", str(ENERGY_WEIGHT),
                "--forces_weight", str(FORCES_WEIGHT),
                "--swa",
                "--start_swa", str(SWA_START_EPOCH),
                "--swa_lr", str(SWA_LR),
                "--swa_energy_weight", str(SWA_ENERGY_WEIGHT),
                "--swa_forces_weight", str(SWA_FORCES_WEIGHT),
                "--error_table", "PerAtomMAE",
                "--plot", "False",
            ],
        },
        load_model_factory_checkpoints=True,
        parse_base_dir=None,
        parse_kwargs=None,
        parser_curation_pipeline=None,
        parser_curation_kwargs=None,
        seed_selection_mode="all",
        seed_k=None,
        seed_selection_curation_pipeline=None,
        seed_selection_curation_kwargs=None,
        seed_selection_uncertainty=None,
        seed_selection_score_key=UNCERTAINTY_SCORE_EXPRESSION,
        seed_selection_random_seed=None,
        sampler=sampler,
        sampler_kwargs=None,
        common_data={"example": "acselectrochem_5c00540"},
        pre_uncertainty_curation_pipeline=None,
        pre_uncertainty_curation_kwargs=None,
        uncertainty=uncertainty,
        # Stop once even the most-uncertain candidate drops below F_err's own
        # "redundant" cutoff (0.1) -- same threshold the paper uses to
        # exclude configurations from being added to the training set.
        uncertainty_stop_key=UNCERTAINTY_SCORE_EXPRESSION,
        uncertainty_stop_threshold=UNCERTAINTY_MIN_SCORE,
        uncertainty_stop_statistic="max",
        uncertainty_stop_expression=UNCERTAINTY_SCORE_EXPRESSION,
        selector=selector,
        uncertainty_curation_pipeline=random_curator,
        coverage_k=0,
        coverage_curation_pipeline=None,
        coverage_curation_kwargs=None,
        dft_runner=dft_runner,
        labeler_kwargs=None,
        template_replacements=None,
        logger=logger,
        artifact_dir=ARTIFACT_DIR,
        plot_dir=PLOT_DIR,
        restart_from_stage=None,
    )

    print("acselectrochem_5c00540 sample complete")
    print(f"selected records: {len(selected_records)}")
    print(f"run dir: {RUN_DIR}")


if __name__ == "__main__":
    main()
