"""Reference sample: reasonable hyperparameters aimed at actually lowering
validation error over real iterations, not just proving the SLURM-chained
mechanics work (that's already covered by workflow_loop.py's coarse/fast
test settings).

Differences from workflow_loop.py, and the reasoning behind each:

  - Bootstraps the initial training pool from real DFT data
    (Pt_surface_test/DFT_data/) via the parser, instead of a single
    hand-supplied seed structure -- gives the AL loop a real, if small,
    starting point, and lets select_seed_records() (not the user) choose
    which structures actually become MD seeds.
  - Plain finetune() of "medium-omat-0" (OMat24 foundation model), not LoRA
    -- naive full fine-tuning, per explicit request.
  - Real epoch budget (EPOCHS) with an SWA phase near the end -- MACE's own
    standard recipe for lower validation error, not just "train longer".
    EPOCHS=5 elsewhere in this repo was only ever a fast mechanics-test
    value, never meant to produce a good model.
  - batch_size/valid_fraction sized for a small, growing AL pool rather than
    a fixed large batch that could exceed early-iteration dataset size.
  - model_factory_gpu_ids parallelizes the N_MODELS committee -- see the
    partition note below for why only 2 GPUs are requested, not N_MODELS.
  - pre_uncertainty_curation_pipeline=ForceCutoffCurate(...) filters
    unphysical MD-sampled candidates (colliding/diverging configurations)
    before wasting ensemble-inference compute scoring them. Passed directly
    (not wrapped in StructureCurationPipeline) since ForceCutoffCurate needs
    forces already computed -- a different precondition than the
    cheap/descriptor/cluster stages StructureCurationPipeline composes, and
    the code consuming pre_uncertainty_curation_pipeline only ever calls
    .curate_records() on it, so passing any BaseStructureCuration works.
  - seed_selection_mode="highest_uncertainty" with a capped seed_k, not
    "all" -- "all" means every iteration's MD sampling cost grows
    unboundedly as the train pool grows; capping keeps cost bounded while
    focusing exploration where the current model is least confident.
  - uncertainty_k/coverage_k sized for a real per-iteration DFT budget: small
    enough to stay adaptive (retrain before committing more DFT), with a
    nonzero coverage_k so selection isn't purely uncertainty-chasing (which
    can bias the training set toward one region of configuration space).
  - ModelErrorStoppingCriteria's defaults (50 meV/A force, 5 meV/atom
    energy) are kept as-is -- already grounded in real literature numbers
    (a well-converged model reports ~14 meV/A), so 50 is an achievable,
    meaningful "good enough" bar to actually reach in a handful of
    iterations, not an arbitrary number.

Partition/GPU note: N_MODELS=4 (matching the literature paper's own
choice), but this account (jgreeley) has no confirmed billing allocation on
the `ai` partition (8 GPUs/node), and `smallgpu` (which it can use) only has
2 GPUs/node. So model_factory_gpu_ids=[0, 1] gives 2x parallelism via
cycling (models 0&2 share GPU 0, models 1&3 share GPU 1), not full 4x --
see runs/workflow_loop_reasonable_defaults/*.run for the matching
--gpus-per-node=2.

Known gaps/approximations:
  - Single explore/resume SLURM job chain, same non-blocking pattern as
    workflow_loop.py -- see that script's own docstring-equivalent comments
    for why dft_root_dir is never passed directly to run_iteration() here.
  - TRAIN_PATH is '.aselmdb' (required for parsed_labeled_records, unlike
    workflow_loop.py's '.extxyz'), so all training pool appends here use
    write_structure_records(..., output_format="aselmdb").
  - All real structures under PARSE_BASE_DIR are pure Pt (78) -- add more
    elements/E0s if pointing this at a system with adsorbates. E0s isn't
    passed here at all: fine-tuning inherits the foundation model's own
    calibrated atomic reference energies, only atomic_numbers is required
    for .aselmdb inputs regardless of training method.
"""

from __future__ import annotations

import sys
from pathlib import Path

from al_dirac.curator.cheap_curate import CheapCurate
from al_dirac.curator.force_cutoff_curate import ForceCutoffCurate
from al_dirac.curator.structure_curation import StructureCurationPipeline
from al_dirac.dft.batch import BatchDFTRunner
from al_dirac.dft.vasp import VASPLabeler
from al_dirac.models.mace_model import MACEModel
from al_dirac.models.model_factory import ModelEnsembleFactory, SeedStrategy
from al_dirac.parser.structure_parser import write_structure_records
from al_dirac.samplers.md import MDSampler
from al_dirac.selection.ensemble_uncertainty_selector import EnsembleUncertaintySelector
from al_dirac.uncertainty.ensemble import EnsembleUncertainty
from al_dirac.workflow.active_learning import ActiveLearningWorkflow
from al_dirac.workflow.restart import (
    collect_completed_dft_from_batches,
    incomplete_dft_batches,
    load_artifact,
    load_state,
    save_artifact,
    save_state,
    stage_artifact_path,
)
from al_dirac.workflow.state import WorkflowState
from al_dirac.workflow.workflow_logger import WorkflowLogger

DEVICE = "cuda"

RUN_DIR = Path("Pt_surface_test/real_example_outputs/workflow_loop_reasonable_defaults")
ARTIFACT_DIR = RUN_DIR / "artifacts"
PLOT_DIR = RUN_DIR / "plots"
MODEL_FACTORY_CHECKPOINT_DIR = RUN_DIR / "model_factory"
DFT_ROOT_DIR = RUN_DIR / "dft_jobs"
STATE_FILE = RUN_DIR / "state.json"
PREPARED_BATCHES_FILE = RUN_DIR / "prepared_batches.pkl"
SUBMIT_SCRIPTS_FILE = RUN_DIR / "submit_scripts.txt"

# Bootstrapped from real DFT data at iteration 0 (see PARSE_BASE_DIR below),
# then grown with newly DFT-labeled structures after every iteration.
# .aselmdb, not .extxyz: parsed_labeled_records can only be appended to an
# .aselmdb train_path.
TRAIN_PATH = RUN_DIR / "train.aselmdb"

PARSE_BASE_DIR = Path("Pt_surface_test/DFT_data")
PARSE_KWARGS = {
    "filename_glob": "*/vasprun.xml",
    "recursive": False,
    "file_format": "vasp-xml",
    "read_index": -1,
    "source": "pt_surface_relaxation",
    "require_sibling_filename": "CONTCAR",
    "require_sibling_contains": {"OUTCAR": "reached required accuracy"},
}
# Modest tolerance, not exact-match (the 6-decimal default is essentially
# exact-match only) -- catches true near-duplicate restart/checkpoint
# repeats without conflating genuinely different structures.
PARSER_CURATION_PIPELINE = StructureCurationPipeline(
    cheap_curate=CheapCurate(position_tolerance=0.01, energy_tolerance=0.001),
    use_cheap=True,
    use_descriptor=False,
    use_cluster=False,
)

SUBMISSION_TEMPLATE = Path("Pt_surface_test/examples/vasp_batch.run")

FOUNDATION_MODEL = "medium-omat-0"
N_MODELS = 4
SEED_START = 7

# Real training budget with an SWA phase -- MACE's standard recipe, not just
# "more epochs". See module docstring.
EPOCHS = 250
SWA_START_EPOCH = 200
SWA_LR = 0.0005
SWA_ENERGY_WEIGHT = 1000.0
SWA_FORCES_WEIGHT = 100.0
LEARNING_RATE = 0.001
ENERGY_WEIGHT = 1.0
FORCES_WEIGHT = 100.0
BATCH_SIZE_TRAIN = 4
VALID_FRACTION = 0.15

# Any already fine-tuned/foundation model -- seeds the MD sampler's
# calculator and EnsembleUncertainty's non-empty constructor check before
# the fresh committee is trained. Point this at any real .model file you
# have (e.g. one produced by mace_finetune.py).
SAMPLER_MODEL_PATH = Path(
    "Pt_surface_test/real_example_outputs/mace_finetune/models/pt_surface_finetune.model"
)

# Unphysical-structure filter applied to freshly MD-sampled candidates,
# before the (comparatively expensive) ensemble-inference scoring step.
# max_force is deliberately permissive: MD runs hot (1000 K, see below) on
# purpose to explore beyond near-equilibrium configurations, so legitimately
# useful high-uncertainty candidates can show real, if elevated, forces --
# this threshold is meant to catch true collisions/divergence, not just
# "hotter than equilibrium".
PRE_UNCERTAINTY_CURATION_PIPELINE = ForceCutoffCurate(max_force=50.0)

# Cap how many labeled structures get used as MD seeds each iteration --
# "all" (workflow_loop.py's choice) means unbounded MD cost growth as the
# train pool grows across iterations. "highest_uncertainty" also focuses
# sampling where the current model is least confident, rather than
# resampling everywhere uniformly.
SEED_SELECTION_MODE = "highest_uncertainty"
SEED_K = 5

DFT_BATCH_SIZE = 3
VASP_CALCULATOR_KWARGS = {
    "xc": "PBE",
    "encut": 400,
    "ediff": 1e-5,
    "ibrion": -1,
    "nsw": 0,
    "ismear": 1,
    "sigma": 0.2,
    "prec": "Accurate",
    "lreal": False,
    "lwave": False,
    "algo": "VeryFast",
    "npar": 16,
    "nelm": 300,
    "kpts": (5, 1, 1),
    "gamma": True,
}

DFT_RUNNER = BatchDFTRunner(
    labeler=VASPLabeler(calculator_kwargs=VASP_CALCULATOR_KWARGS),
    batch_size=DFT_BATCH_SIZE,
    submission_template=SUBMISSION_TEMPLATE,
    run_command_template="mpirun -np $SLURM_NTASKS vasp_std",
)


def run_explore() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DFT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMIT_SCRIPTS_FILE.unlink(missing_ok=True)
    logger = WorkflowLogger(run_dir=RUN_DIR)

    if STATE_FILE.exists():
        state = load_state(STATE_FILE)
    else:
        state = WorkflowState(run_id="workflow_loop_reasonable_defaults", iteration=0)

    iteration_model_factory_dir = (
        MODEL_FACTORY_CHECKPOINT_DIR / f"iteration_{state.iteration:04d}"
    )

    selecting_artifact = stage_artifact_path(ARTIFACT_DIR, state.iteration, "selecting")
    if selecting_artifact.exists():
        print(f"iteration {state.iteration}: resuming from existing selection artifact {selecting_artifact}")
        selected_records = load_artifact(selecting_artifact)
    else:
        sampler_model = MACEModel.load(
            SAMPLER_MODEL_PATH, device=DEVICE, default_dtype="float32", head="Default"
        )
        uncertainty = EnsembleUncertainty(models=[sampler_model])

        selector = EnsembleUncertaintySelector(
            score_expression="force_max_uncertainty",
            larger_is_better=True,
            min_score=None,
        )

        factory = ModelEnsembleFactory(
            MACEModel,
            n_models=N_MODELS,
            name_prefix="pt_surface_finetune_ensemble",
            training_method="finetune",
            base_kwargs={
                "model": "MACE",
                "device": DEVICE,
                "default_dtype": "float32",
                "checkpoint_dir": iteration_model_factory_dir,
            },
        )
        factory.add_strategy(SeedStrategy(seed_start=SEED_START))

        sampler = MDSampler(
            dynamics_name="Langevin",
            timestep_fs=1.0,
            steps=200,
            sample_interval=40,
            initialize_velocities=True,
            velocity_temperature_K=1000.0,
            dynamics_kwargs={"temperature_K": 1000.0, "friction": 0.01},
            calculator=sampler_model.calculator,
        )

        workflow = ActiveLearningWorkflow(
            uncertainty=uncertainty,
            selector=selector,
            pre_uncertainty_curation_pipeline=None,
            parser_curation_pipeline=None,
            model=None,
            model_factory=factory,
            sampler=sampler,
            dft_runner=None,
            coverage_curation_pipeline=None,
        )

        selected_records = workflow.run_iteration(
            state,
            uncertainty_k=5,
            selection_mode="auto",
            cold_start_k=None,
            candidate_structures=None,
            model=None,
            train_path=TRAIN_PATH,
            # No hand-supplied seed_structures -- iteration 0 seeds come
            # from the real parsed pool (below), later iterations from
            # TRAIN_PATH, both via select_seed_records().
            seed_structures=None,
            dft_root_dir=None,
            train_prediction_model=False,
            train_model_ensemble=True,
            train_kwargs=None,
            model_factory=factory,
            model_factory_train_kwargs={
                "foundation_model": FOUNDATION_MODEL,
                "atomic_numbers": [78],
                "energy_key": "REF_energy",
                "forces_key": "REF_forces",
                "stress_key": "REF_stress",
                "batch_size": BATCH_SIZE_TRAIN,
                "valid_batch_size": BATCH_SIZE_TRAIN,
                "valid_fraction": VALID_FRACTION,
                "max_num_epochs": EPOCHS,
                "lr": LEARNING_RATE,
                "ema": True,
                "ema_decay": 0.99,
                "loss": "weighted",
                "num_workers": 0,
                "enable_cueq": True,
                "work_dir": str(iteration_model_factory_dir),
                "extra_args": [
                    "--device", DEVICE,
                    "--default_dtype", "float32",
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
            model_factory_gpu_ids=[0, 1],
            load_model_factory_checkpoints=True,
            parse_base_dir=PARSE_BASE_DIR if state.iteration == 0 else None,
            parse_kwargs=PARSE_KWARGS,
            parser_curation_pipeline=PARSER_CURATION_PIPELINE,
            parser_curation_kwargs=None,
            seed_selection_mode=SEED_SELECTION_MODE,
            seed_k=SEED_K,
            seed_selection_curation_pipeline=None,
            seed_selection_curation_kwargs=None,
            seed_selection_uncertainty=uncertainty,
            seed_selection_score_key="force_max_uncertainty",
            seed_selection_random_seed=None,
            sampler=sampler,
            sampler_kwargs=None,
            common_data={"example": "workflow_loop_reasonable_defaults"},
            pre_uncertainty_curation_pipeline=PRE_UNCERTAINTY_CURATION_PIPELINE,
            pre_uncertainty_curation_kwargs=None,
            uncertainty=uncertainty,
            uncertainty_stop_key="force_max_uncertainty",
            uncertainty_stop_expression=None,
            model_error_stop_force_threshold=0.05,
            model_error_stop_energy_threshold=0.005,
            model_error_stop_statistic="max",
            selector=selector,
            coverage_k=2,
            coverage_curation_pipeline=None,
            coverage_curation_kwargs=None,
            dft_runner=None,
            labeler_kwargs=None,
            template_replacements=None,
            logger=logger,
            artifact_dir=ARTIFACT_DIR,
            plot_dir=PLOT_DIR,
            restart_from_stage=None,
        )

        save_state(STATE_FILE, state)

    if not selected_records:
        print(f"workflow_loop_reasonable_defaults stopped at iteration {state.iteration}: {state.stop_reason}")
        return

    structures = [record["atoms"] for record in selected_records]
    iteration_dft_dir = DFT_ROOT_DIR / f"iteration_{state.iteration:04d}"
    prepared_batches = DFT_RUNNER.prepare_batches(
        structures=structures,
        root_dir=iteration_dft_dir,
        records=selected_records,
    )
    save_artifact(PREPARED_BATCHES_FILE, prepared_batches)

    submit_scripts = [str(batch["submit_script"]) for batch in prepared_batches]
    SUBMIT_SCRIPTS_FILE.write_text("\n".join(submit_scripts) + "\n")

    total_jobs = sum(len(batch.get("jobs", [])) for batch in prepared_batches)
    logger.log_note(
        f"DFT batches prepared in {iteration_dft_dir}: "
        f"{len(prepared_batches)} batch(es), {total_jobs} job(s)",
        state=state,
        dft_dir=str(iteration_dft_dir),
        n_batches=len(prepared_batches),
        n_jobs=total_jobs,
    )

    print(f"iteration {state.iteration}: prepared {len(prepared_batches)} DFT batch(es)")
    print(f"submit scripts written to {SUBMIT_SCRIPTS_FILE}")


def run_resume() -> None:
    state = load_state(STATE_FILE)
    logger = WorkflowLogger(run_dir=RUN_DIR)
    prepared_batches = load_artifact(PREPARED_BATCHES_FILE)
    total_jobs = sum(len(batch.get("jobs", [])) for batch in prepared_batches)
    dft_dir = (
        str(prepared_batches[0]["batch_dir"].parent) if prepared_batches else None
    )

    incomplete = incomplete_dft_batches(prepared_batches, DFT_RUNNER)
    if incomplete:
        print(f"warning: {len(incomplete)} DFT batch(es) did not complete successfully")

    labeled_records = collect_completed_dft_from_batches(prepared_batches, DFT_RUNNER)
    print(f"collected {len(labeled_records)} labeled structures")

    logger.log_note(
        f"DFT results collected from {dft_dir}: "
        f"{len(labeled_records)} of {total_jobs} job(s) labeled, "
        f"{len(incomplete)} incomplete batch(es)",
        state=state,
        dft_dir=dft_dir,
        n_jobs=total_jobs,
        n_labeled=len(labeled_records),
        n_incomplete_batches=len(incomplete),
    )

    if total_jobs > 0 and not labeled_records:
        raise RuntimeError(
            f"All {total_jobs} DFT job(s) failed or did not complete -- no "
            "labeled structures were collected. Not continuing the loop with "
            "unchanged training data; fix the underlying DFT issue and "
            "resubmit this iteration manually (see al_dirac.workflow.restart)."
        )

    if labeled_records:
        write_structure_records(
            labeled_records,
            TRAIN_PATH,
            output_format="aselmdb",
            split="train",
            is_labeled=True,
            is_selected=False,
            source="active_learning",
        )

    save_state(STATE_FILE, state)
    print(f"training pool updated, next iteration: {state.iteration}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"explore", "resume"}:
        raise SystemExit("usage: workflow_loop_reasonable_defaults.py [explore|resume]")
    (run_explore if sys.argv[1] == "explore" else run_resume)()


if __name__ == "__main__":
    main()
