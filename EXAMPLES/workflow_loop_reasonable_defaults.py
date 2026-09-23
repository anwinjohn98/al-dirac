"""Reference sample: reasonable hyperparameters aimed at actually lowering
validation error over real iterations, not just proving the SLURM-chained
mechanics work (that's already covered by workflow_loop.py's coarse/fast
test settings). Reasoning for individual choices lives next to the
relevant constant/function below, not here.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from mace.calculators import mace_mp
from mace.calculators.foundations_models import download_mace_mp_checkpoint

from al_dirac.curator.cheap_curate import CheapCurate
from al_dirac.curator.cluster_curate import ClusterCurate
from al_dirac.curator.force_cutoff_curate import ForceCutoffCurate
from al_dirac.curator.structure_curation import StructureCurationPipeline
from al_dirac.dft.batch import BatchDFTRunner
from al_dirac.dft.vasp import VASPLabeler
from al_dirac.features.representative_selection import RepresentativeSelection
from al_dirac.features.similarity_graph import SimilarityGraph
from al_dirac.features.soap_descriptor import SOAPDescriptor
from al_dirac.features.structure_similarity import StructureSimilarity
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
# Same convention as mace_finetune.py/mace_lora_finetune.py/
# mace_multihead_finetune.py's foundational_models/MACE/mace-mh-1.model --
# a plain, gitignored, user-visible path. Downloaded here once (see
# build_sampler_model() below) and reused on every later call; you can also
# pre-populate it yourself to skip the download entirely.
FOUNDATION_MODEL_LOCAL_PATH = Path("foundational_models/MACE") / f"{FOUNDATION_MODEL}.model"
# N_MODELS=4 matches the literature paper's choice, but jgreeley has no
# confirmed billing allocation on the `ai` partition (8 GPUs/node) -- only
# `smallgpu` (2 GPUs/node). model_factory_gpu_ids=[0, 1] below therefore
# gives 2x parallelism via cycling (models 0&2 share GPU 0, 1&3 share GPU
# 1), not full 4x -- see runs/workflow_loop_reasonable_defaults/*.run's
# matching --gpus-per-node=2.
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
VALID_FRACTION = 0.15

# batch_size is computed dynamically (see compute_batch_size() below) from
# however many structures are actually in the training pool at the time --
# this script is meant to work across any dataset, not just the small
# real Pt-surface bootstrap it happens to start from here, so a fixed
# constant would be wrong for a much larger or smaller real dataset. Range
# grounded in real MACE/catalysis fine-tuning practice: MACE's own doc
# examples default to batch_size=2, catalysis fine-tuning papers commonly
# use 8-10, and 32 is a standard upper bound before memory becomes the
# limiting factor (arxiv 2605.09394, arxiv 2601.18852).
MIN_BATCH_SIZE = 2
MAX_BATCH_SIZE = 32
TARGET_BATCHES_PER_EPOCH = 4

# Name shared between the ensemble factory (below) and the "find the
# previous iteration's trained model" lookup, so they can't drift apart.
ENSEMBLE_NAME_PREFIX = "pt_surface_finetune_ensemble"

# Which model seeds the MD sampler's calculator and EnsembleUncertainty's
# non-empty constructor check before the fresh committee is trained:
# no trained model exists yet (iteration 0, or a missing/incomplete
# previous checkpoint) -- fall back to the raw foundation model.
# a previous iteration's committee member 0 exists -- use that, since
# it knows more about this specific system than the generic foundation
# model does. See build_sampler_model() below.

# Unphysical-structure filter applied to freshly MD-sampled candidates,
# before the (comparatively expensive) ensemble-inference scoring step.
# max_force is deliberately permissive: MD runs hot (1000 K, see below) on
# purpose to explore beyond near-equilibrium configurations, so legitimately
# useful high-uncertainty candidates can show real, if elevated, forces --
# this threshold is meant to catch true collisions/divergence, not just
# "hotter than equilibrium".
PRE_UNCERTAINTY_CURATION_PIPELINE = ForceCutoffCurate(max_force=50.0)

# Coverage channel picks a structurally diverse subset of low-uncertainty
# leftovers (see ActiveLearningWorkflow._select_for_labeling_with_report,
# which already filters coverage candidates to score < selector.min_score)
# -- ClusterCurate avoids picking several near-duplicate structures from
# the same narrow region. Same real, working setup as
# Pt_surface_test/examples/curation_example.py. species is pure Pt (78),
# matching PARSE_BASE_DIR's real data -- extend for a system with
# adsorbates.
COVERAGE_CURATION_PIPELINE = StructureCurationPipeline(
    cheap_curate=CheapCurate(),
    cluster_curate=ClusterCurate(
        soap_descriptor=SOAPDescriptor(
            species=["Pt"],
            r_cut=6.0,
            n_max=6,
            l_max=4,
            sigma=0.5,
            periodic=True,
            sparse=False,
        ),
        structure_similarity=StructureSimilarity(metric="cosine"),
        similarity_graph=SimilarityGraph(eps=0.005),
        # "energy_mean", not "energy" -- coverage curation runs on already-
        # scored records (EnsembleUncertainty.predict() output merged in),
        # which never carry a plain "energy" field.
        representative_selection=RepresentativeSelection(priority=[("energy_mean", "min")]),
        position_mode="all",
        n_jobs=1,
    ),
    use_cheap=True,
    use_descriptor=False,
    use_cluster=True,
)

# Cap how many labeled structures get used as MD seeds each iteration --
# "all" (workflow_loop.py's choice) means unbounded MD cost growth as the
# train pool grows across iterations. "highest_uncertainty" also focuses
# sampling where the current model is least confident, rather than
# resampling everywhere uniformly. seed_k itself is computed dynamically
# (see compute_seed_k() below) from the actual pool size at the time,
# same reasoning as batch_size -- a fixed constant wouldn't generalize
# across different dataset sizes.
SEED_SELECTION_MODE = "highest_uncertainty"
SEED_FRACTION = 0.25
MIN_SEED_K = 2
MAX_SEED_K = 20

# F_err-equivalent relative score (same expression/band as the
# acselectrochem_5c00540 literature sample): per-atom force disagreement
# divided by that atom's own mean force magnitude, reduced over atoms.
# min_score=0.1 excludes redundant candidates; max_score=0.3 excludes
# candidates so poorly sampled/out-of-distribution that the ensemble's
# disagreement is closer to noise than genuine uncertainty -- these two
# combine with the count cap below (band filter first, then top-K of what
# passes -- see EnsembleUncertaintySelector.select()).
UNCERTAINTY_SCORE_EXPRESSION = "max(force_atomwise_uncertainty / force_magnitude_per_atom)"
UNCERTAINTY_MIN_SCORE = 0.1
UNCERTAINTY_MAX_SCORE = 0.3

# Same dynamic-scaling idea for how many scored candidates get sent to DFT
# each iteration -- resolved against the actual candidate pool size (see
# compute_uncertainty_k() below), not a fixed constant. Kept to a modest
# fraction: small enough to stay adaptive (retrain before committing more
# DFT budget), matching the reasoning discussed for why uncertainty_k
# exists as a cap at all.
UNCERTAINTY_FRACTION = 0.2
MIN_UNCERTAINTY_K = 2
MAX_UNCERTAINTY_K = 15

# Same idea for coverage_k -- resolved against the size of the pool it's
# actually drawing from (the below-min-score leftovers, filtered by
# ActiveLearningWorkflow._select_for_labeling_with_report before this ever
# runs, see select_for_coverage()). Kept smaller than the uncertainty
# channel's range: coverage is meant to supplement primary uncertainty-
# driven selection, not compete with it for DFT budget. No stronger
# (e.g. compute-cost-derived) grounding behind these exact numbers.
COVERAGE_FRACTION = 0.15
MIN_COVERAGE_K = 1
MAX_COVERAGE_K = 8

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


def build_sampler_model(iteration: int) -> MACEModel:
    """No trained model yet -> the raw foundation model. A previous
    iteration's committee exists -> its member 0 (it knows more about this
    specific system than the generic foundation model does)."""
    previous_model_path = (
        MODEL_FACTORY_CHECKPOINT_DIR
        / f"iteration_{iteration - 1:04d}"
        / f"{ENSEMBLE_NAME_PREFIX}_000.model"
    )
    if previous_model_path.exists():
        return MACEModel.load(
            previous_model_path, device=DEVICE, default_dtype="float32", head="Default"
        )

    if not FOUNDATION_MODEL_LOCAL_PATH.exists():
        downloaded_path = download_mace_mp_checkpoint(FOUNDATION_MODEL)
        FOUNDATION_MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(downloaded_path, FOUNDATION_MODEL_LOCAL_PATH)

    sampler_model = MACEModel(model="MACE", device=DEVICE, default_dtype="float32")
    sampler_model.calculator = mace_mp(
        model=str(FOUNDATION_MODEL_LOCAL_PATH), device=DEVICE, default_dtype="float32"
    )
    return sampler_model


def compute_batch_size(train_size: int) -> int:
    """Passed directly as the batch_size/valid_batch_size value in
    model_factory_train_kwargs -- al_dirac calls this itself with the exact
    training pool size right before training starts (see
    ActiveLearningWorkflow's _resolve_train_kwargs_callables), so no
    estimation is needed here. Aims for TARGET_BATCHES_PER_EPOCH gradient
    updates per epoch regardless of pool size, clamped to
    [MIN_BATCH_SIZE, MAX_BATCH_SIZE] (see that constant's comment for where
    that range comes from)."""
    if train_size <= 0:
        return MIN_BATCH_SIZE
    approx_train_only = max(1, round(train_size * (1 - VALID_FRACTION)))
    batch_size = max(MIN_BATCH_SIZE, approx_train_only // TARGET_BATCHES_PER_EPOCH)
    return min(MAX_BATCH_SIZE, batch_size)


def compute_valid_batch_size(train_size: int) -> int:
    """Same idea as compute_batch_size(), but sized off the held-out
    validation portion (train_size * VALID_FRACTION) instead of the
    training portion -- reusing compute_batch_size() directly for both
    would incorrectly apply the (1 - VALID_FRACTION) discount to the
    validation side too."""
    if train_size <= 0:
        return MIN_BATCH_SIZE
    approx_valid_only = max(1, round(train_size * VALID_FRACTION))
    batch_size = max(MIN_BATCH_SIZE, approx_valid_only // TARGET_BATCHES_PER_EPOCH)
    return min(MAX_BATCH_SIZE, batch_size)


def compute_seed_k(pool_size: int) -> int:
    """Passed directly as run_iteration()'s seed_k value -- al_dirac calls
    this with the exact seed-candidate pool size right before seed
    selection. Same reasoning as compute_batch_size(): a fixed seed_k would
    be a huge fraction of a small pool and negligible for a large one."""
    return max(MIN_SEED_K, min(MAX_SEED_K, round(pool_size * SEED_FRACTION)))


def compute_uncertainty_k(pool_size: int) -> int:
    """Passed directly as run_iteration()'s uncertainty_k value -- al_dirac
    calls this with the exact scored candidate pool size right before
    selection-for-labeling, not an estimate (unlike batch_size/seed_k, this
    one genuinely can't be pre-computed: the candidate pool only exists
    after sampling+scoring, which happens after uncertainty_k would need to
    be decided if it weren't a callable)."""
    return max(MIN_UNCERTAINTY_K, min(MAX_UNCERTAINTY_K, round(pool_size * UNCERTAINTY_FRACTION)))


def compute_coverage_k(pool_size: int) -> int:
    """Passed directly as run_iteration()'s coverage_k value -- al_dirac
    calls this inside select_for_coverage() with the size of the pool it's
    actually drawing from (the below-min-score leftovers), so the fraction
    is naturally "% of the low-uncertainty pool", not the full candidate
    pool."""
    return max(MIN_COVERAGE_K, min(MAX_COVERAGE_K, round(pool_size * COVERAGE_FRACTION)))


def run_explore() -> None:
    """One AL iteration: build the sampler/uncertainty/selector/factory,
    run it through selection, then hand the selected records to
    BatchDFTRunner. dft_root_dir/dft_runner=None -- DFT submission is
    handled here as a separate non-blocking SLURM step (matching
    workflow_loop.py), not inside run_iteration() itself. parse_base_dir is
    only passed at iteration 0, since later iterations grow TRAIN_PATH via
    run_resume() instead of re-parsing the original DFT data."""
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
        sampler_model = build_sampler_model(state.iteration)
        uncertainty = EnsembleUncertainty(models=[sampler_model])

        selector = EnsembleUncertaintySelector(
            score_expression=UNCERTAINTY_SCORE_EXPRESSION,
            larger_is_better=True,
            min_score=UNCERTAINTY_MIN_SCORE,
            max_score=UNCERTAINTY_MAX_SCORE,
        )

        factory = ModelEnsembleFactory(
            MACEModel,
            n_models=N_MODELS,
            name_prefix=ENSEMBLE_NAME_PREFIX,
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
            coverage_curation_pipeline=COVERAGE_CURATION_PIPELINE,
        )

        selected_records = workflow.run_iteration(
            state,
            uncertainty_k=compute_uncertainty_k,
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
                "batch_size": compute_batch_size,
                "valid_batch_size": compute_valid_batch_size,
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
            seed_k=compute_seed_k,
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
            uncertainty_stop_key=UNCERTAINTY_SCORE_EXPRESSION,
            uncertainty_stop_expression=UNCERTAINTY_SCORE_EXPRESSION,
            model_error_stop_force_threshold=0.05,
            model_error_stop_energy_threshold=0.005,
            model_error_stop_statistic="max",
            selector=selector,
            coverage_k=compute_coverage_k,
            coverage_curation_pipeline=COVERAGE_CURATION_PIPELINE,
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
    """Collects finished DFT jobs from the batches run_explore() prepared
    and appends the labeled structures to TRAIN_PATH (.aselmdb, matching
    parsed_labeled_records' required format) for the next iteration. Only
    hard-stops if every job in the batch failed -- partial failures still
    let the loop continue with whatever did label successfully."""
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
