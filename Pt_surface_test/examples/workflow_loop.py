from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ase.io import read

from al_dirac.dft.batch import BatchDFTRunner
from al_dirac.dft.vasp import VASPLabeler
from al_dirac.models.mace_model import MACEModel
from al_dirac.models.model_factory import (
    HyperparameterStrategy,
    ModelEnsembleFactory,
    SeedStrategy,
)
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

RUN_DIR = Path("Pt_surface_test/real_example_outputs/workflow_loop")
ARTIFACT_DIR = RUN_DIR / "artifacts"
PLOT_DIR = RUN_DIR / "plots"
MODEL_FACTORY_CHECKPOINT_DIR = RUN_DIR / "model_factory"
DFT_ROOT_DIR = RUN_DIR / "dft_jobs"
STATE_FILE = RUN_DIR / "state.json"
PREPARED_BATCHES_FILE = RUN_DIR / "prepared_batches.pkl"
# One submit-script path per line -- read by workflow_loop_explore.run (bash,
# on the host) to sbatch each DFT batch itself, since sbatch is not reachable
# from inside the apptainer container.
SUBMIT_SCRIPTS_FILE = RUN_DIR / "submit_scripts.txt"

# Growing training pool -- seeded once from the parsed real structures, then
# appended to with newly DFT-labeled structures after every iteration.
TRAIN_PATH = RUN_DIR / "train.extxyz"
INITIAL_TRAIN_SOURCE = Path("Pt_surface_test/real_example_outputs/parser/train.extxyz")
VALID_FILE = Path("Pt_surface_test/real_example_outputs/model_factory/data/valid.extxyz")
TEST_FILE = Path("Pt_surface_test/real_example_outputs/model_factory/data/test.extxyz")
SEED_SOURCE_FILE = Path("Pt_surface_test/real_example_outputs/parser/train.extxyz")

SUBMISSION_TEMPLATE = Path("Pt_surface_test/examples/vasp_batch.run")

FOUNDATION_MODEL = "medium"
N_MODELS = 3
EPOCHS = 5
BATCH_SIZE_TRAIN = 2
SEED_START = 7
LORA_RANKS = [2, 4, 8]
LORA_ALPHA = 8

SAMPLER_MODEL_PATH = Path(
    "Pt_surface_test/real_example_outputs/mace_finetune/models/pt_surface_finetune.model"
)

DFT_BATCH_SIZE = 3
VASP_CALCULATOR_KWARGS = {
    "xc": "PBE",
    "encut": 400,
    "ediff": 1e-5,
    # Single-point labeling, not relaxation: the whole point of active
    # learning is to label the exact candidate geometry that was queried, not
    # a relaxed one -- ibrion/isif/potim/ediffg are irrelevant with nsw=0.
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


def load_seed_structures() -> list:
    structures = read(SEED_SOURCE_FILE, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    return [structures[0].copy()]


def run_explore() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DFT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMIT_SCRIPTS_FILE.unlink(missing_ok=True)
    logger = WorkflowLogger(run_dir=RUN_DIR)

    if STATE_FILE.exists():
        state = load_state(STATE_FILE)
    else:
        state = WorkflowState(run_id="workflow_loop", iteration=0)
        if not TRAIN_PATH.exists():
            shutil.copy(INITIAL_TRAIN_SOURCE, TRAIN_PATH)

    # Each iteration is a separate process invocation of run_explore(), and
    # MACE writes its trained .model/checkpoints/logs under a single
    # work_dir/checkpoint_dir -- scoping that dir by iteration keeps one
    # iteration's retraining from overwriting the previous iteration's
    # models and appending to its log files.
    iteration_model_factory_dir = (
        MODEL_FACTORY_CHECKPOINT_DIR / f"iteration_{state.iteration:04d}"
    )

    # run_iteration() already checkpoints the selected structures to disk. If
    # a previous explore run got this far but crashed before DFT submission,
    # reuse that instead of retraining/resampling from scratch.
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
            name_prefix="pt_surface_lora_ensemble",
            training_method="lora_finetune",
            base_kwargs={
                "model": "MACE",
                "device": DEVICE,
                "default_dtype": "float32",
                "checkpoint_dir": iteration_model_factory_dir,
            },
        )
        factory.add_strategy(SeedStrategy(seed_start=SEED_START))
        factory.add_strategy(
            HyperparameterStrategy(
                hyperparameters=[{"lora_rank": rank} for rank in LORA_RANKS]
            )
        )

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
            uncertainty_k=3,
            selection_mode="auto",
            cold_start_k=None,
            candidate_structures=None,
            model=None,
            train_path=TRAIN_PATH,
            seed_structures=load_seed_structures(),
            dft_root_dir=None,
            valid_path=VALID_FILE,
            train_prediction_model=False,
            train_model_ensemble=True,
            train_kwargs=None,
            model_factory=factory,
            model_factory_train_kwargs={
                "test_file": TEST_FILE,
                "foundation_model": FOUNDATION_MODEL,
                "lora_alpha": LORA_ALPHA,
                "energy_key": "REF_energy",
                "forces_key": "REF_forces",
                "stress_key": "REF_stress",
                "E0s": "average",
                "batch_size": BATCH_SIZE_TRAIN,
                "valid_batch_size": BATCH_SIZE_TRAIN,
                "max_num_epochs": EPOCHS,
                "lr": 0.0005,
                "ema": True,
                "ema_decay": 0.99,
                "loss": "weighted",
                "num_workers": 0,
                "enable_cueq": True,
                "work_dir": str(iteration_model_factory_dir),
                "extra_args": [
                    "--device", DEVICE,
                    "--default_dtype", "float32",
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
            seed_selection_score_key="force_max_uncertainty",
            seed_selection_random_seed=None,
            sampler=sampler,
            sampler_kwargs=None,
            common_data={"example": "workflow_loop"},
            pre_uncertainty_curation_pipeline=None,
            pre_uncertainty_curation_kwargs=None,
            uncertainty=uncertainty,
            uncertainty_stop_key="force_max_uncertainty",
            uncertainty_stop_threshold=0.001,
            uncertainty_stop_statistic="max",
            uncertainty_stop_expression=None,
            selector=selector,
            coverage_k=0,
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
        print(f"workflow_loop stopped at iteration {state.iteration}: {state.stop_reason}")
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
        write_structure_records(labeled_records, TRAIN_PATH, output_format="extxyz", append=True)

    # run_iteration()'s own _finish_iteration() already bumped state.iteration
    # before explore saved it -- don't bump it again here.
    save_state(STATE_FILE, state)
    print(f"training pool updated, next iteration: {state.iteration}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"explore", "resume"}:
        raise SystemExit("usage: workflow_loop.py [explore|resume]")
    (run_explore if sys.argv[1] == "explore" else run_resume)()


if __name__ == "__main__":
    main()
