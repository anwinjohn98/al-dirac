# al_dirac API Reference

Minimal parameter reference for every class in the package. For runnable examples, see `Pt_surface_test/examples/` and `examples_mock/`.

## Table of contents

- [curator](#curator)
  - [CurationStageReport](#curationstagereport)
  - [CurationResult](#curationresult)
  - [BaseStructureCuration](#basestructurecuration)
  - [CheapCurate](#cheapcurate)
  - [ClusterCurate](#clustercurate)
  - [DescriptorCurate](#descriptorcurate)
  - [ForceCutoffCurate](#forcecutoffcurate)
  - [RandomCurator](#randomcurator)
  - [StructureCurationPipeline](#structurecurationpipeline)
- [dft](#dft)
  - [BaseDFTLabeler](#basedftlabeler)
  - [BatchDFTRunner](#batchdftrunner)
  - [VASPLabeler](#vasplabeler)
- [features](#features)
  - [SOAPDescriptor](#soapdescriptor)
  - [StructureSimilarity](#structuresimilarity)
  - [SimilarityGraph](#similaritygraph)
  - [RepresentativeSelection](#representativeselection)
- [models](#models)
  - [BaseModel](#basemodel)
  - [MACEModel](#macemodel)
  - [ModelFactoryStrategy](#modelfactorystrategy)
  - [SeedStrategy](#seedstrategy)
  - [HyperparameterStrategy](#hyperparameterstrategy)
  - [UserDataSplitStrategy](#userdatasplitstrategy)
  - [RandomDataSplitStrategy](#randomdatasplitstrategy)
  - [ModelEnsembleFactory](#modelensemblefactory)
- [samplers](#samplers)
  - [BaseSampler](#basesampler)
  - [RattleSampler](#rattlesampler)
  - [DynamicsSpec](#dynamicsspec)
  - [MDSampler](#mdsampler)
  - [DimerSampler](#dimersampler)
- [selection](#selection)
  - [BaseSelector](#baseselector)
  - [ScoreExpressionEvaluator](#scoreexpressionevaluator)
  - [EnsembleUncertaintySelector](#ensembleuncertaintyselector)
- [uncertainty](#uncertainty)
  - [BaseUncertainty](#baseuncertainty)
  - [EnsembleUncertainty](#ensembleuncertainty)
- [workflow](#workflow)
  - [WorkflowState](#workflowstate)
  - [StopDecision](#stopdecision)
  - [UncertaintyStoppingCriteria](#uncertaintystoppingcriteria)
  - [ModelErrorStoppingCriteria](#modelerrorstoppingcriteria)
  - [WorkflowLogger](#workflowlogger)
  - [RestartPlan](#restartplan)
  - [restart module-level functions](#restart-module-level-functions)
  - [ActiveLearningWorkflow](#activelearningworkflow)

---

## curator

### CurationStageReport
Dataclass: summary of one curation stage's effect on a record set.
- `stage: str` -- curation method name.
- `input_count: int` -- records in.
- `kept_count: int` -- records kept.
- `removed_count: int` -- records removed.
- `details: dict[str, Any] = {}` -- stage-specific extra info.

### CurationResult
Dataclass: full output of a curation call.
- `kept_records: list[dict]`
- `removed_records: list[dict]`
- `stage_reports: list[CurationStageReport]`

### BaseStructureCuration
Abstract base for all curators. Subclasses implement `curate(structures, **kwargs) -> list[Atoms]`; `curate_records(records, **kwargs) -> CurationResult` is provided by the base class and calls `curate()` under the hood.
- `__init__(method_name: str)`

### CheapCurate
Deduplicates structures via cheap signatures (composition, cell, rounded energy, position match) -- no descriptors.
- `position_decimals: int = 6` -- rounding for exact-match position comparison.
- `cell_decimals: int = 6` -- rounding for cell comparison.
- `energy_decimals: int | None = None` -- power-of-10 energy bin width via `round(x, n)`.
- `energy_bin_width: float | None = None` -- arbitrary energy bin width; takes priority over `energy_decimals`.
- `energy_tolerance: float | None = None` -- max energy difference still considered a duplicate.
- `position_tolerance: float | None = None` -- max per-atom minimum-image displacement still considered a duplicate (replaces exact rounded-position match when set).

### ClusterCurate
Clusters structures by descriptor similarity and keeps one representative per cluster.
- `soap_descriptor: SOAPDescriptor` -- descriptor generator.
- `structure_similarity: StructureSimilarity` -- similarity/distance metric.
- `similarity_graph: SimilarityGraph` -- builds clusters from a distance matrix.
- `representative_selection: RepresentativeSelection` -- picks one representative per cluster.
- `position_mode: str = "all"` -- passed to `SOAPDescriptor.create_local_batch`.
- `n_jobs: int = 1` -- descriptor computation parallelism.

### DescriptorCurate
Removes structures redundant with an already-kept one by descriptor similarity/distance threshold.
- `soap_descriptor: SOAPDescriptor`
- `structure_similarity: StructureSimilarity`
- `similarity_threshold: float | None = None` -- redundant if similarity >= this (at least one of this/`distance_threshold` required).
- `distance_threshold: float | None = None` -- redundant if distance <= this.
- `position_mode: str = "all"`
- `n_jobs: int = 1`

### ForceCutoffCurate
Removes structures whose max atomic force magnitude exceeds a cutoff.
- `max_force: float` -- max allowed per-atom force norm (constraint-adjusted; skipped if no forces available).

### RandomCurator
Randomly subsamples a set of structures/records to a fixed count or fraction.
- `fraction: float | None = None` -- keep `round(n * fraction)`, clamped to `[1, n]`. Exactly one of `fraction`/`k` required.
- `k: int | None = None` -- keep exactly `min(k, n)`.
- `seed: int | None = None` -- RNG seed for reproducibility.

### StructureCurationPipeline
Chains CheapCurate -> DescriptorCurate -> ClusterCurate as one curator.
- `cheap_curate: CheapCurate | None = None`
- `descriptor_curate: DescriptorCurate | None = None`
- `cluster_curate: ClusterCurate | None = None`
- `use_cheap: bool = True` -- run the cheap-curate stage (raises if `True` and `cheap_curate is None`).
- `use_descriptor: bool = False` -- run the descriptor-curate stage.
- `use_cluster: bool = True` -- run the cluster-curate stage.

`curate_records(records, **kwargs) -> CurationResult` runs each enabled stage in order and returns a `CurationResult` whose `stage_reports` contains one entry per enabled sub-stage plus a final aggregate `"structure_curation"` report.

---

## dft

### BaseDFTLabeler
Abstract base for DFT labelers. Subclasses implement:
- `prepare_job(atoms, workdir, **kwargs) -> Path`
- `job_finished(workdir, **kwargs) -> bool`
- `job_succeeded(workdir, **kwargs) -> bool`
- `collect_result(workdir, **kwargs) -> dict`
- `__init__(labeler_name: str)`

### BatchDFTRunner
Batches candidate structures into DFT jobs, submits them, and collects labeled results.
- `labeler: BaseDFTLabeler` -- e.g. `VASPLabeler`.
- `batch_size: int` -- structures per batch (must be positive).
- `max_concurrent_batches: int = 1` -- throttling for non-scheduler (`bash`) submission only.
- `poll_interval_seconds: int = 300` -- polling interval for non-scheduler submission.
- `submission_template: str | Path | None = None` -- required (raises if left `None`); template `.run`/shell script with `__BATCH_DIR__`/`__NUM_JOBS__`/`__JOB_LOOP__` placeholders.
- `submit_command: list[str] | None = None` -- defaults to `["bash"]`; use `["sbatch"]` for SLURM.
- `run_command_template: str | None = None` -- required (raises if left `None`); command run per job inside the batch script (e.g. `"mpirun -np $SLURM_NTASKS vasp_std"`).
- `submit_script_name: str = "submit_batch.sh"` -- filename written per batch.

Key methods:
- `prepare_batches(structures, root_dir, *, records=None, labeler_kwargs=None, template_replacements=None) -> list[dict]` -- writes per-job input files and one submit script per batch; returns batch dicts with `batch_index`, `batch_dir`, `submit_script`, `jobs`.
- `write_submit_script(submit_script, batch_dir, prepared_jobs, template_replacements=None) -> Path`
- `submit_batches(prepared_batches) -> list[str | None]` -- `sbatch` returns real SLURM job IDs; `bash` blocks until batches finish (bounded by `max_concurrent_batches`).
- `collect_results(prepared_batches) -> list[dict]` -- gathers `collect_result()` output for every succeeded job.
- `run(structures, root_dir, *, records=None, labeler_kwargs=None, template_replacements=None) -> list[dict]` -- prepare+submit+collect in one call; raises if `submit_command` is `sbatch` (use prepare/submit/collect separately with a dependency-chained follow-up job instead).

### VASPLabeler
`BaseDFTLabeler` for VASP via ASE's `Vasp` calculator.
- `calculator_kwargs: dict | None = None` -- passed to `ase.calculators.vasp.Vasp(...)`.
- `write_input_kwargs: dict | None = None` -- passed to `calculator.write_input(...)`.
- `output_filename: str = "OUTCAR"` -- file checked for completion/parsed for results.
- `completion_marker: str = "General timing and accounting informations for this job"` -- string marking a finished VASP run.
- `required_outputs: tuple[str, ...] = ("OUTCAR", "vasprun.xml")` -- must all exist and be non-empty for `job_finished()`.

---

## features

### SOAPDescriptor
Wraps `dscribe.descriptors.SOAP`, adapting to different dscribe versions' kwarg names.
- `species: list[str]` -- chemical species present.
- `r_cut: float = 5.0` -- cutoff radius.
- `n_max: int = 8` -- radial basis functions.
- `l_max: int = 6` -- angular basis functions.
- `sigma: float = 0.5` -- Gaussian width.
- `rbf: str = "gto"` -- radial basis function type.
- `periodic: bool = False`
- `crossover: bool = True`
- `sparse: bool = False`

Key methods:
- `create_local(atoms, position_mode="all", positions=None, n_jobs=1, verbose=False, **kwargs) -> np.ndarray`
- `create_local_batch(structures, position_mode="all", positions_list=None, n_jobs=1, verbose=False, **kwargs) -> list[np.ndarray]`

### StructureSimilarity
Computes structure similarity/distance from descriptors via a REMatch kernel.
- `method: str = "rematch"` -- only `"rematch"` supported.
- `alpha: float = 1.0` -- REMatch kernel entropy parameter.
- `threshold: float = 1e-6` -- REMatch convergence threshold.
- `metric: str = "linear"` -- base kernel metric.
- `gamma: float | None = None` -- kernel gamma (for non-linear metrics).
- `normalize_descriptors: bool = True` -- L2-normalize descriptors before comparison.

Key methods:
- `similarity(descriptors_a, descriptors_b) -> float`
- `distance(descriptors_a, descriptors_b) -> float` -- `sqrt(max(0, 2 - 2*similarity))`.
- `similarity_matrix(descriptor_list) -> np.ndarray`
- `distance_matrix(descriptor_list) -> np.ndarray`

### SimilarityGraph
Builds a graph from a distance matrix (edge if distance < `eps`) and finds connected components (clusters).
- `eps: float` -- distance threshold for an edge (must be positive).

Key methods:
- `graph_from_distance_matrix(distance_matrix) -> dict[int, set[int]]`
- `connected_components(graph) -> list[list[int]]`
- `cluster_from_distance_matrix(distance_matrix) -> list[list[int]]` -- both steps combined.

### RepresentativeSelection
Picks one representative record per cluster by a priority-ordered list of fields.
- `priority: list[tuple[str, str]]` -- `(field_name, "min"|"max")` pairs, most important last is applied first via stable sort (i.e. list order = priority order, first entry wins ties).

Key method: `select_representatives(clusters, records) -> list[dict]`.

---

## models

### BaseModel
Abstract base for trainable MLIP wrappers. Subclasses implement:
- `train(train_path, valid_path=None) -> None`
- `predict(atoms) -> dict`
- `save(checkpoint_path) -> None`
- `load(checkpoint_path) -> BaseModel` (classmethod)
- `__init__(model_name: str)`

### MACEModel
`BaseModel` wrapper around MACE's `mace_run_train` CLI (subprocess-based) and `MACECalculator` (in-process inference).

Constructor:
- `name: str = "al_dirac_mace"` -- MACE `--name`; also the base filename for checkpoints/results.
- `model: str = "MACE"` -- MACE `--model` architecture (e.g. `"MACE"`, `"ScaleShiftMACE"`).
- `checkpoint_path: str | Path | None = None` -- if set, builds the calculator immediately (a ready-to-use model).
- `checkpoint_dir: str | Path | None = None` -- directory used to derive `expected_checkpoint_path()` and to locate `results/*.txt` for `latest_validation_metrics()`.
- `device: str = "cpu"` -- `"cpu"` or `"cuda"`.
- `default_dtype: str = "float64"`
- `head: str | None = None` -- MACE head name for multihead models.
- `run_train_script: str | Path | None = None` -- fallback if `mace_run_train` isn't on `PATH`.
- `training_kwargs: dict | None = None` -- default kwargs merged into every `train()` call.

Key methods:
- `expected_checkpoint_path() -> Path | None` -- `checkpoint_path` if set, else `checkpoint_dir / f"{name}.model"`.
- `latest_validation_metrics() -> dict | None` -- reads the last `"mode": "eval"` JSON-lines entry from `checkpoint_dir/results/{name}_run-*_train.txt` (MACE's periodic validation-set eval during training). Keys include `mae_f` (eV/A), `mae_e_per_atom` (eV/atom), `rmse_f`, `rmse_e_per_atom`, `epoch`, `head`, etc.
- `select_replay_dataset(configs_pt, configs_ft, output, model, num_samples, subselect=None, filtering_type=None, atomic_numbers=None, head_pt=None, head_ft=None, weight_pt=None, weight_ft=None) -> Path` -- runs `mace.cli.fine_tuning_select`.
- `train(train_file, valid_file=None, test_file=None, statistics_file=None, work_dir=None, hidden_irreps=None, atomic_numbers=None, config_type_weights=None, energy_key=None, forces_key=None, stress_key=None, E0s=None, batch_size=None, valid_batch_size=None, max_num_epochs=None, valid_fraction=None, r_max=None, lr=None, ema=None, ema_decay=None, loss=None, num_channels=None, max_L=None, num_workers=None, enable_cueq=None, only_cueq=None, distributed=False, launch_mode="single", nproc_per_node=1, cuda_visible_devices=None, seed=None, extra_args=None) -> None` -- trains from scratch. `.aselmdb` train files with no `valid_file` are auto-split (10% by default).
- `finetune(train_file, foundation_model, valid_file=None, test_file=None, statistics_file=None, multiheads_finetuning=False, distributed=False, launch_mode="single", nproc_per_node=1, cuda_visible_devices=None, extra_args=None, **kwargs) -> None` -- fine-tunes a foundation model; `multiheads_finetuning` forced `False` by default (MACE itself defaults it `True`).
- `multihead_finetune(train_file, foundation_model, valid_file=None, test_file=None, pt_train_file=None, pt_valid_file=None, method="direct", selected_pt_output=None, num_samples_pt=None, subselect_pt=None, filter_type_pt=None, atomic_numbers=None, weight_pt=None, weight_ft=None, head_pt=None, head_ft=None, statistics_file=None, force_mh_ft_lr=None, distributed=False, launch_mode="single", nproc_per_node=1, cuda_visible_devices=None, extra_args=None, **kwargs) -> None` -- `method` is `"direct"` (requires `pt_train_file`), `"preprocess"` (subselects a replay set first via `select_replay_dataset`), or `"mp"` (uses MACE's built-in MP replay set).
- `lora_finetune(train_file, foundation_model, valid_file=None, test_file=None, statistics_file=None, lora_rank=None, lora_alpha=None, multiheads_finetuning=False, distributed=False, launch_mode="single", nproc_per_node=1, cuda_visible_devices=None, extra_args=None, **kwargs) -> None` -- LoRA fine-tuning.
- Common to `train`/`finetune`/`multihead_finetune`/`lora_finetune`: `launch_mode` is `"single"` or `"torchrun"` (DDP across `nproc_per_node` GPUs on one node); `cuda_visible_devices` pins the training subprocess to a specific physical GPU (e.g. `"2"`) via env var, independent of `launch_mode`.
- `predict(atoms) -> dict` -- `{"energy", "forces", "stress"}` via the loaded `MACECalculator` (`stress` is `None` if unsupported).
- `save(checkpoint_path) -> None` -- sets `checkpoint_path` and rebuilds the calculator (does not copy files; assumes MACE already wrote the model there).
- `load(checkpoint_path, device="cpu", default_dtype="float64", head=None) -> MACEModel` (classmethod).

### ModelFactoryStrategy
Base class for per-ensemble-member customization strategies (no-op by default).
- `apply(index, kwargs) -> dict` -- modifies a model's constructor kwargs.
- `prepare_training(index, train_kwargs) -> dict` -- modifies a model's per-call training kwargs.

### SeedStrategy
Assigns a distinct RNG seed to each ensemble member.
- `seed_start: int` -- member `i` gets `seed_start + i`.
- `seed_key: str = "seed"` -- kwarg name to set.
- `training_kwargs_key: str = "training_kwargs"` -- nested dict key `apply()` writes into.
- `overwrite: bool = False` -- overwrite an existing seed if already set.

### HyperparameterStrategy
Assigns per-member hyperparameters from an explicit list (e.g. sweeping `lora_rank`).
- `hyperparameters: Sequence[Mapping[str, Any]]` -- one dict per model index, in order (must be non-empty, one entry per model or more).
- `training_kwargs_key: str = "training_kwargs"`
- `overwrite: bool = True` -- overwrite existing keys.

### UserDataSplitStrategy
Assigns explicit, user-provided train/valid file paths per ensemble member.
- `train_paths: Sequence[str | Path]` -- one per model (must be non-empty).
- `valid_paths: Sequence[str | Path | None] | None = None` -- one per model if given; must match `train_paths` length.
- `train_key: str = "train_file"`
- `valid_key: str = "valid_file"`

### RandomDataSplitStrategy
Generates a distinct random train/valid split per ensemble member from one source file.
- `split_root: str | Path` -- directory splits are written under.
- `train_fraction: float = 1.0` -- fraction of the source kept for training (`0 < x <= 1`).
- `valid_fraction: float = 0.1` -- fraction kept for validation (`0 <= x < 1`; `train_fraction + valid_fraction <= 1`).
- `seed_start: int = 0` -- member `i` shuffles with seed `seed_start + i`.
- `file_format: str = "extxyz"` -- format for written split files.
- `dirname_prefix: str = "split"` -- per-member subdirectory name prefix.
- `train_filename: str = "train.extxyz"`
- `valid_filename: str = "valid.extxyz"`
- `start_index: int = 0` -- offset added to the member index in split directory names.
- `width: int = 3` -- zero-padding width for split directory numbering.
- `train_key: str = "train_file"`
- `valid_key: str = "valid_file"`

### ModelEnsembleFactory
Builds and trains a committee of `n_models` model instances of the same class.
- `model_cls: type[ModelT]` -- e.g. `MACEModel`.
- `n_models: int` -- committee size (must be positive).
- `name_prefix: str` -- member `i` is named `f"{name_prefix}_{i:0{width}d}"`.
- `training_method: str = "train"` -- which method on the model to call (`"train"`, `"finetune"`, `"lora_finetune"`, `"multihead_finetune"`).
- `checkpoint_root: str | Path | None = None` -- if set, each member gets its own `checkpoint_root/{name}` subdirectory.
- `base_kwargs: dict | None = None` -- constructor kwargs shared by every member.
- `name_key: str = "name"` -- constructor kwarg name for the model's name.
- `checkpoint_dir_key: str = "checkpoint_dir"` -- constructor kwarg name for its checkpoint directory.
- `checkpoint_dir_extra_arg: str | None = None` -- if set, also appends `[flag, checkpoint_dir]` to `training_kwargs["extra_args"]`.
- `training_kwargs_key: str = "training_kwargs"`
- `extra_args_key: str = "extra_args"`
- `train_key: str = "train_file"` -- kwarg name the training method expects for the train file.
- `valid_key: str = "valid_file"` -- kwarg name the training method expects for the valid file.
- `start_index: int = 0` -- offset added to member indices in generated names.
- `width: int = 3` -- zero-padding width for member numbering.

Key methods:
- `add_strategy(strategy: ModelFactoryStrategy) -> None`
- `build_kwargs() -> list[dict]` -- per-member constructor kwargs after all strategies' `apply()`.
- `build() -> list[ModelT]` -- instantiates every member.
- `train(train_path, *, valid_path=None, train_kwargs=None, load_checkpoints=True, gpu_ids=None) -> list[ModelT]` -- builds, then trains every member (each strategy's `prepare_training()` runs first); if `gpu_ids` is given, all members train **concurrently** in separate threads, each pinned to `gpu_ids[index % len(gpu_ids)]` via `cuda_visible_devices` (raises if `gpu_ids` is an empty sequence); if `gpu_ids` is `None` (default), members train sequentially on the caller's own device. `load_checkpoints=True` re-saves each model from its `expected_checkpoint_path()` once training finishes.

---

## samplers

### BaseSampler
Abstract base for structure samplers. Subclasses implement `sample(atoms, **kwargs) -> list[Atoms]`.
- `__init__(sampler_name: str)`

### RattleSampler
Generates structures by randomly perturbing atomic positions, rejecting samples with atoms too close together.
- `stdev: float = 0.05` -- Gaussian rattle standard deviation (Angstrom).
- `n_samples: int = 10` -- number of accepted samples to generate.
- `seed: int | None = None`
- `min_distance_scale: float = 0.7` -- reject a sample if any neighbor pair is closer than this fraction of their summed natural cutoffs.
- `max_attempts_per_sample: int = 20` -- retries before giving up on one sample slot.

### DynamicsSpec
Dataclass mapping a dynamics name to its ASE module/class for `MDSampler`.
- `module: str`
- `class_name: str`

### MDSampler
Runs short MD trajectories and periodically saves frames as candidate structures. Requires an explicit `calculator`.
- `dynamics_name: str = "Langevin"` -- one of `VelocityVerlet`, `Langevin`, `NoseHooverChainNVT`, `Bussi`, `Andersen`, `NVTBerendsen`, `NPTBerendsen`, `IsotropicMTKNPT`, `MTKNPT`, `MaskedMTKNPT`, `LangevinBAOAB`, `MelchionnaNPT`, `ContourExploration`.
- `timestep_fs: float = 1.0` -- must be positive.
- `steps: int = 1000` -- total MD steps (must be positive).
- `sample_interval: int = 10` -- steps between saved frames (must be positive).
- `initialize_velocities: bool = True` -- draw a Maxwell-Boltzmann velocity distribution before running.
- `velocity_temperature_K: float | None = 300.0` -- required if `initialize_velocities=True`.
- `zero_center_of_mass_momentum: bool = False`
- `dynamics_kwargs: dict | None = None` -- passed to the underlying ASE dynamics class (validated against its `__init__` signature).
- `log_progress: bool = False` -- print step progress.
- `log_interval: int | None = None` -- defaults to `sample_interval` if `log_progress=True`.
- `calculator: Any = None` -- **required** (raises `ValueError` in `sample()` if `None`); attached to a copy of the input atoms, not the caller's own object.

### DimerSampler
Runs a dimer-method saddle-point search/min-mode translation and returns the resulting structure(s). Requires an explicit `calculator`.
- `control_kwargs: dict | None = None` -- passed to `ase.mep.DimerControl(...)`.
- `displace_kwargs: dict | None = None` -- passed to `MinModeAtoms.displace(...)`; a default `[0,0,0.1]` displacement on the last atom is used if `method="vector"` and no vector is given.
- `translate_kwargs: dict | None = None` -- passed to `MinModeTranslate(...)`.
- `run_kwargs: dict | None = None` -- passed to `optimizer.run(...)`.
- `eigenmodes: Any = None` -- initial eigenmode guess for `MinModeAtoms`.
- `random_seed: int | None = None`
- `comm: Any = None` -- MPI communicator for `MinModeAtoms`.
- `collect_trajectory: bool = False` -- if `True`, returns every frame at `sample_interval`; if `False`, returns only the final structure.
- `sample_interval: int = 1` -- must be positive.
- `calculator: Any = None` -- **required** (raises `ValueError` in `sample()` if `None`).

All of the above (except `calculator`/`sample_interval`) can be overridden per-call via matching keys in `sample(atoms, **kwargs)`.

---

## selection

### BaseSelector
Abstract base for candidate selectors. Subclasses implement `select(records, k) -> list[dict]`.
- `__init__(selector_name: str)`

### ScoreExpressionEvaluator
Safe AST-based evaluator for score expressions over record fields (not `eval()` -- a hardcoded whitelist).
- `score_expression: str` -- e.g. `"force_max_uncertainty"` or `"max(force_atomwise_uncertainty / force_magnitude_per_atom)"`.

Supports: field name lookup (scalar or array), `+ - * /`, unary `+ -`, and single-argument calls to `max`, `min`, `mean`, `abs` (mapped to their numpy equivalents). `evaluate(record) -> float | None` returns `None` on any error or if the result isn't reducible to a scalar.

### EnsembleUncertaintySelector
`BaseSelector` that scores and ranks/filters records by a `ScoreExpressionEvaluator` expression.
- `score_expression: str = "force_max_uncertainty"`
- `larger_is_better: bool = True` -- sort direction.
- `min_score: float | None = None` -- drop records scoring below this.
- `max_score: float | None = None` -- drop records scoring above this.

`select(records, k)` returns records sorted by score (best first), truncated to `k` (or all if `k is None`); each kept record gets a `selection_score` field.

---

## uncertainty

### BaseUncertainty
Abstract base for uncertainty estimators. Subclasses implement `predict(atoms) -> dict`.
- `__init__(method_name: str)`

### EnsembleUncertainty
`BaseUncertainty` via committee disagreement across `models`.
- `models: list[BaseModel]` -- must be non-empty.

`predict(atoms)` calls every model's `.predict(atoms)` and returns a dict including: `n_models`, `n_atoms`, `energy_predictions`, `force_predictions`, `stress_predictions`, `energy_mean`, `energy_std`, `energy_rho`, `forces_mean`, `forces_std`, `force_atomwise_uncertainty` (per-atom norm of force std across models), `force_mean_uncertainty`, `force_max_uncertainty`, `force_magnitude_per_atom` (norm of mean force per atom), `force_rho`, `stress_mean`, `stress_std`. Energy/force/stress fields stay `None` if any model returned `None` for that property.

---

## workflow

### WorkflowState
Dataclass tracking one AL run's progress; persisted to/from JSON.
- `run_id: str`
- `iteration: int = 0`
- `status: str = WORKFLOW_STATUS_INITIALIZED`
- `model_name: str | None = None`
- `train_size: int = 0`
- `valid_size: int = 0`
- `test_size: int = 0`
- `pool_size: int = 0`
- `selected_size: int = 0`
- `last_checkpoint: str | None = None`
- `last_completed_stage: str | None = None`
- `last_completed_artifact: str | None = None`
- `best_metric: float | None = None`
- `stop_reason: str | None = None`
- `metadata: dict = {}`

Methods: `to_dict()`, `from_dict(data)` (classmethod), `mark_stage_completed(stage, artifact=None)`.

### StopDecision
Dataclass: the outcome of a stopping-criteria check.
- `should_stop: bool`
- `reason: str | None = None`
- `value: float | None = None`
- `details: dict[str, float] | None = None`

### UncertaintyStoppingCriteria
Stops the loop when a committee-disagreement score across the candidate pool drops below a threshold. Not wired into `ActiveLearningWorkflow` by default (superseded by `ModelErrorStoppingCriteria`); still usable standalone.
- `key: str = "force_max_uncertainty"` -- field name (used if `expression` is `None`).
- `threshold: float | None = None` -- `check()` never stops if `None`.
- `statistic: str = "max"` -- `"max"`, `"mean"`, or `"median"` reduction across the pool.
- `expression: str | None = None` -- overrides `key` with a `ScoreExpressionEvaluator` expression.

`check(records) -> StopDecision`.

### ModelErrorStoppingCriteria
Stops the loop once the worst (by default) ensemble member's own validation-set force and energy error both drop below threshold -- a direct accuracy check, not committee disagreement. This is what `ActiveLearningWorkflow` wires in by default.
- `force_threshold: float | None = 0.05` -- eV/A (50 meV/A default).
- `energy_threshold: float | None = 0.005` -- eV/atom (5 meV/atom default).
- `statistic: str = "max"` -- `"max"`, `"mean"`, or `"median"` reduction across ensemble members.

`check(models) -> StopDecision` -- reads each model's `latest_validation_metrics()` (duck-typed; models without it are skipped); stops only if **both** thresholds are satisfied (AND logic); a threshold that's set but has no data available to check is treated as not-yet-satisfied.

### WorkflowLogger
Writes JSON-lines events, a compact JSON state snapshot, and a human-readable text log.
- `run_dir: str | Path`
- `events_filename: str = DEFAULT_EVENTS_FILE`
- `metrics_filename: str = DEFAULT_METRICS_FILE`
- `text_filename: str = DEFAULT_LOG_FILE`

Key methods:
- `log_event(event, *, state=None, **data) -> None` -- appends to `events.jsonl` only.
- `log_state(state) -> None` -- overwrites `metrics.json` with `compact_state(state)`.
- `log_iteration_start(state)`, `log_iteration_end(state)` -- text + event + state.
- `log_stage_completed(state, stage, *, artifact=None, **data)` -- text + event + state; a `curation_stage_reports` kwarg (list of `CurationStageReport`) is rendered as its own indented lines rather than flattened.
- `log_note(message, *, state=None, **data)` -- a free-form text + event line (used for stop notifications, DFT batch/collection summaries, etc.).
- `log_error(state, error, **data)`

### RestartPlan
Dataclass: where to resume a workflow run from.
- `state: WorkflowState`
- `last_completed_stage: str | None`
- `last_completed_artifact: str | None`
- `next_stage: str | None`
- `last_event: dict | None`

### restart module-level functions
(`al_dirac/workflow/restart.py`, not classes, but used directly by scripts.)
- `stage_artifact_path(artifact_dir, iteration, stage) -> Path`
- `save_artifact(path, value) -> Path` / `load_artifact(path) -> Any` -- pickle-based.
- `save_state(path, state)` / `load_state(path) -> WorkflowState` -- JSON-based.
- `read_events(path) -> list[dict]` -- reads `events.jsonl`.
- `last_completed_event(events, iteration=None) -> dict | None`
- `last_event(events) -> dict | None`
- `next_stage(last_stage, stage_order=DEFAULT_STAGE_ORDER) -> str | None`
- `build_restart_plan(run_dir, *, state_filename=DEFAULT_METRICS_FILE, events_filename=DEFAULT_EVENTS_FILE, stage_order=DEFAULT_STAGE_ORDER) -> RestartPlan`
- `collect_completed_dft_from_batches(prepared_batches, dft_runner) -> list[dict]`
- `incomplete_dft_batches(prepared_batches, dft_runner) -> list[dict]`

`DEFAULT_STAGE_ORDER = ("parsing", "training", "seed_selecting", "sampling", "curating", "scoring", "selecting", "labeling")`.

### ActiveLearningWorkflow
The main orchestrator: runs the full parse -> train -> sample -> curate -> score -> select -> label loop.

**Constructor:**
- `uncertainty: BaseUncertainty` -- required.
- `selector: BaseSelector` -- required.
- `pre_uncertainty_curation_pipeline: StructureCurationPipeline | None = None` -- default curation applied to sampled candidates before scoring.
- `parser_curation_pipeline: StructureCurationPipeline | None = None` -- default curation applied to parsed structures.
- `model: BaseModel | None = None` -- single model trained via `train_model()`.
- `model_factory: ModelEnsembleFactory | Sequence[ModelEnsembleFactory] | None = None` -- committee trainer(s).
- `sampler: BaseSampler | None = None`
- `dft_runner: BatchDFTRunner | None = None`
- `coverage_curation_pipeline: StructureCurationPipeline | None = None` -- default curation for the "coverage" selection channel.
- `uncertainty_curation_pipeline: BaseStructureCuration | None = None` -- default post-selection thinning (e.g. `RandomCurator`) applied to the uncertainty channel.

**`run_iteration(state, uncertainty_k=None, *, ...)`** -- runs one iteration. Parameters:
- `state: WorkflowState` -- required; mutated in place.
- `uncertainty_k: int | None = None` -- number of top-scored candidates to select for labeling (`None` = all that pass the selector's score band).
- `selection_mode: str = "auto"` -- `"auto"` (cold-start at iteration 0 with no training data, else uncertainty), `"cold_start"`, or `"uncertainty"`.
- `cold_start_k: int | None = None` -- how many to pick in cold-start mode.
- `candidate_structures: list[Atoms] | None = None` -- skip sampling and use these directly as the candidate pool.
- `model: BaseModel | None = None` -- overrides the constructor's `model` for this call.
- `train_path: str | Path | None = None` -- training pool file/db; required for training and for uncertainty-mode seed loading.
- `seed_structures: list[Atoms] | None = None` -- explicit seeds for sampling (skips loading from `train_path`).
- `dft_root_dir: str | Path | None = None` -- if set, runs DFT labeling on the selected records; if `None`, the iteration stops after selection.
- `valid_path: str | Path | None = None` -- validation file for training.
- `train_prediction_model: bool = True` -- train the single `model`.
- `train_model_ensemble: bool = True` -- train the `model_factory` committee.
- `train_kwargs: dict | None = None` -- kwargs for `model.train()`.
- `model_factory: ModelEnsembleFactory | Sequence[ModelEnsembleFactory] | None = None` -- overrides the constructor's `model_factory`.
- `model_factory_train_kwargs: dict | Sequence[dict | None] | None = None` -- kwargs for `factory.train()`; a dict is broadcast to every factory, a sequence gives one dict per factory.
- `model_factory_gpu_ids: Sequence[int] | None = None` -- GPU IDs to train ensemble members concurrently on (see `ModelEnsembleFactory.train`).
- `load_model_factory_checkpoints: bool = True`
- `parse_base_dir: str | Path | None = None` -- parse raw DFT outputs from this directory (iteration 0 only).
- `parse_kwargs: dict | None = None`
- `parser_output_paths: Mapping[str, str | Path] | None = None` -- also write parsed records to these additional output files/formats.
- `parser_curation_pipeline: StructureCurationPipeline | None = None` -- overrides the constructor default for this call.
- `parser_curation_kwargs: dict | None = None`
- `seed_selection_mode: str = "all"` -- how to pick seeds from the training pool for sampling.
- `seed_k: int | None = None`
- `seed_selection_curation_pipeline: StructureCurationPipeline | None = None`
- `seed_selection_curation_kwargs: dict | None = None`
- `seed_selection_uncertainty: BaseUncertainty | None = None` -- overrides which uncertainty estimator scores seed candidates.
- `seed_selection_score_key: str = "force_max_uncertainty"`
- `seed_selection_random_seed: int | None = None`
- `sampler: BaseSampler | None = None` -- overrides the constructor's `sampler`.
- `sampler_kwargs: dict | None = None`
- `common_data: dict | None = None` -- merged into every record's top-level/workflow metadata.
- `pre_uncertainty_curation_pipeline: StructureCurationPipeline | None = None` -- overrides the constructor default for this call.
- `pre_uncertainty_curation_kwargs: dict | None = None`
- `uncertainty: BaseUncertainty | None = None` -- overrides the constructor's `uncertainty`.
- `uncertainty_stop_key: str = "force_max_uncertainty"` -- score field summarized (min/mean/max) in the scoring-stage log; not a stopping condition by itself.
- `uncertainty_stop_expression: str | None = None` -- overrides `uncertainty_stop_key` with a `ScoreExpressionEvaluator` expression for that same logging.
- `model_error_stop_force_threshold: float | None = None` -- passed to `ModelErrorStoppingCriteria`; `None` disables the force check.
- `model_error_stop_energy_threshold: float | None = None` -- passed to `ModelErrorStoppingCriteria`; `None` disables the energy check.
- `model_error_stop_statistic: str = "max"` -- passed to `ModelErrorStoppingCriteria`.
- `selector: BaseSelector | None = None` -- overrides the constructor's `selector`.
- `uncertainty_curation_pipeline: BaseStructureCuration | None = None` -- overrides the constructor default for this call.
- `coverage_k: int | None = None` -- how many additional "coverage" records to add alongside the uncertainty selection (`0` disables it).
- `coverage_curation_pipeline: StructureCurationPipeline | None = None`
- `coverage_curation_kwargs: dict | None = None`
- `dft_runner: BatchDFTRunner | None = None` -- overrides the constructor's `dft_runner`.
- `labeler_kwargs: dict | None = None`
- `template_replacements: dict[str, str] | None = None`
- `logger: WorkflowLogger | None = None`
- `artifact_dir: str | Path | None = None` -- required if `restart_from_stage` is used or to enable restart at all.
- `plot_dir: str | Path | None = None`
- `restart_from_stage: str | None = None` -- one of `DEFAULT_STAGE_ORDER`; skips stages before it and loads their checkpointed artifacts instead.

Returns the list of selected (or labeled, if `dft_root_dir` was given) records for that iteration; returns `[]` if a stopping criterion fires or nothing was selected.

**`run(state, *, ...)`** -- identical parameter list to `run_iteration()`, except `candidate_structures`/`seed_structures` are replaced by per-iteration variants:
- `seed_structures_by_iteration: list[list[Atoms]] | None = None`
- `candidate_structures_by_iteration: list[list[Atoms]] | None = None`
- At least one of these or `parse_base_dir` must be given; the number of iterations run = the longest of the provided per-iteration lists (or 1 if only `parse_base_dir` is given).
- `dft_root_dir` (if given) gets an `iteration_XXXX` subdirectory appended automatically per iteration.
- Loops over `run_iteration()` once per iteration, passing every other parameter straight through unchanged each time.
- Returns `list[list[dict]]` -- one records list per iteration, instead of a single list.

**`resume(run_dir, **run_kwargs)`** -- loads a `RestartPlan` from `run_dir` (via `build_restart_plan`) and calls `run()` with `state`, `artifact_dir=run_dir/"artifacts"`, `restart_from_stage` all pre-filled; `plot_dir` defaults to `run_dir/"plots"` unless overridden in `run_kwargs`.

**Other public methods** (mostly used internally by `run_iteration`, but callable directly):
- `train_model(train_path=None, *, valid_path=None, train_kwargs=None, model=None) -> None`
- `generate_candidate_records(seed_records, *, iteration, sampler=None, sampler_kwargs=None, common_data=None) -> list[dict]`
- `curate_parsed_records(records, *, parser_curation_pipeline=None, parser_curation_kwargs=None) -> list[dict]`
- `score_records(records, *, uncertainty=None) -> list[dict]`
- `select_seed_records(records, *, mode="all", k=None, curation_pipeline=None, curation_kwargs=None, uncertainty=None, score_key="force_max_uncertainty", random_seed=None) -> list[dict]`
- `curate_records(records, *, curation_pipeline=None, curation_kwargs=None) -> list[dict]`
- `select_for_labeling(scored_records, uncertainty_k=None, *, selector=None, uncertainty_curation_pipeline=None, coverage_k=None, coverage_curation_pipeline=None, coverage_curation_kwargs=None) -> list[dict]`
- `select_for_coverage(records, coverage_k, *, coverage_curation_pipeline=None, coverage_curation_kwargs=None, require_pipeline=False, channel="coverage") -> list[dict]`
- `label_selected(selected_records, *, root_dir, dft_runner=None, labeler_kwargs=None, template_replacements=None) -> list[dict]`
- `append_records_to_db(db, records, *, split, is_labeled=None, is_selected=None, source="active_learning") -> list[int]`
- `append_selected_to_db(db, selected_records, *, split="selected", source="active_learning") -> list[int]`
