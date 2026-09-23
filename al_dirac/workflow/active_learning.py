from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.db.core import Database
from ase.io import read

from al_dirac.constants import (
    WORKFLOW_STATUS_COMPLETE,
    WORKFLOW_STATUS_CURATING,
    WORKFLOW_STATUS_LABELING,
    WORKFLOW_STATUS_PARSING,
    WORKFLOW_STATUS_SAMPLING,
    WORKFLOW_STATUS_SCORING,
    WORKFLOW_STATUS_SELECTING,
    WORKFLOW_STATUS_TRAINING,
)
from al_dirac.curator.base import (
    BaseStructureCuration,
    CurationResult,
    CurationStageReport,
)
from al_dirac.curator.structure_curation import StructureCurationPipeline
from al_dirac.dft.batch import BatchDFTRunner
from al_dirac.models.base import BaseModel
from al_dirac.models.model_factory import ModelEnsembleFactory
from al_dirac.parser.structure_parser import (
    infer_label_properties,
    parse_structure_records,
    write_structure_records,
)
from al_dirac.samplers.base import BaseSampler
from al_dirac.selection.base import BaseSelector
from al_dirac.selection.score_expression import ScoreExpressionEvaluator
from al_dirac.uncertainty.base import BaseUncertainty
from al_dirac.workflow.restart import (
    build_restart_plan,
    load_artifact,
    save_artifact,
    stage_artifact_path,
)
from al_dirac.workflow.outputs import (
    annotate_dft_labels,
    write_iteration_extxyz_outputs,
    write_iteration_plots,
    write_labeling_plots,
    write_records_extxyz,
    write_seed_selection_plot,
)
from al_dirac.workflow.records import (
    build_records,
    get_iteration_value,
    records_from_curation_result,
    split_labeled_records,
    with_workflow_stage,
    workflow_template,
)
from al_dirac.workflow.seed_selection import (
    record_identity,
    select_seed_records as select_seed_records_impl,
)
from al_dirac.workflow.state import WorkflowState
from al_dirac.workflow.stopping import ModelErrorStoppingCriteria
from al_dirac.workflow.train_store import (
    append_labeled_records_to_train_path,
    append_records_to_db,
    append_selected_to_db,
    load_labeled_records_from_train_path,
    train_path_has_data,
)
from al_dirac.workflow.workflow_logger import WorkflowLogger


WORKFLOW_STAGE_SEQUENCE = (
    "parsing",
    "training",
    "seed_selecting",
    "sampling",
    "curating",
    "scoring",
    "selecting",
    "labeling",
)


def _score_statistics(
    records: list[dict[str, Any]],
    *,
    key: str,
    expression: str | None,
) -> dict[str, float] | None:
    evaluator = ScoreExpressionEvaluator(score_expression=expression or key)
    values = [
        value
        for record in records
        if (value := evaluator.evaluate(record)) is not None
    ]
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "score_min": float(np.min(array)),
        "score_mean": float(np.mean(array)),
        "score_max": float(np.max(array)),
    }


def _count_train_records(train_path: str | Path) -> int:
    path = Path(train_path)
    if path.suffix == ".aselmdb":
        return len(connect(str(path)))
    if not path.exists():
        return 0
    structures = read(path, index=":")
    return len(structures) if isinstance(structures, list) else 1


def _resolve_train_kwargs_callables(
    train_kwargs: dict[str, Any] | None,
    *,
    train_size: int,
) -> dict[str, Any] | None:
    # Any value in train_kwargs (e.g. batch_size, valid_batch_size) may be a
    # pure function of the current training pool size instead of a fixed
    # constant, so it can scale with however much data actually exists
    # rather than being tuned for one specific dataset size.
    if train_kwargs is None:
        return None
    return {
        key: (value(train_size) if callable(value) else value)
        for key, value in train_kwargs.items()
    }


class ActiveLearningWorkflow:
    def __init__(
        self,
        uncertainty: BaseUncertainty,
        selector: BaseSelector,
        pre_uncertainty_curation_pipeline: StructureCurationPipeline | None = None,
        parser_curation_pipeline: StructureCurationPipeline | None = None,
        model: BaseModel | None = None,
        model_factory: ModelEnsembleFactory | Sequence[ModelEnsembleFactory] | None = None,
        sampler: BaseSampler | None = None,
        dft_runner: BatchDFTRunner | None = None,
        coverage_curation_pipeline: StructureCurationPipeline | None = None,
        uncertainty_curation_pipeline: BaseStructureCuration | None = None,
    ) -> None:
        self.uncertainty = uncertainty
        self.selector = selector
        self.pre_uncertainty_curation_pipeline = pre_uncertainty_curation_pipeline
        self.parser_curation_pipeline = parser_curation_pipeline
        self.model = model
        self.model_factory = model_factory
        self.sampler = sampler
        self.dft_runner = dft_runner
        self.coverage_curation_pipeline = coverage_curation_pipeline
        self.uncertainty_curation_pipeline = uncertainty_curation_pipeline

    def _checkpoint_records(
        self,
        records: list[dict[str, Any]],
        *,
        state: WorkflowState,
        stage: str,
        artifact_dir: str | Path | None,
        logger: WorkflowLogger | None,
        **data: Any,
    ) -> None:
        artifact_path: Path | None = None
        if artifact_dir is not None:
            artifact_path = stage_artifact_path(
                artifact_dir,
                state.iteration,
                stage,
            )
            save_artifact(artifact_path, records)

        if logger is not None:
            logger.log_stage_completed(
                state,
                stage,
                artifact=artifact_path,
                record_count=len(records),
                **data,
            )

    def _stage_allowed(
        self,
        stage: str,
        restart_from_stage: str | None,
    ) -> bool:
        if restart_from_stage is None:
            return True
        if stage not in WORKFLOW_STAGE_SEQUENCE:
            raise ValueError(f"Unknown workflow stage: {stage}")
        if restart_from_stage not in WORKFLOW_STAGE_SEQUENCE:
            raise ValueError(f"Unknown restart stage: {restart_from_stage}")
        return WORKFLOW_STAGE_SEQUENCE.index(stage) >= WORKFLOW_STAGE_SEQUENCE.index(
            restart_from_stage
        )

    def _load_checkpoint_records(
        self,
        *,
        state: WorkflowState,
        stage: str,
        artifact_dir: str | Path | None,
    ) -> list[dict[str, Any]]:
        if artifact_dir is None:
            raise ValueError("artifact_dir is required to resume from saved stages.")
        path = stage_artifact_path(artifact_dir, state.iteration, stage)
        if not path.exists():
            raise FileNotFoundError(f"Missing restart artifact: {path}")
        return load_artifact(path)

    def _finish_iteration(
        self,
        state: WorkflowState,
        logger: WorkflowLogger | None,
        plot_dir: str | Path | None = None,
    ) -> None:
        state.iteration += 1
        state.status = WORKFLOW_STATUS_COMPLETE
        if logger is not None:
            logger.log_iteration_end(state)
            if plot_dir is not None:
                from al_dirac.plotting.plots import plot_active_learning_progress

                plot_active_learning_progress(logger.events_path, plot_dir)

    def _write_parser_outputs(
        self,
        records: list[dict[str, Any]],
        *,
        state: WorkflowState,
        parser_output_paths: Mapping[str, str | Path] | None,
        train_path: str | Path | None,
        split: str = "train",
        logger: WorkflowLogger | None = None,
    ) -> None:
        if parser_output_paths is None:
            return

        outputs: dict[str, Any] = {}
        train_path_resolved = (
            None if train_path is None else Path(train_path).resolve()
        )
        for output_format, output_path in parser_output_paths.items():
            output_path = Path(output_path)
            if (
                train_path_resolved is not None
                and output_path.resolve() == train_path_resolved
            ):
                outputs[output_format] = {
                    "path": str(output_path),
                    "skipped": "same_as_train_path",
                }
                continue

            result = write_structure_records(
                records,
                output_path,
                output_format=output_format,
                split=split,
                iteration=state.iteration,
                is_labeled=None,
                is_selected=False,
            )
            outputs[output_format] = (
                {"path": str(output_path), "rows": len(result)}
                if isinstance(result, list)
                else {"path": str(result)}
            )

        if logger is not None:
            logger.log_event(
                "parser_outputs_written",
                state=state,
                outputs=outputs,
            )

    @staticmethod
    def _resolve_model_factory_train_kwargs(
        train_kwargs: dict[str, Any] | Sequence[dict[str, Any] | None] | None,
        n_factories: int,
    ) -> list[dict[str, Any] | None]:
        # A plain dict is broadcast to every factory (the single-factory case,
        # unchanged from before); a sequence gives each factory its own
        # kwargs, since heterogeneous backends (e.g. MACE vs. a future UMA
        # wrapper) generally need different training arguments.
        if train_kwargs is None or isinstance(train_kwargs, dict):
            return [train_kwargs] * n_factories

        resolved = list(train_kwargs)
        if len(resolved) != n_factories:
            raise ValueError(
                "model_factory_train_kwargs must be a single dict (applied to "
                "every factory) or a sequence with one dict per model_factory."
            )
        return resolved

    def train_model(
        self,
        train_path: str | Path | None = None,
        *,
        valid_path: str | Path | None = None,
        train_kwargs: dict[str, Any] | None = None,
        model: BaseModel | None = None,
    ) -> None:
        model = self.model if model is None else model
        if model is None:
            raise RuntimeError("model is required to train inside the workflow.")

        if train_path is None:
            raise ValueError("train_path is required for model training.")

        kwargs = {} if train_kwargs is None else dict(train_kwargs)
        model.train(train_path, valid_path, **kwargs)

    def generate_candidate_records(
        self,
        seed_records: list[dict[str, Any]],
        *,
        iteration: int,
        sampler: BaseSampler | None = None,
        sampler_kwargs: dict[str, Any] | None = None,
        common_data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        sampler = self.sampler if sampler is None else sampler
        if sampler is None:
            raise RuntimeError("sampler is required to generate candidates.")

        kwargs = {} if sampler_kwargs is None else dict(sampler_kwargs)
        candidate_records: list[dict[str, Any]] = []
        for seed_index, seed_record in enumerate(seed_records):
            sampled = sampler.sample(seed_record["atoms"], **kwargs)
            sampled_structures = [sampled] if isinstance(sampled, Atoms) else list(sampled)
            seed_id = seed_record.get(
                "structure_id",
                f"seed_{iteration}_{seed_index}",
            )
            for local_index, atoms in enumerate(sampled_structures):
                candidate_index = len(candidate_records)
                record = {
                    **(common_data or {}),
                    "candidate_index": candidate_index,
                    "structure_id": f"candidate_{iteration}_{candidate_index}",
                    "atoms": atoms,
                    "iteration": iteration,
                    "parent_structure_id": seed_id,
                    "workflow": workflow_template(
                        {
                            **((common_data or {}).get("workflow", {})),
                            "sampler": {
                                "sampler": sampler.sampler_name,
                                "sampler_kwargs": kwargs,
                                "seed_structure_id": seed_id,
                                "seed_index": seed_index,
                                "sample_index": local_index,
                            },
                        }
                    ),
                }
                candidate_records.append(record)

        return candidate_records

    def curate_parsed_records(
        self,
        records: list[dict[str, Any]],
        *,
        parser_curation_pipeline: StructureCurationPipeline | None = None,
        parser_curation_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        pipeline = (
            self.parser_curation_pipeline
            if parser_curation_pipeline is None
            else parser_curation_pipeline
        )
        return self.curate_records(
            records,
            curation_pipeline=pipeline,
            curation_kwargs=parser_curation_kwargs,
        )

    def _resolve_selection_mode(
        self,
        selection_mode: str,
        *,
        state: WorkflowState,
        train_path: str | Path | None,
        parsed_labeled_records: list[dict[str, Any]],
    ) -> str:
        if selection_mode not in {"auto", "cold_start", "uncertainty"}:
            raise ValueError(
                "selection_mode must be 'auto', 'cold_start', or 'uncertainty'."
            )
        if selection_mode != "auto":
            return selection_mode

        has_training_data = bool(parsed_labeled_records) or train_path_has_data(
            train_path
        )
        return (
            "cold_start"
            if state.iteration == 0 and not has_training_data
            else "uncertainty"
        )

    def score_records(
        self,
        records: list[dict[str, Any]],
        *,
        uncertainty: BaseUncertainty | None = None,
    ) -> list[dict[str, Any]]:
        uncertainty = self.uncertainty if uncertainty is None else uncertainty
        scored_records: list[dict[str, Any]] = []
        for record in records:
            uncertainty_result = uncertainty.predict(record["atoms"])
            updated_record = dict(record)
            updated_record.update(uncertainty_result)
            workflow = workflow_template(updated_record.get("workflow"))
            uncertainty_models = getattr(uncertainty, "models", None)
            workflow["uncertainty"] = {
                "method": uncertainty.method_name,
                "class": type(uncertainty).__name__,
                "score_keys": sorted(uncertainty_result.keys()),
                "n_models": (
                    None if uncertainty_models is None else len(uncertainty_models)
                ),
            }
            updated_record["workflow"] = workflow
            scored_records.append(updated_record)
        return scored_records

    def select_seed_records(
        self,
        records: list[dict[str, Any]],
        *,
        mode: str = "all",
        k: int | Callable[[int], int] | None = None,
        curation_pipeline: StructureCurationPipeline | None = None,
        curation_kwargs: dict[str, Any] | None = None,
        uncertainty: BaseUncertainty | None = None,
        score_key: str = "force_max_uncertainty",
        random_seed: int | None = None,
    ) -> list[dict[str, Any]]:
        return select_seed_records_impl(
            records,
            mode=mode,
            k=k,
            curation_pipeline=curation_pipeline,
            curation_kwargs=curation_kwargs,
            uncertainty=uncertainty,
            score_key=score_key,
            random_seed=random_seed,
            score_records=self.score_records,
            curate_records=self.curate_records,
        )

    def curate_records(
        self,
        records: list[dict[str, Any]],
        *,
        curation_pipeline: StructureCurationPipeline | None = None,
        curation_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if curation_pipeline is None:
            return records

        result = curation_pipeline.curate_records(
            records,
            **(curation_kwargs or {}),
        )
        return records_from_curation_result(result)

    def _curate_records_with_result(
        self,
        records: list[dict[str, Any]],
        *,
        curation_pipeline: StructureCurationPipeline | None = None,
        curation_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], CurationResult | None]:
        if curation_pipeline is None:
            return records, None

        result = curation_pipeline.curate_records(
            records,
            **(curation_kwargs or {}),
        )
        return records_from_curation_result(result), result

    def select_for_labeling(
        self,
        scored_records: list[dict[str, Any]],
        uncertainty_k: int | Callable[[int], int] | None = None,
        *,
        selector: BaseSelector | None = None,
        uncertainty_curation_pipeline: BaseStructureCuration | None = None,
        coverage_k: int | Callable[[int], int] | None = None,
        coverage_curation_pipeline: StructureCurationPipeline | None = None,
        coverage_curation_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        merged, _ = self._select_for_labeling_with_report(
            scored_records,
            uncertainty_k,
            selector=selector,
            uncertainty_curation_pipeline=uncertainty_curation_pipeline,
            coverage_k=coverage_k,
            coverage_curation_pipeline=coverage_curation_pipeline,
            coverage_curation_kwargs=coverage_curation_kwargs,
        )
        return merged

    def _select_for_labeling_with_report(
        self,
        scored_records: list[dict[str, Any]],
        uncertainty_k: int | Callable[[int], int] | None = None,
        *,
        selector: BaseSelector | None = None,
        uncertainty_curation_pipeline: BaseStructureCuration | None = None,
        coverage_k: int | Callable[[int], int] | None = None,
        coverage_curation_pipeline: StructureCurationPipeline | None = None,
        coverage_curation_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], CurationStageReport | None]:
        # uncertainty_k may be a plain int/None, or a pure function of the
        # scored candidate pool size -- resolved here (not inside the
        # selector) with the pool size actually available at this point in
        # the workflow, so it can scale with however many candidates exist
        # this iteration instead of a fixed constant.
        if callable(uncertainty_k):
            uncertainty_k = uncertainty_k(len(scored_records))
        if uncertainty_k is not None and uncertainty_k < 0:
            raise ValueError("uncertainty_k must be non-negative or None.")
        # coverage_k resolved later, inside select_for_coverage() itself
        # (with the post-min-score-filter pool size) -- just skip this
        # early check for a callable, select_for_coverage() validates the
        # resolved value itself.
        if coverage_k is not None and not callable(coverage_k) and coverage_k < 0:
            raise ValueError("coverage_k must be non-negative or None.")

        selector = self.selector if selector is None else selector
        uncertainty_records = selector.select(scored_records, uncertainty_k)
        uncertainty_records = [
            with_workflow_stage(
                record,
                "selection",
                {
                    "channel": "uncertainty",
                    "selector": selector.selector_name,
                    "selector_class": type(selector).__name__,
                    "k": uncertainty_k,
                    "selection_score": record.get("selection_score"),
                },
            )
            for record in uncertainty_records
        ]

        curation_pipeline = (
            self.uncertainty_curation_pipeline
            if uncertainty_curation_pipeline is None
            else uncertainty_curation_pipeline
        )
        uncertainty_curation_report: CurationStageReport | None = None
        if curation_pipeline is not None and uncertainty_records:
            curation_result = curation_pipeline.curate_records(uncertainty_records)
            uncertainty_records = curation_result.kept_records
            uncertainty_curation_report = curation_result.stage_reports[0]

        selected_ids = {record_identity(record) for record in uncertainty_records}
        remaining_records = [
            record
            for record in scored_records
            if record_identity(record) not in selected_ids
        ]

        # Coverage specifically targets records the selector is confidently
        # sure about (score below its own min_score) -- an ensemble can be
        # falsely confident (low disagreement) on out-of-distribution
        # structures if all members share the same blind spot, so these are
        # worth diverse coverage too, not an arbitrary slice of leftovers.
        # No-ops (keeps remaining_records as-is) for any selector without
        # these attributes.
        coverage_min_score = getattr(selector, "min_score", None)
        coverage_score_expression = getattr(selector, "score_expression", None)
        if coverage_min_score is not None and coverage_score_expression is not None:
            evaluator = ScoreExpressionEvaluator(score_expression=coverage_score_expression)
            remaining_records = [
                record
                for record in remaining_records
                if (score := evaluator.evaluate(record)) is not None
                and score < coverage_min_score
            ]

        coverage_records: list[dict[str, Any]] = []
        if coverage_k != 0 and remaining_records:
            coverage_records = self.select_for_coverage(
                remaining_records,
                coverage_k,
                coverage_curation_pipeline=coverage_curation_pipeline,
                coverage_curation_kwargs=coverage_curation_kwargs,
                channel="coverage",
            )

        merged: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for record in [*uncertainty_records, *coverage_records]:
            record_id = record_identity(record)
            if record_id in seen_ids:
                continue
            merged.append(record)
            seen_ids.add(record_id)

        return merged, uncertainty_curation_report

    def select_for_coverage(
        self,
        records: list[dict[str, Any]],
        coverage_k: int | Callable[[int], int] | None,
        *,
        coverage_curation_pipeline: StructureCurationPipeline | None = None,
        coverage_curation_kwargs: dict[str, Any] | None = None,
        require_pipeline: bool = False,
        channel: str = "coverage",
    ) -> list[dict[str, Any]]:
        # coverage_k may be a plain int/None, or a pure function of the
        # records pool size -- resolved here with whatever pool this call
        # actually received (e.g. _select_for_labeling_with_report already
        # filters this to the below-min-score leftover pool before calling
        # here, so a fraction is naturally "% of the low-uncertainty pool").
        if callable(coverage_k):
            coverage_k = coverage_k(len(records))
        if coverage_k is not None and coverage_k < 0:
            raise ValueError("coverage_k must be non-negative or None.")
        if coverage_k == 0 or not records:
            return []

        pipeline = coverage_curation_pipeline or self.coverage_curation_pipeline
        if require_pipeline and pipeline is None:
            raise ValueError(
                "Cold-start selection requires a coverage_curation_pipeline."
            )
        if pipeline is not None:
            records = self.curate_records(
                records,
                curation_pipeline=pipeline,
                curation_kwargs=coverage_curation_kwargs,
            )
        selected_records = records if coverage_k is None else records[:coverage_k]
        return [
            with_workflow_stage(
                record,
                "selection",
                {
                    "channel": channel,
                    "k": coverage_k,
                    "coverage_pipeline": (
                        None if pipeline is None else pipeline.method_name
                    ),
                },
            )
            for record in selected_records
        ]

    def label_selected(
        self,
        selected_records: list[dict[str, Any]],
        *,
        root_dir: str | Path,
        dft_runner: BatchDFTRunner | None = None,
        labeler_kwargs: dict[str, Any] | None = None,
        template_replacements: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        dft_runner = self.dft_runner if dft_runner is None else dft_runner
        if dft_runner is None:
            raise RuntimeError("dft_runner is required to label selected structures.")

        structures = [record["atoms"] for record in selected_records]
        labeled_records = dft_runner.run(
            structures=structures,
            root_dir=root_dir,
            records=selected_records,
            labeler_kwargs=labeler_kwargs,
            template_replacements=template_replacements,
        )
        for record in labeled_records:
            is_labeled, label_properties = infer_label_properties(record["atoms"])
            record["is_labeled"] = is_labeled
            record["label_properties"] = label_properties
            workflow = workflow_template(record.get("workflow"))
            dft_labeling = dict(workflow.get("dft_labeling") or {})
            dft_labeling["label_inferred"] = is_labeled
            dft_labeling["label_properties"] = label_properties
            workflow["dft_labeling"] = dft_labeling
            record["workflow"] = workflow
        return labeled_records

    def run_iteration(
        self,
        state: WorkflowState,
        uncertainty_k: int | Callable[[int], int] | None = None,
        *,
        selection_mode: str = "auto",
        cold_start_k: int | None = None,
        candidate_structures: list[Atoms] | None = None,
        model: BaseModel | None = None,
        train_path: str | Path | None = None,
        seed_structures: list[Atoms] | None = None,
        dft_root_dir: str | Path | None = None,
        valid_path: str | Path | None = None,
        train_prediction_model: bool = True,
        train_model_ensemble: bool = True,
        train_kwargs: dict[str, Any] | None = None,
        model_factory: ModelEnsembleFactory | Sequence[ModelEnsembleFactory] | None = None,
        model_factory_train_kwargs: (
            dict[str, Any] | Sequence[dict[str, Any] | None] | None
        ) = None,
        model_factory_gpu_ids: Sequence[int] | None = None,
        load_model_factory_checkpoints: bool = True,
        parse_base_dir: str | Path | None = None,
        parse_kwargs: dict[str, Any] | None = None,
        parser_output_paths: Mapping[str, str | Path] | None = None,
        parser_curation_pipeline: StructureCurationPipeline | None = None,
        parser_curation_kwargs: dict[str, Any] | None = None,
        seed_selection_mode: str = "all",
        seed_k: int | Callable[[int], int] | None = None,
        seed_selection_curation_pipeline: StructureCurationPipeline | None = None,
        seed_selection_curation_kwargs: dict[str, Any] | None = None,
        seed_selection_uncertainty: BaseUncertainty | None = None,
        seed_selection_score_key: str = "force_max_uncertainty",
        seed_selection_random_seed: int | None = None,
        sampler: BaseSampler | None = None,
        sampler_kwargs: dict[str, Any] | None = None,
        common_data: dict[str, Any] | None = None,
        pre_uncertainty_curation_pipeline: StructureCurationPipeline | None = None,
        pre_uncertainty_curation_kwargs: dict[str, Any] | None = None,
        uncertainty: BaseUncertainty | None = None,
        uncertainty_stop_key: str = "force_max_uncertainty",
        uncertainty_stop_expression: str | None = None,
        model_error_stop_force_threshold: float | None = None,
        model_error_stop_energy_threshold: float | None = None,
        model_error_stop_statistic: str = "max",
        selector: BaseSelector | None = None,
        uncertainty_curation_pipeline: BaseStructureCuration | None = None,
        coverage_k: int | Callable[[int], int] | None = None,
        coverage_curation_pipeline: StructureCurationPipeline | None = None,
        coverage_curation_kwargs: dict[str, Any] | None = None,
        dft_runner: BatchDFTRunner | None = None,
        labeler_kwargs: dict[str, Any] | None = None,
        template_replacements: dict[str, str] | None = None,
        logger: WorkflowLogger | None = None,
        artifact_dir: str | Path | None = None,
        plot_dir: str | Path | None = None,
        restart_from_stage: str | None = None,
    ) -> list[dict[str, Any]]:
        if logger is not None:
            logger.log_iteration_start(state)

        state.pool_size = 0
        state.selected_size = 0
        state.metadata["raw_pool_size"] = 0
        state.metadata["parsed_labeled_count"] = 0
        state.metadata["parsed_invalid_count"] = 0

        parsed_labeled_records: list[dict[str, Any]] = []
        curation_result: CurationResult | None = None
        if parse_base_dir is not None and state.iteration != 0:
            raise ValueError("parse_base_dir is only allowed at iteration 0.")

        if parse_base_dir is not None and self._stage_allowed(
            "parsing",
            restart_from_stage,
        ):
            state.status = WORKFLOW_STATUS_PARSING
            parsed_records = parse_structure_records(
                parse_base_dir,
                iteration=state.iteration,
                common_data=common_data,
                **(parse_kwargs or {}),
            )
            if (
                parser_curation_pipeline is not None
                or self.parser_curation_pipeline is not None
            ):
                state.status = WORKFLOW_STATUS_CURATING
            parsed_records = self.curate_parsed_records(
                parsed_records,
                parser_curation_pipeline=parser_curation_pipeline,
                parser_curation_kwargs=parser_curation_kwargs,
            )
            (
                parsed_labeled_records,
                parsed_invalid_records,
            ) = split_labeled_records(parsed_records)
            self._checkpoint_records(
                parsed_records,
                state=state,
                stage="parsing",
                artifact_dir=artifact_dir,
                logger=logger,
                labeled_count=len(parsed_labeled_records),
                invalid_count=len(parsed_invalid_records),
            )
            state.metadata["parsed_labeled_count"] = len(parsed_labeled_records)
            state.metadata["parsed_invalid_count"] = len(parsed_invalid_records)

            if parsed_labeled_records:
                if train_path is None:
                    raise ValueError(
                        "Parsed labeled structures require an '.aselmdb' train_path "
                        "so they can be used for training."
                    )
                append_labeled_records_to_train_path(
                    train_path,
                    parsed_labeled_records,
                )
                state.train_size = _count_train_records(train_path)

            self._write_parser_outputs(
                parsed_records,
                state=state,
                parser_output_paths=parser_output_paths,
                train_path=train_path,
                split="train",
                logger=logger,
            )
        elif parse_base_dir is not None:
            parsed_records = self._load_checkpoint_records(
                state=state,
                stage="parsing",
                artifact_dir=artifact_dir,
            )
            (
                parsed_labeled_records,
                parsed_invalid_records,
            ) = split_labeled_records(parsed_records)
            state.metadata["parsed_labeled_count"] = len(parsed_labeled_records)
            state.metadata["parsed_invalid_count"] = len(parsed_invalid_records)

        resolved_selection_mode = self._resolve_selection_mode(
            selection_mode,
            state=state,
            train_path=train_path,
            parsed_labeled_records=parsed_labeled_records,
        )
        state.metadata["selection_mode"] = resolved_selection_mode
        if (
            resolved_selection_mode == "cold_start"
            and dft_root_dir is not None
            and train_path is None
        ):
            raise ValueError(
                "Cold-start DFT labeling requires an '.aselmdb' train_path so "
                "new labels can be used in the next iteration."
            )

        if (
            self._stage_allowed("training", restart_from_stage)
            and resolved_selection_mode != "cold_start"
            and train_path is not None
            and (train_prediction_model or train_model_ensemble)
        ):
            state.status = WORKFLOW_STATUS_TRAINING
            trained_any_model = False
            trained_models: list[BaseModel] = []
            workflow_model = self.model if model is None else model
            if train_prediction_model and workflow_model is not None:
                self.train_model(
                    train_path=train_path,
                    valid_path=valid_path,
                    train_kwargs=train_kwargs,
                    model=workflow_model,
                )
                trained_any_model = True

            if train_model_ensemble:
                workflow_factory = self.model_factory if model_factory is None else model_factory
                if workflow_factory is not None:
                    factories = (
                        [workflow_factory]
                        if isinstance(workflow_factory, ModelEnsembleFactory)
                        else list(workflow_factory)
                    )
                    factory_train_kwargs = self._resolve_model_factory_train_kwargs(
                        model_factory_train_kwargs, len(factories)
                    )
                    current_train_size = _count_train_records(train_path)
                    for factory, factory_kwargs in zip(factories, factory_train_kwargs):
                        trained_models.extend(
                            factory.train(
                                train_path,
                                valid_path=valid_path,
                                train_kwargs=_resolve_train_kwargs_callables(
                                    factory_kwargs, train_size=current_train_size
                                ),
                                load_checkpoints=load_model_factory_checkpoints,
                                gpu_ids=model_factory_gpu_ids,
                            )
                        )
                    workflow_uncertainty = self.uncertainty if uncertainty is None else uncertainty
                    if not hasattr(workflow_uncertainty, "models"):
                        raise TypeError(
                            "train_model_ensemble=True requires an uncertainty object with a models attribute."
                        )
                    workflow_uncertainty.models = trained_models
                    trained_any_model = True

            if not trained_any_model:
                raise RuntimeError(
                    "train_path was provided, but no model or model_factory was available for training."
                )
            state.train_size = _count_train_records(train_path)
            if logger is not None:
                logger.log_stage_completed(
                    state,
                    "training",
                    artifact=train_path,
                    train_size=state.train_size,
                )

            model_error_stop_decision = ModelErrorStoppingCriteria(
                force_threshold=model_error_stop_force_threshold,
                energy_threshold=model_error_stop_energy_threshold,
                statistic=model_error_stop_statistic,
            ).check(trained_models)
            if model_error_stop_decision.should_stop:
                state.stop_reason = model_error_stop_decision.reason
                if logger is not None:
                    logger.log_event(
                        "stopping_criteria_met",
                        state=state,
                        reason=model_error_stop_decision.reason,
                        details=model_error_stop_decision.details,
                        force_threshold=model_error_stop_force_threshold,
                        energy_threshold=model_error_stop_energy_threshold,
                        statistic=model_error_stop_statistic,
                    )
                    logger.log_note(
                        f"stopping criteria met: {model_error_stop_decision.reason}",
                        state=state,
                    )
                self._finish_iteration(state, logger, plot_dir)
                return []

        if self._stage_allowed("sampling", restart_from_stage):
            if candidate_structures is None:
                seed_records: list[dict[str, Any]] = []
                if seed_structures is not None:
                    seed_records = build_records(
                        seed_structures,
                        iteration=state.iteration,
                        common_data=common_data,
                    )
                elif parsed_labeled_records:
                    seed_records = parsed_labeled_records
                elif train_path is not None:
                    seed_records = load_labeled_records_from_train_path(
                        train_path,
                        iteration=state.iteration,
                        common_data=common_data,
                    )

                if seed_records:
                    if self._stage_allowed("seed_selecting", restart_from_stage):
                        # Captured before select_seed_records() reassigns
                        # seed_records to its (smaller) output below -- needed
                        # to re-resolve a callable seed_k for logging with the
                        # same input size it was actually resolved against.
                        seed_pool_size = len(seed_records)
                        seed_records = self.select_seed_records(
                            seed_records,
                            mode=seed_selection_mode,
                            k=seed_k,
                            curation_pipeline=seed_selection_curation_pipeline,
                            curation_kwargs=seed_selection_curation_kwargs,
                            uncertainty=(
                                seed_selection_uncertainty
                                if seed_selection_uncertainty is not None
                                else (
                                    self.uncertainty
                                    if uncertainty is None
                                    else uncertainty
                                )
                            ),
                            score_key=seed_selection_score_key,
                            random_seed=seed_selection_random_seed,
                        )
                        resolved_seed_k = (
                            seed_k(seed_pool_size) if callable(seed_k) else seed_k
                        )
                        self._checkpoint_records(
                            seed_records,
                            state=state,
                            stage="seed_selecting",
                            artifact_dir=artifact_dir,
                            logger=logger,
                            selection_mode=seed_selection_mode,
                            seed_k=resolved_seed_k,
                        )
                        write_seed_selection_plot(
                            seed_records,
                            state=state,
                            plot_dir=plot_dir,
                            logger=logger,
                        )
                    else:
                        seed_records = self._load_checkpoint_records(
                            state=state,
                            stage="seed_selecting",
                            artifact_dir=artifact_dir,
                        )
                    state.status = WORKFLOW_STATUS_SAMPLING
                    candidate_records = self.generate_candidate_records(
                        seed_records,
                        iteration=state.iteration,
                        sampler=sampler,
                        sampler_kwargs=sampler_kwargs,
                        common_data=common_data,
                    )
                    self._checkpoint_records(
                        candidate_records,
                        state=state,
                        stage="sampling",
                        artifact_dir=artifact_dir,
                        logger=logger,
                    )
                else:
                    candidate_records = []
            else:
                candidate_records = build_records(
                    candidate_structures,
                    iteration=state.iteration,
                    common_data=common_data,
                )
                self._checkpoint_records(
                    candidate_records,
                    state=state,
                    stage="sampling",
                    artifact_dir=artifact_dir,
                    logger=logger,
                )
        else:
            candidate_records = self._load_checkpoint_records(
                state=state,
                stage="sampling",
                artifact_dir=artifact_dir,
            )

        state.metadata["raw_pool_size"] = len(candidate_records)
        if not candidate_records:
            raise ValueError(
                "Provide candidate_structures or seed_structures; parsed labeled "
                "structures require a sampler to generate candidates."
            )

        pipeline = (
            self.pre_uncertainty_curation_pipeline
            if pre_uncertainty_curation_pipeline is None
            else pre_uncertainty_curation_pipeline
        )
        if pipeline is not None:
            if self._stage_allowed("curating", restart_from_stage):
                state.status = WORKFLOW_STATUS_CURATING
                candidate_records, curation_result = self._curate_records_with_result(
                    candidate_records,
                    curation_pipeline=pipeline,
                    curation_kwargs=pre_uncertainty_curation_kwargs,
                )
                self._checkpoint_records(
                    candidate_records,
                    state=state,
                    stage="curating",
                    artifact_dir=artifact_dir,
                    logger=logger,
                    curation_stage_reports=(
                        curation_result.stage_reports
                        if curation_result is not None
                        else None
                    ),
                )
            else:
                candidate_records = self._load_checkpoint_records(
                    state=state,
                    stage="curating",
                    artifact_dir=artifact_dir,
                )
        state.pool_size = len(candidate_records)
        if state.pool_size == 0:
            raise ValueError("Pre-uncertainty curation removed every candidate.")

        state.status = WORKFLOW_STATUS_SELECTING
        if resolved_selection_mode == "cold_start":
            if self._stage_allowed("selecting", restart_from_stage):
                selected_records = self.select_for_coverage(
                    candidate_records,
                    cold_start_k,
                    coverage_curation_pipeline=coverage_curation_pipeline,
                    coverage_curation_kwargs=coverage_curation_kwargs,
                    require_pipeline=True,
                    channel="cold_start",
                )
            else:
                selected_records = self._load_checkpoint_records(
                    state=state,
                    stage="selecting",
                    artifact_dir=artifact_dir,
                )
        else:
            if self._stage_allowed("scoring", restart_from_stage):
                state.status = WORKFLOW_STATUS_SCORING
                scored_records = self.score_records(
                    candidate_records,
                    uncertainty=uncertainty,
                )
                score_stats = _score_statistics(
                    scored_records,
                    key=uncertainty_stop_key,
                    expression=uncertainty_stop_expression,
                )
                self._checkpoint_records(
                    scored_records,
                    state=state,
                    stage="scoring",
                    artifact_dir=artifact_dir,
                    logger=logger,
                    **(score_stats or {}),
                )
            else:
                scored_records = self._load_checkpoint_records(
                    state=state,
                    stage="scoring",
                    artifact_dir=artifact_dir,
                )
            uncertainty_curation_report: CurationStageReport | None = None
            if self._stage_allowed("selecting", restart_from_stage):
                state.status = WORKFLOW_STATUS_SELECTING
                selected_records, uncertainty_curation_report = (
                    self._select_for_labeling_with_report(
                        scored_records,
                        uncertainty_k=uncertainty_k,
                        selector=selector,
                        uncertainty_curation_pipeline=uncertainty_curation_pipeline,
                        coverage_k=coverage_k,
                        coverage_curation_pipeline=coverage_curation_pipeline,
                        coverage_curation_kwargs=coverage_curation_kwargs,
                    )
                )
            else:
                selected_records = self._load_checkpoint_records(
                    state=state,
                    stage="selecting",
                    artifact_dir=artifact_dir,
                )
        if self._stage_allowed("selecting", restart_from_stage):
            uncertainty_curation_data: dict[str, Any] = {}
            if uncertainty_curation_report is not None:
                uncertainty_curation_data = {
                    "uncertainty_curation_input_count": (
                        uncertainty_curation_report.input_count
                    ),
                    "uncertainty_curation_kept_count": (
                        uncertainty_curation_report.kept_count
                    ),
                    "uncertainty_curation_removed_count": (
                        uncertainty_curation_report.removed_count
                    ),
                }
            self._checkpoint_records(
                selected_records,
                state=state,
                stage="selecting",
                artifact_dir=artifact_dir,
                logger=logger,
                selection_mode=resolved_selection_mode,
                **uncertainty_curation_data,
            )
        state.selected_size = len(selected_records)
        plot_records = candidate_records if resolved_selection_mode == "cold_start" else scored_records
        write_iteration_plots(
            plot_records,
            state=state,
            plot_dir=plot_dir,
            selected_records=selected_records,
            curation_result=curation_result,
            logger=logger,
        )
        write_iteration_extxyz_outputs(
            plot_records,
            selected_records,
            state=state,
            plot_dir=plot_dir,
        )

        if dft_root_dir is None or not selected_records:
            self._finish_iteration(state, logger, plot_dir)
            return selected_records

        state.status = WORKFLOW_STATUS_LABELING
        labeled_records = self.label_selected(
            selected_records,
            root_dir=dft_root_dir,
            dft_runner=dft_runner,
            labeler_kwargs=labeler_kwargs,
            template_replacements=template_replacements,
        )
        self._checkpoint_records(
            labeled_records,
            state=state,
            stage="labeling",
            artifact_dir=artifact_dir,
            logger=logger,
            dft_root_dir=str(dft_root_dir),
        )
        if train_path is not None:
            append_labeled_records_to_train_path(train_path, labeled_records)
            state.train_size = _count_train_records(train_path)

        labeled_records = annotate_dft_labels(labeled_records)
        write_labeling_plots(
            labeled_records,
            state=state,
            plot_dir=plot_dir,
            logger=logger,
        )
        if plot_dir is not None:
            iteration_dir = Path(plot_dir) / f"iteration_{state.iteration:04d}"
            write_records_extxyz(
                labeled_records,
                iteration_dir / "labeled_records.extxyz",
            )

        self._finish_iteration(state, logger, plot_dir)
        return labeled_records

    def run(
        self,
        state: WorkflowState,
        *,
        model: BaseModel | None = None,
        train_path: str | Path | None = None,
        selection_mode: str = "auto",
        cold_start_k: int | None = None,
        seed_structures_by_iteration: list[list[Atoms]] | None = None,
        candidate_structures_by_iteration: list[list[Atoms]] | None = None,
        dft_root_dir: str | Path | None = None,
        uncertainty_k: int | Callable[[int], int] | None = None,
        valid_path: str | Path | None = None,
        train_prediction_model: bool = True,
        train_model_ensemble: bool = True,
        train_kwargs: dict[str, Any] | None = None,
        model_factory: ModelEnsembleFactory | Sequence[ModelEnsembleFactory] | None = None,
        model_factory_train_kwargs: (
            dict[str, Any] | Sequence[dict[str, Any] | None] | None
        ) = None,
        model_factory_gpu_ids: Sequence[int] | None = None,
        load_model_factory_checkpoints: bool = True,
        parse_base_dir: str | Path | None = None,
        parse_kwargs: dict[str, Any] | None = None,
        parser_output_paths: Mapping[str, str | Path] | None = None,
        parser_curation_pipeline: StructureCurationPipeline | None = None,
        parser_curation_kwargs: dict[str, Any] | None = None,
        seed_selection_mode: str = "all",
        seed_k: int | Callable[[int], int] | None = None,
        seed_selection_curation_pipeline: StructureCurationPipeline | None = None,
        seed_selection_curation_kwargs: dict[str, Any] | None = None,
        seed_selection_uncertainty: BaseUncertainty | None = None,
        seed_selection_score_key: str = "force_max_uncertainty",
        seed_selection_random_seed: int | None = None,
        sampler: BaseSampler | None = None,
        sampler_kwargs: dict[str, Any] | None = None,
        common_data: dict[str, Any] | None = None,
        pre_uncertainty_curation_pipeline: StructureCurationPipeline | None = None,
        pre_uncertainty_curation_kwargs: dict[str, Any] | None = None,
        uncertainty: BaseUncertainty | None = None,
        uncertainty_stop_key: str = "force_max_uncertainty",
        uncertainty_stop_expression: str | None = None,
        model_error_stop_force_threshold: float | None = None,
        model_error_stop_energy_threshold: float | None = None,
        model_error_stop_statistic: str = "max",
        selector: BaseSelector | None = None,
        uncertainty_curation_pipeline: BaseStructureCuration | None = None,
        coverage_k: int | Callable[[int], int] | None = None,
        coverage_curation_pipeline: StructureCurationPipeline | None = None,
        coverage_curation_kwargs: dict[str, Any] | None = None,
        dft_runner: BatchDFTRunner | None = None,
        labeler_kwargs: dict[str, Any] | None = None,
        template_replacements: dict[str, str] | None = None,
        logger: WorkflowLogger | None = None,
        artifact_dir: str | Path | None = None,
        plot_dir: str | Path | None = None,
        restart_from_stage: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        if (
            seed_structures_by_iteration is None
            and candidate_structures_by_iteration is None
            and parse_base_dir is None
        ):
            raise ValueError(
                "seed_structures_by_iteration, candidate_structures_by_iteration, "
                "or parse_base_dir must be provided."
            )

        lengths = [
            len(values)
            for values in (
                seed_structures_by_iteration,
                candidate_structures_by_iteration,
            )
            if values is not None
        ]
        if parse_base_dir is not None:
            lengths.append(1)
        num_iterations = max(lengths)

        all_iteration_records: list[list[dict[str, Any]]] = []
        for local_iteration in range(num_iterations):
            candidate_structures = get_iteration_value(
                candidate_structures_by_iteration,
                local_iteration,
            )
            seed_structures = get_iteration_value(
                seed_structures_by_iteration,
                local_iteration,
            )
            iteration_parse_base_dir = parse_base_dir if local_iteration == 0 else None
            iteration_dft_root = (
                Path(dft_root_dir) / f"iteration_{state.iteration:04d}"
                if dft_root_dir is not None
                else None
            )

            records = self.run_iteration(
                state,
                uncertainty_k=uncertainty_k,
                selection_mode=selection_mode,
                cold_start_k=cold_start_k,
                candidate_structures=candidate_structures,
                model=model,
                train_path=train_path,
                seed_structures=seed_structures,
                dft_root_dir=iteration_dft_root,
                valid_path=valid_path,
                train_prediction_model=train_prediction_model,
                train_model_ensemble=train_model_ensemble,
                train_kwargs=train_kwargs,
                model_factory=model_factory,
                model_factory_train_kwargs=model_factory_train_kwargs,
                model_factory_gpu_ids=model_factory_gpu_ids,
                load_model_factory_checkpoints=load_model_factory_checkpoints,
                parse_base_dir=iteration_parse_base_dir,
                parse_kwargs=parse_kwargs,
                parser_output_paths=parser_output_paths,
                parser_curation_pipeline=parser_curation_pipeline,
                parser_curation_kwargs=parser_curation_kwargs,
                seed_selection_mode=seed_selection_mode,
                seed_k=seed_k,
                seed_selection_curation_pipeline=seed_selection_curation_pipeline,
                seed_selection_curation_kwargs=seed_selection_curation_kwargs,
                seed_selection_uncertainty=seed_selection_uncertainty,
                seed_selection_score_key=seed_selection_score_key,
                seed_selection_random_seed=seed_selection_random_seed,
                sampler=sampler,
                sampler_kwargs=sampler_kwargs,
                common_data=common_data,
                pre_uncertainty_curation_pipeline=pre_uncertainty_curation_pipeline,
                pre_uncertainty_curation_kwargs=pre_uncertainty_curation_kwargs,
                uncertainty=uncertainty,
                uncertainty_stop_key=uncertainty_stop_key,
                uncertainty_stop_expression=uncertainty_stop_expression,
                model_error_stop_force_threshold=model_error_stop_force_threshold,
                model_error_stop_energy_threshold=model_error_stop_energy_threshold,
                model_error_stop_statistic=model_error_stop_statistic,
                selector=selector,
                uncertainty_curation_pipeline=uncertainty_curation_pipeline,
                coverage_k=coverage_k,
                coverage_curation_pipeline=coverage_curation_pipeline,
                coverage_curation_kwargs=coverage_curation_kwargs,
                dft_runner=dft_runner,
                labeler_kwargs=labeler_kwargs,
                template_replacements=template_replacements,
                logger=logger,
                artifact_dir=artifact_dir,
                plot_dir=plot_dir,
                restart_from_stage=restart_from_stage,
            )
            all_iteration_records.append(records)

        return all_iteration_records

    def resume(
        self,
        run_dir: str | Path,
        **run_kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        run_dir = Path(run_dir)
        plan = build_restart_plan(run_dir)
        run_kwargs.setdefault("plot_dir", run_dir / "plots")
        return self.run(
            state=plan.state,
            artifact_dir=run_dir / "artifacts",
            restart_from_stage=plan.next_stage,
            **run_kwargs,
        )

    def append_records_to_db(
        self,
        db: Database,
        records: list[dict[str, Any]],
        *,
        split: str,
        is_labeled: bool | None = None,
        is_selected: bool | None = None,
        source: str = "active_learning",
    ) -> list[int]:
        return append_records_to_db(
            db,
            records,
            split=split,
            is_labeled=is_labeled,
            is_selected=is_selected,
            source=source,
        )

    def append_selected_to_db(
        self,
        db: Database,
        selected_records: list[dict[str, Any]],
        *,
        split: str = "selected",
        source: str = "active_learning",
    ) -> list[int]:
        return append_selected_to_db(
            db,
            selected_records,
            split=split,
            source=source,
        )
