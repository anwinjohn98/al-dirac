from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

from ase import Atoms

from al_dirac.dft.base import BaseDFTLabeler

_SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")


class BatchDFTRunner:
    def __init__(
        self,
        labeler: BaseDFTLabeler,
        batch_size: int,
        max_concurrent_batches: int = 1,
        poll_interval_seconds: int = 300,
        submission_template: str | Path | None = None,
        submit_command: list[str] | None = None,
        run_command_template: str | None = None,
        submit_script_name: str = "submit_batch.sh",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_concurrent_batches <= 0:
            raise ValueError("max_concurrent_batches must be positive.")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive.")
        if submission_template is None:
            raise ValueError("submission_template must be provided.")
        if run_command_template is None:
            raise ValueError("run_command_template must be provided.")

        self.labeler = labeler
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.poll_interval_seconds = poll_interval_seconds
        self.submission_template = Path(submission_template)
        self.submit_command = ["bash"] if submit_command is None else list(submit_command)
        self.run_command_template = run_command_template
        self.submit_script_name = submit_script_name

    def _chunk_records(
        self,
        records: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        return [
            records[start : start + self.batch_size]
            for start in range(0, len(records), self.batch_size)
        ]

    def _job_dir(
        self,
        batch_dir: Path,
        record: dict[str, Any],
        local_index: int,
    ) -> Path:
        structure_id = str(record.get("structure_id", f"job_{local_index:06d}"))
        safe_id = structure_id.replace("/", "__")
        return batch_dir / safe_id

    def _build_job_loop(
        self,
        prepared_jobs: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []

        for job in prepared_jobs:
            job_dir = Path(job["job_dir"]).resolve()
            lines.append(f'cd "{job_dir}"')
            lines.append(self.run_command_template)
            lines.append("")

        return "\n".join(lines).strip()

    def _render_template(
        self,
        template_text: str,
        replacements: dict[str, str],
    ) -> str:
        rendered = template_text
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)
        return rendered

    def prepare_batches(
        self,
        structures: list[Atoms],
        root_dir: str | Path,
        *,
        records: list[dict[str, Any]] | None = None,
        labeler_kwargs: dict[str, Any] | None = None,
        template_replacements: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        if records is None:
            records = [
                {"atoms": atoms, "structure_id": f"structure_{i:06d}"}
                for i, atoms in enumerate(structures)
            ]

        if len(records) != len(structures):
            raise ValueError("records and structures must have the same length.")

        normalized_records: list[dict[str, Any]] = []
        for atoms, record in zip(structures, records):
            normalized = dict(record)
            normalized["atoms"] = atoms
            normalized_records.append(normalized)

        batches = self._chunk_records(normalized_records)
        prepared_batches: list[dict[str, Any]] = []

        for batch_index, batch_records in enumerate(batches):
            batch_dir = root_dir / f"batch_{batch_index:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            prepared_jobs: list[dict[str, Any]] = []
            for local_index, record in enumerate(batch_records):
                job_dir = self._job_dir(batch_dir, record, local_index)
                self.labeler.prepare_job(
                    record["atoms"],
                    job_dir,
                    **(labeler_kwargs or {}),
                )
                prepared_jobs.append(
                    {
                        "record": record,
                        "job_dir": job_dir,
                    }
                )

            submit_script = batch_dir / self.submit_script_name
            self.write_submit_script(
                submit_script=submit_script,
                batch_dir=batch_dir,
                prepared_jobs=prepared_jobs,
                template_replacements=template_replacements,
            )

            prepared_batches.append(
                {
                    "batch_index": batch_index,
                    "batch_dir": batch_dir,
                    "submit_script": submit_script,
                    "jobs": prepared_jobs,
                }
            )

        return prepared_batches

    def write_submit_script(
        self,
        submit_script: str | Path,
        batch_dir: str | Path,
        prepared_jobs: list[dict[str, Any]],
        template_replacements: dict[str, str] | None = None,
    ) -> Path:
        submit_script = Path(submit_script)
        batch_dir = Path(batch_dir)

        template_text = self.submission_template.read_text()
        job_loop = self._build_job_loop(prepared_jobs)

        replacements = {
            "__BATCH_DIR__": str(batch_dir.resolve()),
            "__NUM_JOBS__": str(len(prepared_jobs)),
            "__JOB_LOOP__": job_loop,
            **(template_replacements or {}),
        }

        rendered = self._render_template(template_text, replacements)
        submit_script.write_text(rendered)
        submit_script.chmod(0o755)

        return submit_script

    def _is_scheduler_submission(self) -> bool:
        return bool(self.submit_command) and self.submit_command[0] == "sbatch"

    def submit_batches(
        self,
        prepared_batches: list[dict[str, Any]],
    ) -> list[str | None]:
        if self._is_scheduler_submission():
            # sbatch queues the job and returns almost immediately -- there is
            # no actual work to poll here. Capture the real SLURM job ID from
            # its stdout so the caller can chain a follow-up job with
            # `sbatch --dependency=afterok:<job_id>` instead of assuming the
            # DFT run is done once this call returns.
            job_ids: list[str | None] = []
            for batch in prepared_batches:
                result = subprocess.run(
                    [*self.submit_command, str(batch["submit_script"])],
                    cwd=batch["batch_dir"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                match = _SBATCH_JOB_ID_PATTERN.search(result.stdout)
                if match is None:
                    raise RuntimeError(
                        "Could not parse a SLURM job ID from sbatch output: "
                        f"{result.stdout!r}"
                    )
                batch["job_id"] = match.group(1)
                job_ids.append(batch["job_id"])
            return job_ids

        # Non-scheduler submission (e.g. the default "bash", which runs the
        # batch script directly): this subprocess *is* the actual DFT work,
        # so wait for it, throttling how many run concurrently.
        active: list[dict[str, Any]] = []
        remaining = list(prepared_batches)

        while remaining or active:
            while remaining and len(active) < self.max_concurrent_batches:
                batch = remaining.pop(0)
                process = subprocess.Popen(
                    [*self.submit_command, str(batch["submit_script"])],
                    cwd=batch["batch_dir"],
                )
                batch["process"] = process
                active.append(batch)

            still_active: list[dict[str, Any]] = []
            for batch in active:
                process = batch["process"]
                if process.poll() is None:
                    still_active.append(batch)

            active = still_active

            if remaining or active:
                time.sleep(self.poll_interval_seconds)

        return [None for _ in prepared_batches]

    def collect_results(
        self,
        prepared_batches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        labeled_records: list[dict[str, Any]] = []

        for batch in prepared_batches:
            for job in batch["jobs"]:
                job_dir = Path(job["job_dir"])
                record = dict(job["record"])

                if not self.labeler.job_succeeded(job_dir):
                    continue

                result = self.labeler.collect_result(job_dir)
                merged = dict(record)
                merged.update(result)
                workflow = dict(merged.get("workflow", {}))
                workflow["dft_labeling"] = {
                    "labeler": self.labeler.labeler_name,
                    "labeler_class": type(self.labeler).__name__,
                    "batch_index": batch["batch_index"],
                    "batch_dir": str(Path(batch["batch_dir"])),
                    "job_dir": str(job_dir),
                }
                merged["workflow"] = workflow
                labeled_records.append(merged)

        return labeled_records

    def run(
        self,
        structures: list[Atoms],
        root_dir: str | Path,
        *,
        records: list[dict[str, Any]] | None = None,
        labeler_kwargs: dict[str, Any] | None = None,
        template_replacements: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_scheduler_submission():
            raise ValueError(
                "run() submits and immediately collects results, but sbatch "
                "returns before the DFT job actually finishes -- this would "
                "silently return too few labeled records. Call "
                "prepare_batches()/submit_batches() to submit, then "
                "collect_results() separately once the SLURM jobs have "
                "completed (e.g. via a --dependency=afterok chained job)."
            )

        prepared_batches = self.prepare_batches(
            structures=structures,
            root_dir=root_dir,
            records=records,
            labeler_kwargs=labeler_kwargs,
            template_replacements=template_replacements,
        )
        self.submit_batches(prepared_batches)
        return self.collect_results(prepared_batches)
