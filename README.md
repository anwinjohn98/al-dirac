# al-dirac

Active Learning with Diversity, Informativeness, and Representativeness for Atomic Catalysis.

`al-dirac` automates the active-learning loop for training machine-learned interatomic
potentials (currently [MACE](https://github.com/ACEsuit/mace)) against VASP DFT data,
built around a Pt-surface catalysis (ORR-type O/OH/OOH) use case. It handles the full
cycle: parsing DFT results, training model ensembles, sampling new configurations,
curating and de-duplicating structures, scoring candidates by ensemble uncertainty,
selecting what to label next, and submitting/collecting real VASP jobs.

## Features

- **Parsing**: import VASP outputs (`vasprun.xml`/OUTCAR) into `.aselmdb`/`.extxyz`
  training pools, with automatic labeled/unlabeled detection.
- **Model training**: a MACE wrapper (`al_dirac.models.mace_model.MACEModel`) supporting
  training from scratch, plain fine-tuning, multihead fine-tuning, and LoRA fine-tuning,
  plus a backend-agnostic `ModelEnsembleFactory` for building ensembles across seeds,
  hyperparameters, or data splits.
- **Uncertainty**: ensemble-disagreement-based uncertainty (`EnsembleUncertainty`) used
  to score and select candidates for labeling, with a configurable stopping criterion.
- **Sampling**: structure perturbation/exploration samplers (rattle, dimer, MD-based).
- **Curation**: cheap deduplication (energy-binned, periodic-boundary-aware position
  matching), SOAP-descriptor-based diversity curation, clustering-based selection, and a
  force-cutoff filter for removing unphysical (e.g. diverging) structures.
- **DFT labeling**: batches candidate structures into real VASP jobs, submits them via a
  user-provided SLURM template, and collects results once they finish.
- **Workflow orchestration**: `ActiveLearningWorkflow` runs the full loop end to end, with
  restart/resume support so it can be split across separate SLURM job submissions (see
  `examples/workflow_loop.py`).

## Installation

```bash
pip install -e .[ml,dft,dev]
```

- `ml` — MACE/PyTorch training dependencies (`torch`, `mace-torch`, `fairchem-core`).
- `dft` — DFT-adjacent tooling (`pymatgen`, `custodian`).
- `dev` — testing/linting tools.

MACE training is GPU- and dependency-heavy (PyTorch, `cuequivariance`, CUDA toolchain).
An `apptainer.def` container definition is included for running on HPC clusters where
these need to be isolated from the host environment:

```bash
apptainer build al_dirac-python.sif apptainer.def
```

## Usage

The `examples/` directory has runnable, end-to-end scripts against real data, each
independently testable:

| Script | Demonstrates |
|---|---|
| `parser_example.py` | Parsing VASP outputs into a training pool |
| `mace_train_scratch.py` / `mace_train_scratch_aselmdb.py` | Training MACE from scratch (extxyz / aselmdb) |
| `mace_finetune.py`, `mace_multihead_finetune.py`, `mace_lora_finetune.py` | Fine-tuning a foundation model |
| `model_factory_example.py` | Building a model ensemble with `ModelEnsembleFactory` |
| `sampler_example.py` | Structure sampling |
| `curation_example.py`, `cheap_curate_trajectory_example.py` | Structure curation/deduplication |
| `uncertainty_selection_example.py` | Scoring and selecting candidates by ensemble uncertainty |
| `stopping_example.py` | Checking the uncertainty-based stopping criterion |
| `dft_batch_example.py` | Preparing and submitting real VASP batch jobs |
| `workflow_loop.py` | The full active-learning loop, split into SLURM-job-safe stages |

Each `examples/<name>.py` has a matching `runs/<name>/<name>.run` SLURM submission script
where one is needed.

## Project status

Actively developed against a single real system (Pt-surface ORR intermediates) as the
validation case; the model/uncertainty/sampling APIs are designed to be backend-agnostic
so other MLIP backends can be added. Not yet packaged with an automated test suite.

## License

MIT — see [LICENSE](LICENSE).
