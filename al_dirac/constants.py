from __future__ import annotations

PACKAGE_NAME = "al_dirac"
CLI_NAME = "al-dirac"
DEFAULT_RANDOM_SEED = 42

DEFAULT_CONFIG_NAME = "default.yaml"
DEFAULT_STATE_FILE = "state.json"
DEFAULT_RUN_CONFIG_FILE = "run_config.json"
DEFAULT_METRICS_FILE = "metrics.json"
DEFAULT_EVENTS_FILE = "events.jsonl"
DEFAULT_LOG_FILE = "al_dirac.log"

DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_SPLIT_DIR = "splits"
DEFAULT_PLOT_DIR = "plots"
DEFAULT_DFT_DIR = "dft_jobs"

DEFAULT_DATA_FORMAT = "aselmdb"
DEFAULT_TRAIN_FILE = "train.aselmdb"
DEFAULT_VALID_FILE = "valid.aselmdb"
DEFAULT_TEST_FILE = "test.aselmdb"
DEFAULT_POOL_FILE = "pool.aselmdb"
DEFAULT_SELECTED_FILE = "selected.aselmdb"
DEFAULT_LABELED_FILE = "labeled.aselmdb"

DEFAULT_EXTXYZ_FORMAT = "extxyz"
DEFAULT_EXTXYZ_TRAIN_FILE = "train.extxyz"
DEFAULT_EXTXYZ_VALID_FILE = "valid.extxyz"

SUPPORTED_MODELS = (
    "mace",
    "fairchem",
)

SUPPORTED_SAMPLERS = (
    "rattle",
    "strain",
    "md",
    "relaxation",
    "bhopping",
    "defects",
    "dimer",
    "udd",
)

SUPPORTED_UNCERTAINTY_METHODS = (
    "ensemble",
    "latent_distance",
    "calibration",
)

SUPPORTED_SELECTION_METHODS = (
    "threshold",
    "diversity",
    "clustering",
    "score",
)

SUPPORTED_DFT_ENGINES = (
    "vasp",
)

WORKFLOW_STATUS_INITIALIZED = "initialized"
WORKFLOW_STATUS_BOOTSTRAPPING = "bootstrapping"
WORKFLOW_STATUS_TRAINING = "training"
WORKFLOW_STATUS_SAMPLING = "sampling"
WORKFLOW_STATUS_PARSING = "parsing"
WORKFLOW_STATUS_CURATING = "curating"
WORKFLOW_STATUS_SCORING = "scoring"
WORKFLOW_STATUS_SELECTING = "selecting"
WORKFLOW_STATUS_LABELING = "labeling"
WORKFLOW_STATUS_EVALUATING = "evaluating"
WORKFLOW_STATUS_COMPLETE = "complete"
WORKFLOW_STATUS_FAILED = "failed"

WORKFLOW_STATUSES = (
    WORKFLOW_STATUS_INITIALIZED,
    WORKFLOW_STATUS_BOOTSTRAPPING,
    WORKFLOW_STATUS_TRAINING,
    WORKFLOW_STATUS_SAMPLING,
    WORKFLOW_STATUS_PARSING,
    WORKFLOW_STATUS_CURATING,
    WORKFLOW_STATUS_SCORING,
    WORKFLOW_STATUS_SELECTING,
    WORKFLOW_STATUS_LABELING,
    WORKFLOW_STATUS_EVALUATING,
    WORKFLOW_STATUS_COMPLETE,
    WORKFLOW_STATUS_FAILED,
)
