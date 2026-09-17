from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import random
from typing import Any, Mapping, Sequence, TypeVar

from al_dirac.models.base import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def _nested_dict(kwargs: dict[str, Any], key: str) -> dict[str, Any]:
    value = kwargs.get(key)
    if value is None:
        value = {}
        kwargs[key] = value
    if not isinstance(value, dict):
        raise TypeError(f"The value for {key!r} must be a dictionary.")
    return value


class ModelFactoryStrategy:
    def apply(self, index: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs

    def prepare_training(
        self,
        index: int,
        train_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return train_kwargs


class SeedStrategy(ModelFactoryStrategy):
    def __init__(
        self,
        seed_start: int,
        *,
        seed_key: str = "seed",
        training_kwargs_key: str = "training_kwargs",
        overwrite: bool = False,
    ) -> None:
        self.seed_start = seed_start
        self.seed_key = seed_key
        self.training_kwargs_key = training_kwargs_key
        self.overwrite = overwrite

    def apply(self, index: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        training_kwargs = _nested_dict(kwargs, self.training_kwargs_key)
        if self.overwrite or self.seed_key not in training_kwargs:
            training_kwargs[self.seed_key] = self.seed_start + index
        return kwargs

    def prepare_training(
        self,
        index: int,
        train_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        # apply() seeds the constructor's training_kwargs, which only
        # MACEModel.train() ever reads. finetune()/lora_finetune()/
        # multihead_finetune() take `seed` as a call-time argument instead, so
        # seed it here too.
        if self.overwrite or self.seed_key not in train_kwargs:
            train_kwargs[self.seed_key] = self.seed_start + index
        return train_kwargs


class HyperparameterStrategy(ModelFactoryStrategy):
    def __init__(
        self,
        hyperparameters: Sequence[Mapping[str, Any]],
        *,
        training_kwargs_key: str = "training_kwargs",
        overwrite: bool = True,
    ) -> None:
        if not hyperparameters:
            raise ValueError("hyperparameters must not be empty.")

        self.hyperparameters = [dict(values) for values in hyperparameters]
        self.training_kwargs_key = training_kwargs_key
        self.overwrite = overwrite

    def apply(self, index: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        if index >= len(self.hyperparameters):
            raise ValueError(f"Missing hyperparameters for model index {index}.")

        training_kwargs = _nested_dict(kwargs, self.training_kwargs_key)
        for key, value in self.hyperparameters[index].items():
            if self.overwrite or key not in training_kwargs:
                training_kwargs[key] = value
        return kwargs

    def prepare_training(
        self,
        index: int,
        train_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        # Same rationale as SeedStrategy.prepare_training: apply() only
        # reaches MACEModel.train() via the constructor's training_kwargs, so
        # hyperparameters need to be set here too for finetune-style methods.
        if index >= len(self.hyperparameters):
            raise ValueError(f"Missing hyperparameters for model index {index}.")

        for key, value in self.hyperparameters[index].items():
            if self.overwrite or key not in train_kwargs:
                train_kwargs[key] = value
        return train_kwargs


class UserDataSplitStrategy(ModelFactoryStrategy):
    def __init__(
        self,
        train_paths: Sequence[str | Path],
        *,
        valid_paths: Sequence[str | Path | None] | None = None,
        train_key: str = "train_file",
        valid_key: str = "valid_file",
    ) -> None:
        if not train_paths:
            raise ValueError("train_paths must not be empty")
        if valid_paths is not None and len(valid_paths) != len(train_paths):
            raise ValueError("valid_paths must have the same length as train_paths")

        self.train_paths = [Path(path) for path in train_paths]
        self.valid_paths = (
            None
            if valid_paths is None
            else [None if path is None else Path(path) for path in valid_paths]
        )
        self.train_key = train_key
        self.valid_key = valid_key

    def prepare_training(
        self,
        index: int,
        train_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if index >= len(self.train_paths):
            raise ValueError(f"Missing user data split for model index {index}.")

        train_kwargs[self.train_key] = self.train_paths[index]
        if self.valid_paths is not None:
            train_kwargs[self.valid_key] = self.valid_paths[index]
        return train_kwargs


class RandomDataSplitStrategy(ModelFactoryStrategy):
    def __init__(
        self,
        split_root: str | Path,
        *,
        train_fraction: float = 1.0,
        valid_fraction: float = 0.1,
        seed_start: int = 0,
        file_format: str = "extxyz",
        dirname_prefix: str = "split",
        train_filename: str = "train.extxyz",
        valid_filename: str = "valid.extxyz",
        start_index: int = 0,
        width: int = 3,
        train_key: str = "train_file",
        valid_key: str = "valid_file",
    ) -> None:
        if not 0.0 < train_fraction <= 1.0:
            raise ValueError("train_fraction must satisfy 0 < train_fraction <= 1.")

        if not 0.0 <= valid_fraction < 1.0:
            raise ValueError("valid_fraction must satisfy 0 <= valid_fraction < 1.")

        if train_fraction + valid_fraction > 1.0:
            raise ValueError("train_fraction + valid_fraction must be <= 1.")

        self.split_root = Path(split_root)
        self.train_fraction = train_fraction
        self.valid_fraction = valid_fraction
        self.seed_start = seed_start
        self.file_format = file_format
        self.dirname_prefix = dirname_prefix
        self.train_filename = train_filename
        self.valid_filename = valid_filename
        self.start_index = start_index
        self.width = width
        self.train_key = train_key
        self.valid_key = valid_key
        self._cached_source: Path | None = None
        self._cached_structures: list[Any] | None = None

    def _load_structures(self, train_path: str | Path) -> list[Any]:
        source = Path(train_path)
        if self._cached_source == source and self._cached_structures is not None:
            return self._cached_structures

        from ase.io import read

        structures = read(source, index=":")
        if not isinstance(structures, list):
            structures = [structures]
        if not structures:
            raise ValueError(f"No structures found in {source}.")

        self._cached_source = source
        self._cached_structures = structures
        return structures

    def prepare_training(
        self,
        index: int,
        train_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        from ase.io import write

        source_train_path = train_kwargs.get(self.train_key)
        if source_train_path is None:
            raise ValueError(
                f"train_kwargs[{self.train_key!r}] must already be set before "
                "RandomDataSplitStrategy runs."
            )

        structures = self._load_structures(source_train_path)
        indices = list(range(len(structures)))
        random.Random(self.seed_start + index).shuffle(indices)

        n_total = len(indices)
        n_train = int(round(n_total * self.train_fraction))
        n_valid = int(round(n_total * self.valid_fraction))

        n_train = max(1, min(n_train, n_total))
        if self.valid_fraction > 0.0 and n_total > 1:
            n_valid = max(1, n_valid)
            n_valid = min(n_valid, n_total - n_train)

        train_indices = set(indices[:n_train])
        valid_indices = set(indices[n_train:n_train + n_valid])

        train_structures = [
            atoms
            for structure_index, atoms in enumerate(structures)
            if structure_index in train_indices
        ]
        valid_structures = [
            atoms
            for structure_index, atoms in enumerate(structures)
            if structure_index in valid_indices
        ]

        split_index = self.start_index + index
        split_dir = (
            self.split_root / f"{self.dirname_prefix}_{split_index:0{self.width}d}"
        )
        split_dir.mkdir(parents=True, exist_ok=True)

        split_train_path = split_dir / self.train_filename
        write(split_train_path, train_structures, format=self.file_format)
        train_kwargs[self.train_key] = split_train_path

        if n_valid == 0:
            return train_kwargs

        split_valid_path = split_dir / self.valid_filename
        write(split_valid_path, valid_structures, format=self.file_format)
        train_kwargs[self.valid_key] = split_valid_path
        return train_kwargs


class ModelEnsembleFactory:
    def __init__(
        self,
        model_cls: type[ModelT],
        *,
        n_models: int,
        name_prefix: str,
        training_method: str = "train",
        checkpoint_root: str | Path | None = None,
        base_kwargs: dict[str, Any] | None = None,
        name_key: str = "name",
        checkpoint_dir_key: str = "checkpoint_dir",
        checkpoint_dir_extra_arg: str | None = None,
        training_kwargs_key: str = "training_kwargs",
        extra_args_key: str = "extra_args",
        train_key: str = "train_file",
        valid_key: str = "valid_file",
        start_index: int = 0,
        width: int = 3,
    ) -> None:
        if n_models <= 0:
            raise ValueError("n_models must be positive.")

        self.model_cls = model_cls
        self.n_models = n_models
        self.name_prefix = name_prefix
        self.training_method = training_method
        self.checkpoint_root = (
            Path(checkpoint_root) if checkpoint_root is not None else None
        )
        self.base_kwargs = {} if base_kwargs is None else dict(base_kwargs)
        self.name_key = name_key
        self.checkpoint_dir_key = checkpoint_dir_key
        self.checkpoint_dir_extra_arg = checkpoint_dir_extra_arg
        self.training_kwargs_key = training_kwargs_key
        self.extra_args_key = extra_args_key
        # Configurable rather than hardcoded so the factory can drive any
        # model backend's training method (MACEModel's finetune-style methods
        # use train_file/valid_file; a different backend, e.g. a future UMA
        # wrapper, may name these differently), the same way MDSampler passes
        # dynamics_kwargs through without assuming a specific dynamics class.
        self.train_key = train_key
        self.valid_key = valid_key
        self.start_index = start_index
        self.width = width
        self.strategies: list[ModelFactoryStrategy] = []

    def add_strategy(self, strategy: ModelFactoryStrategy) -> None:
        self.strategies.append(strategy)

    def build_kwargs(self) -> list[dict[str, Any]]:
        all_kwargs: list[dict[str, Any]] = []
        for index in range(self.n_models):
            model_index = self.start_index + index
            kwargs = deepcopy(self.base_kwargs)
            kwargs[self.name_key] = f"{self.name_prefix}_{model_index:0{self.width}d}"

            if self.checkpoint_root is not None:
                checkpoint_dir = (
                    self.checkpoint_root
                    / f"{self.name_prefix}_{model_index:0{self.width}d}"
                )
                kwargs[self.checkpoint_dir_key] = checkpoint_dir

                if self.checkpoint_dir_extra_arg is not None:
                    training_kwargs = _nested_dict(kwargs, self.training_kwargs_key)
                    extra_args = list(training_kwargs.get(self.extra_args_key, []))
                    extra_args.extend(
                        [self.checkpoint_dir_extra_arg, str(checkpoint_dir)]
                    )
                    training_kwargs[self.extra_args_key] = extra_args

            for strategy in self.strategies:
                kwargs = strategy.apply(index, kwargs)
            all_kwargs.append(kwargs)
        return all_kwargs

    def build(self) -> list[ModelT]:
        return [self.model_cls(**kwargs) for kwargs in self.build_kwargs()]

    def _train_one(
        self,
        index: int,
        model: ModelT,
        *,
        train_path: str | Path,
        valid_path: str | Path | None,
        base_train_kwargs: dict[str, Any],
        load_checkpoints: bool,
        gpu_id: int | None,
    ) -> None:
        model_train_kwargs = dict(base_train_kwargs)
        model_train_kwargs.setdefault(self.train_key, train_path)
        if valid_path is not None:
            model_train_kwargs.setdefault(self.valid_key, valid_path)
        if gpu_id is not None:
            model_train_kwargs.setdefault("cuda_visible_devices", str(gpu_id))

        for strategy in self.strategies:
            model_train_kwargs = strategy.prepare_training(index, model_train_kwargs)

        method = getattr(model, self.training_method, None)
        if method is None or not callable(method):
            raise ValueError(
                f"{type(model).__name__} does not support training_method "
                f"'{self.training_method}'."
            )

        # Fully keyword, with no assumptions about which arguments the
        # training method accepts beyond train_key/valid_key: this makes
        # the factory work with train()/finetune()/lora_finetune()/
        # multihead_finetune() alike (which differ in whether
        # foundation_model is a required positional argument), and with
        # any future model backend by reconfiguring train_key/valid_key
        # to match its own method signature.
        method(**model_train_kwargs)
        if not load_checkpoints:
            return
        expected_checkpoint = getattr(model, "expected_checkpoint_path", None)
        if expected_checkpoint is None:
            return
        checkpoint_path = expected_checkpoint()
        if checkpoint_path is not None and checkpoint_path.exists():
            model.save(checkpoint_path)

    def train(
        self,
        train_path: str | Path,
        *,
        valid_path: str | Path | None = None,
        train_kwargs: dict[str, Any] | None = None,
        load_checkpoints: bool = True,
        gpu_ids: Sequence[int] | None = None,
    ) -> list[ModelT]:
        base_train_kwargs = {} if train_kwargs is None else dict(train_kwargs)
        models = self.build()

        if gpu_ids is None:
            for index, model in enumerate(models):
                self._train_one(
                    index,
                    model,
                    train_path=train_path,
                    valid_path=valid_path,
                    base_train_kwargs=base_train_kwargs,
                    load_checkpoints=load_checkpoints,
                    gpu_id=None,
                )
            return models

        # gpu_ids given: each ensemble member is an independent training run
        # (different seed/hyperparameters), so training them concurrently on
        # separate GPUs -- rather than one after another on a single GPU --
        # is safe and cuts ensemble wall-clock time roughly by GPU count.
        # cuda_visible_devices pins each subprocess to one physical GPU via
        # its own environment, so concurrent subprocess.run calls don't
        # collide on the same device even though they share this process.
        if not gpu_ids:
            raise ValueError("gpu_ids must be non-empty when provided.")

        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = [
                executor.submit(
                    self._train_one,
                    index,
                    model,
                    train_path=train_path,
                    valid_path=valid_path,
                    base_train_kwargs=base_train_kwargs,
                    load_checkpoints=load_checkpoints,
                    gpu_id=gpu_ids[index % len(gpu_ids)],
                )
                for index, model in enumerate(models)
            ]
            for future in futures:
                future.result()

        return models
