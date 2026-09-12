from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from mace.calculators import MACECalculator

from al_dirac.models.base import BaseModel
from al_dirac.workflow.train_store import split_aselmdb_train_valid

# mace_run_train can only ASE-read these formats via check_path_ase_read();
# valid_fraction auto-splitting does not work for any of them.
_NON_ASE_READABLE_SUFFIXES = {".h5", ".hdf5", ".lmdb", ".aselmdb", ".mdb"}


class MACEModel(BaseModel):
    def __init__(
        self,
        name: str = "al_dirac_mace",
        model: str = "MACE",
        checkpoint_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        device: str = "cpu",
        default_dtype: str = "float64",
        head: str | None = None,
        run_train_script: str | Path | None = None,
        training_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_name=model)
        self.name = name
        self.model = model
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.device = device
        self.default_dtype = default_dtype
        self.head = head
        self.run_train_script = (
            Path(run_train_script) if run_train_script is not None else None
        )
        self.training_kwargs = {} if training_kwargs is None else dict(training_kwargs)
        self.calculator: MACECalculator | None = None

        if self.checkpoint_path is not None:
            self._build_calculator()

    def expected_checkpoint_path(self) -> Path | None:
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / f"{self.name}.model"

    def _build_calculator(self) -> None:
        if self.checkpoint_path is None:
            raise RuntimeError("checkpoint_path is required to build the calculator.")

        kwargs: dict[str, Any] = {
            "model_paths": str(self.checkpoint_path),
            "device": self.device,
            "default_dtype": self.default_dtype,
        }
        if self.head is not None:
            kwargs["head"] = self.head

        self.calculator = MACECalculator(**kwargs)

    def _resolve_train_entry(self) -> list[str]:
        mace_cli = shutil.which("mace_run_train")
        if mace_cli is not None:
            return [mace_cli]

        if self.run_train_script is not None:
            return [sys.executable, str(self.run_train_script)]

        raise RuntimeError(
            "Could not find 'mace_run_train', and no run_train.py path was provided."
        )

    def _resolve_fine_tuning_select_entry(self) -> list[str]:
        return [sys.executable, "-m", "mace.cli.fine_tuning_select"]

    def _extend_optional_args(
        self,
        args: list[str],
        *pairs: tuple[str, Any],
    ) -> list[str]:
        for flag, value in pairs:
            if value is None:
                continue
            args.extend([flag, str(value)])
        return args

    def _build_train_args(
        self,
        train_file: str | Path,
        valid_file: str | Path | None = None,
        test_file: str | Path | None = None,
        foundation_model: str | Path | None = None,
        foundation_model_kwargs: str | None = None,
        foundation_head: str | None = None,
        foundation_model_readout: bool = False,
        foundation_model_elements: bool | None = None,
        pt_train_file: str | Path | None = None,
        pt_valid_file: str | Path | None = None,
        statistics_file: str | Path | None = None,
        work_dir: str | Path | None = None,
        heads: str | None = None,
        hidden_irreps: str | None = None,
        atomic_numbers: Sequence[int] | None = None,
        config_type_weights: str | None = None,
        energy_key: str | None = None,
        forces_key: str | None = None,
        stress_key: str | None = None,
        E0s: str | Path | None = None,
        batch_size: int | None = None,
        valid_batch_size: int | None = None,
        max_num_epochs: int | None = None,
        valid_fraction: float | None = None,
        r_max: float | None = None,
        lr: float | None = None,
        ema: bool | None = None,
        ema_decay: float | None = None,
        loss: str | None = None,
        num_channels: int | None = None,
        max_L: int | None = None,
        num_interactions: int | None = None,
        num_workers: int | None = None,
        num_samples_pt: int | None = None,
        subselect_pt: str | None = None,
        filter_type_pt: str | None = None,
        weight_pt: float | None = None,
        multiheads_finetuning: bool | None = None,
        force_mh_ft_lr: bool | None = None,
        lora: bool | None = None,
        lora_rank: int | None = None,
        lora_alpha: float | None = None,
        enable_cueq: bool | None = None,
        only_cueq: bool | None = None,
        distributed: bool = False,
        seed: int | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> list[str]:
        train_path = Path(train_file)
        if valid_file is None and train_path.suffix in _NON_ASE_READABLE_SUFFIXES:
            if train_path.suffix == ".aselmdb":
                train_file, valid_file = split_aselmdb_train_valid(
                    train_path,
                    valid_fraction=valid_fraction if valid_fraction is not None else 0.1,
                    seed=seed,
                )
                valid_fraction = None
                print(
                    f"al_dirac: no valid_file given for '{train_path}' -- "
                    f"auto-split into '{train_file}' and '{valid_file}' "
                    "(mace_run_train's valid_fraction does not support "
                    ".aselmdb train files)."
                )
            else:
                raise ValueError(
                    f"train_file '{train_path}' has format "
                    f"'{train_path.suffix}'; mace_run_train's valid_fraction "
                    "auto-split does not support this format and al_dirac "
                    "does not yet auto-split it either -- pass an explicit "
                    "valid_file."
                )

        args = [
            "--name",
            self.name,
            "--model",
            self.model,
            "--train_file",
            str(train_file),
        ]

        self._extend_optional_args(
            args,
            ("--valid_file", valid_file),
            ("--test_file", test_file),
            ("--foundation_model", foundation_model),
            ("--foundation_model_kwargs", foundation_model_kwargs),
            ("--foundation_head", foundation_head),
            ("--foundation_model_elements", foundation_model_elements),
            ("--pt_train_file", pt_train_file),
            ("--pt_valid_file", pt_valid_file),
            ("--statistics_file", statistics_file),
            ("--work_dir", work_dir),
            ("--heads", heads),
            ("--hidden_irreps", hidden_irreps),
            ("--config_type_weights", config_type_weights),
            ("--energy_key", energy_key),
            ("--forces_key", forces_key),
            ("--stress_key", stress_key),
            ("--E0s", E0s),
            ("--batch_size", batch_size),
            ("--valid_batch_size", valid_batch_size),
            ("--max_num_epochs", max_num_epochs),
            ("--valid_fraction", valid_fraction),
            ("--r_max", r_max),
            ("--lr", lr),
            ("--ema_decay", ema_decay),
            ("--loss", loss),
            ("--num_channels", num_channels),
            ("--max_L", max_L),
            ("--num_interactions", num_interactions),
            ("--num_workers", num_workers),
            ("--num_samples_pt", num_samples_pt),
            ("--subselect_pt", subselect_pt),
            ("--filter_type_pt", filter_type_pt),
            ("--weight_pt", weight_pt),
            ("--multiheads_finetuning", multiheads_finetuning),
            ("--force_mh_ft_lr", force_mh_ft_lr),
            ("--lora", lora),
            ("--lora_rank", lora_rank),
            ("--lora_alpha", lora_alpha),
            ("--enable_cueq", enable_cueq),
            ("--only_cueq", only_cueq),
            ("--seed", seed),
        )

        if ema:
            args.append("--ema")

        if foundation_model_readout:
            args.append("--foundation_model_readout")

        if atomic_numbers is not None:
            args.extend(["--atomic_numbers", str(list(atomic_numbers))])

        if distributed:
            args.append("--distributed")

        if extra_args:
            args.extend(extra_args)

        return args

    def _run_training_command(
        self,
        train_args: list[str],
        launch_mode: str = "single",
        nproc_per_node: int = 1,
    ) -> None:
        train_entry = self._resolve_train_entry()

        if launch_mode == "single":
            cmd = [*train_entry, *train_args]
        elif launch_mode == "torchrun":
            cmd = [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={nproc_per_node}",
                *train_entry,
                *train_args,
            ]
        else:
            raise ValueError("launch_mode must be 'single' or 'torchrun'.")

        subprocess.run(cmd, check=True)

    def select_replay_dataset(
        self,
        configs_pt: str | Path,
        configs_ft: str | Path,
        output: str | Path,
        model: str | Path,
        num_samples: int,
        subselect: str | None = None,
        filtering_type: str | None = None,
        atomic_numbers: Sequence[int] | None = None,
        head_pt: str | None = None,
        head_ft: str | None = None,
        weight_pt: float | None = None,
        weight_ft: float | None = None,
    ) -> Path:
        cmd = [
            *self._resolve_fine_tuning_select_entry(),
            "--configs_pt",
            str(configs_pt),
            "--configs_ft",
            str(configs_ft),
            "--num_samples",
            str(num_samples),
            "--model",
            str(model),
            "--output",
            str(output),
        ]

        self._extend_optional_args(
            cmd,
            ("--subselect", subselect),
            ("--filtering_type", filtering_type),
            ("--head_pt", head_pt),
            ("--head_ft", head_ft),
            ("--weight_pt", weight_pt),
            ("--weight_ft", weight_ft),
        )

        if atomic_numbers is not None:
            cmd.extend(["--atomic_numbers", str(list(atomic_numbers))])

        subprocess.run(cmd, check=True)
        return Path(output)

    def _merge_training_kwargs(
        self,
        valid_file: str | Path | None = None,
        test_file: str | Path | None = None,
        statistics_file: str | Path | None = None,
        work_dir: str | Path | None = None,
        hidden_irreps: str | None = None,
        atomic_numbers: Sequence[int] | None = None,
        config_type_weights: str | None = None,
        energy_key: str | None = None,
        forces_key: str | None = None,
        stress_key: str | None = None,
        E0s: str | Path | None = None,
        batch_size: int | None = None,
        valid_batch_size: int | None = None,
        max_num_epochs: int | None = None,
        valid_fraction: float | None = None,
        r_max: float | None = None,
        lr: float | None = None,
        ema: bool | None = None,
        ema_decay: float | None = None,
        loss: str | None = None,
        num_channels: int | None = None,
        max_L: int | None = None,
        num_workers: int | None = None,
        enable_cueq: bool | None = None,
        only_cueq: bool | None = None,
        distributed: bool = False,
        launch_mode: str = "single",
        nproc_per_node: int = 1,
        seed: int | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        kwargs = dict(self.training_kwargs)
        local_kwargs = {
            "valid_file": valid_file,
            "test_file": test_file,
            "statistics_file": statistics_file,
            "work_dir": work_dir,
            "hidden_irreps": hidden_irreps,
            "atomic_numbers": atomic_numbers,
            "config_type_weights": config_type_weights,
            "energy_key": energy_key,
            "forces_key": forces_key,
            "stress_key": stress_key,
            "E0s": E0s,
            "batch_size": batch_size,
            "valid_batch_size": valid_batch_size,
            "max_num_epochs": max_num_epochs,
            "valid_fraction": valid_fraction,
            "r_max": r_max,
            "lr": lr,
            "ema": ema,
            "ema_decay": ema_decay,
            "loss": loss,
            "num_channels": num_channels,
            "max_L": max_L,
            "num_workers": num_workers,
            "enable_cueq": enable_cueq,
            "only_cueq": only_cueq,
            "distributed": distributed,
            "launch_mode": launch_mode,
            "nproc_per_node": nproc_per_node,
            "seed": seed,
            "extra_args": extra_args,
        }
        kwargs.update(
            {key: value for key, value in local_kwargs.items() if value is not None}
        )
        return kwargs

    def _load_expected_checkpoint_if_available(self) -> None:
        expected_checkpoint = self.expected_checkpoint_path()
        if expected_checkpoint is not None and expected_checkpoint.exists():
            self.save(expected_checkpoint)

    def train(
        self,
        train_file: str | Path,
        valid_file: str | Path | None = None,
        test_file: str | Path | None = None,
        statistics_file: str | Path | None = None,
        work_dir: str | Path | None = None,
        hidden_irreps: str | None = None,
        atomic_numbers: Sequence[int] | None = None,
        config_type_weights: str | None = None,
        energy_key: str | None = None,
        forces_key: str | None = None,
        stress_key: str | None = None,
        E0s: str | Path | None = None,
        batch_size: int | None = None,
        valid_batch_size: int | None = None,
        max_num_epochs: int | None = None,
        valid_fraction: float | None = None,
        r_max: float | None = None,
        lr: float | None = None,
        ema: bool | None = None,
        ema_decay: float | None = None,
        loss: str | None = None,
        num_channels: int | None = None,
        max_L: int | None = None,
        num_workers: int | None = None,
        enable_cueq: bool | None = None,
        only_cueq: bool | None = None,
        distributed: bool = False,
        launch_mode: str = "single",
        nproc_per_node: int = 1,
        seed: int | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> None:
        kwargs = self._merge_training_kwargs(
            valid_file=valid_file,
            test_file=test_file,
            statistics_file=statistics_file,
            work_dir=work_dir,
            hidden_irreps=hidden_irreps,
            atomic_numbers=atomic_numbers,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            E0s=E0s,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            max_num_epochs=max_num_epochs,
            valid_fraction=valid_fraction,
            r_max=r_max,
            lr=lr,
            ema=ema,
            ema_decay=ema_decay,
            loss=loss,
            num_channels=num_channels,
            max_L=max_L,
            num_workers=num_workers,
            enable_cueq=enable_cueq,
            only_cueq=only_cueq,
            distributed=distributed,
            launch_mode=launch_mode,
            nproc_per_node=nproc_per_node,
            seed=seed,
            extra_args=extra_args,
        )

        train_args = self._build_train_args(
            train_file=train_file,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"launch_mode", "nproc_per_node"}
            },
        )
        self._run_training_command(
            train_args=train_args,
            launch_mode=kwargs.get("launch_mode", "single"),
            nproc_per_node=kwargs.get("nproc_per_node", 1),
        )
        self._load_expected_checkpoint_if_available()

    def finetune(
        self,
        train_file: str | Path,
        foundation_model: str | Path,
        valid_file: str | Path | None = None,
        test_file: str | Path | None = None,
        statistics_file: str | Path | None = None,
        multiheads_finetuning: bool = False,
        distributed: bool = False,
        launch_mode: str = "single",
        nproc_per_node: int = 1,
        extra_args: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # mace_run_train defaults --multiheads_finetuning to True, which (for
        # keyword foundation models like "medium") silently turns a plain
        # finetune into multihead finetuning unless disabled explicitly here.
        train_args = self._build_train_args(
            train_file=train_file,
            valid_file=valid_file,
            test_file=test_file,
            foundation_model=foundation_model,
            statistics_file=statistics_file,
            multiheads_finetuning=multiheads_finetuning,
            distributed=distributed,
            extra_args=extra_args,
            **kwargs,
        )
        self._run_training_command(
            train_args=train_args,
            launch_mode=launch_mode,
            nproc_per_node=nproc_per_node,
        )

    def multihead_finetune(
        self,
        train_file: str | Path,
        foundation_model: str | Path,
        valid_file: str | Path | None = None,
        test_file: str | Path | None = None,
        pt_train_file: str | Path | None = None,
        pt_valid_file: str | Path | None = None,
        method: str = "direct",
        selected_pt_output: str | Path | None = None,
        num_samples_pt: int | None = None,
        subselect_pt: str | None = None,
        filter_type_pt: str | None = None,
        atomic_numbers: Sequence[int] | None = None,
        weight_pt: float | None = None,
        weight_ft: float | None = None,
        head_pt: str | None = None,
        head_ft: str | None = None,
        statistics_file: str | Path | None = None,
        force_mh_ft_lr: bool | None = None,
        distributed: bool = False,
        launch_mode: str = "single",
        nproc_per_node: int = 1,
        extra_args: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        replay_file: str | Path | None = pt_train_file

        if method == "preprocess":
            if pt_train_file is None:
                raise ValueError("pt_train_file is required for preprocess mode.")
            if selected_pt_output is None:
                raise ValueError("selected_pt_output is required for preprocess mode.")
            if num_samples_pt is None:
                raise ValueError("num_samples_pt is required for preprocess mode.")

            replay_file = self.select_replay_dataset(
                configs_pt=pt_train_file,
                configs_ft=train_file,
                output=selected_pt_output,
                model=foundation_model,
                num_samples=num_samples_pt,
                subselect=subselect_pt,
                filtering_type=filter_type_pt,
                atomic_numbers=atomic_numbers,
                head_pt=head_pt,
                head_ft=head_ft,
                weight_pt=weight_pt,
                weight_ft=weight_ft,
            )
        elif method == "mp":
            replay_file = "mp"
        elif method == "direct":
            if pt_train_file is None:
                raise ValueError("pt_train_file is required for direct mode.")
        else:
            raise ValueError("method must be 'direct', 'preprocess', or 'mp'.")

        train_args = self._build_train_args(
            train_file=train_file,
            valid_file=valid_file,
            test_file=test_file,
            foundation_model=foundation_model,
            pt_train_file=replay_file,
            pt_valid_file=pt_valid_file,
            statistics_file=statistics_file,
            atomic_numbers=atomic_numbers,
            num_samples_pt=num_samples_pt if method == "direct" else None,
            subselect_pt=subselect_pt if method == "direct" else None,
            filter_type_pt=filter_type_pt if method == "direct" else None,
            weight_pt=weight_pt,
            multiheads_finetuning=True,
            force_mh_ft_lr=force_mh_ft_lr,
            distributed=distributed,
            extra_args=extra_args,
            **kwargs,
        )
        self._run_training_command(
            train_args=train_args,
            launch_mode=launch_mode,
            nproc_per_node=nproc_per_node,
        )

    def lora_finetune(
        self,
        train_file: str | Path,
        foundation_model: str | Path,
        valid_file: str | Path | None = None,
        test_file: str | Path | None = None,
        statistics_file: str | Path | None = None,
        lora_rank: int | None = None,
        lora_alpha: float | None = None,
        multiheads_finetuning: bool = False,
        distributed: bool = False,
        launch_mode: str = "single",
        nproc_per_node: int = 1,
        extra_args: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # mace_run_train defaults --multiheads_finetuning to True, which (for
        # keyword foundation models like "medium") silently turns plain LoRA
        # finetuning into multihead finetuning unless disabled explicitly here.
        train_args = self._build_train_args(
            train_file=train_file,
            valid_file=valid_file,
            test_file=test_file,
            foundation_model=foundation_model,
            statistics_file=statistics_file,
            lora=True,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            multiheads_finetuning=multiheads_finetuning,
            distributed=distributed,
            extra_args=extra_args,
            **kwargs,
        )
        self._run_training_command(
            train_args=train_args,
            launch_mode=launch_mode,
            nproc_per_node=nproc_per_node,
        )

    def predict(self, atoms: Any) -> dict[str, Any]:
        if self.calculator is None:
            raise RuntimeError("Model must be loaded before prediction.")

        atoms.calc = self.calculator
        result = {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
        }

        try:
            result["stress"] = atoms.get_stress()
        except Exception:
            result["stress"] = None

        return result

    def save(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._build_calculator()

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        default_dtype: str = "float64",
        head: str | None = None,
    ) -> "MACEModel":
        return cls(
            checkpoint_path=checkpoint_path,
            device=device,
            default_dtype=default_dtype,
            head=head,
        )
