import json
import shutil
import sys

from pathlib import Path

import optuna
import torch
import torch.serialization

from torch.utils.tensorboard import SummaryWriter

from scxpand.mlp.mlp_params import MLPParam
from scxpand.util.general_util import (
    flatten_nested_dict,
    get_device,
    get_elapsed_time_str,
    get_last_git_commit_link,
    get_local_time,
    metrics_dict_to_table,
    nested_dict_to_flat_str,
    nested_dict_to_multiline_str,
    time_seconds_to_str,
    time_to_str,
)
from scxpand.util.logger import get_logger
from scxpand.util.model_constants import (
    BEST_CHECKPOINT_FILE,
    BEST_MODEL_INFO_FILE,
    BEST_MODEL_METRICS_FILE,
    LAST_CHECKPOINT_FILE,
    RUN_INFO_FILE,
    SUMMARY_INFO_FILE,
)


logger = get_logger()


class TrainLogger:
    def __init__(
        self,
        save_path: Path,
        resume_exp_path: Path | None = None,
        trial: optuna.Trial | None = None,
    ):
        torch.serialization.add_safe_globals(
            {
                "torch.optim.optimizer.Optimizer": torch.optim.Optimizer,
                "torch.optim.lr_scheduler._LRScheduler": torch.optim.lr_scheduler._LRScheduler,
                "collections.OrderedDict": dict,
            }
        )

        if resume_exp_path:
            save_path = Path(resume_exp_path)
            logger.info(f"Resuming training from checkpoint {resume_exp_path}.")
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving results to {self.save_path}.")
        self.trial_number = trial.number if trial else None
        self.best_model_epoch = 0
        self.start_time = get_local_time()
        self.run_info = {
            "run_script": sys.argv[0],
            "run_args": sys.argv[1:],
            "start_time": time_to_str(self.start_time),
            "last_git_commit": get_last_git_commit_link(),
        }
        with (self.save_path / RUN_INFO_FILE).open("w") as f:
            json.dump(self.run_info, f, indent=4)

    def init_writer(self, n_epochs: int, n_train_batches: int, prm: MLPParam):
        self.writer = SummaryWriter(log_dir=self.save_path)
        self.n_train_batches = n_train_batches
        self.n_epochs = n_epochs
        self.n_total_steps = n_epochs * n_train_batches
        self.best_model_score = None  # The best score value of a checkpoint so far
        self.best_model_metrics = {}
        self.writer.add_text("run_info", nested_dict_to_multiline_str(self.run_info))
        self.writer.add_text("parameters", nested_dict_to_multiline_str(prm.__dict__))
        self.save_path.mkdir(parents=True, exist_ok=True)

    def add_scalars(
        self,
        scalars: dict,
        group: str,
        global_step: int,
        epoch: int | None = None,
        i_batch: int | None = None,
        one_line: bool = False,
    ):
        scalars_flat = flatten_nested_dict(scalars)
        for key, value in scalars_flat.items():
            self.writer.add_scalar(tag=f"{group}/{key}", scalar_value=value, global_step=global_step)

        completed_percent = 100 * global_step / self.n_total_steps
        total_seconds_elapsed = self.get_elapsed_time_in_seconds()
        if global_step > 0:
            seconds_per_step = total_seconds_elapsed / global_step
            total_expected_seconds = seconds_per_step * self.n_total_steps
            remaining_seconds = total_expected_seconds - total_seconds_elapsed
            remaining_seconds_str = time_seconds_to_str(remaining_seconds)
        else:
            remaining_seconds_str = "unknown"

        print_str = f"({completed_percent:.1f}%) ({group}),"
        if self.trial_number is not None:
            print_str = f"(trial {self.trial_number}) " + print_str
        if epoch is not None:
            print_str += f" epoch {epoch + 1}/{self.n_epochs},"
        if i_batch is not None:
            print_str += f" batch {i_batch + 1}/{self.n_train_batches},"
        if one_line:
            print_str += f" {nested_dict_to_flat_str(scalars, omit_keys=['epoch', 'batch'])}"
        else:
            print_str += "\n" + metrics_dict_to_table(scalars, title="Training Metrics")
        print_str += f", elapsed: {time_seconds_to_str(total_seconds_elapsed)}, remaining: {remaining_seconds_str}"
        logger.info(print_str)

    def flush(self):
        self.writer.flush()

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        epoch: int,
        model_score: float,
        dev_set_metrics: dict | None = None,
    ):
        self.best_model_metrics = dev_set_metrics

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": (lr_scheduler.state_dict() if lr_scheduler else None),
            "epoch": epoch,
            "model_score": model_score,
            "best_model_score": self.best_model_score,
            "best_model_epoch": self.best_model_epoch,
            "dev_set_metrics": dev_set_metrics,
        }
        ckpt_save_path = self.save_path / LAST_CHECKPOINT_FILE
        torch.save(checkpoint, ckpt_save_path)
        logger.info(f"Saved checkpoint to {ckpt_save_path}")

        if self.best_model_score is None or model_score > self.best_model_score:
            self.best_model_score = model_score
            self.best_model_epoch = epoch
            logger.info(f"New best score: {model_score:.3f}")
            best_ckpt_path = self.save_path / BEST_CHECKPOINT_FILE
            shutil.copy(ckpt_save_path, best_ckpt_path)
            logger.info(f"Saved checkpoint to {best_ckpt_path}")

            if dev_set_metrics:
                json_path = self.save_path / BEST_MODEL_METRICS_FILE
                with json_path.open("w") as f:
                    json.dump(dev_set_metrics, f, indent=4)

            model_info = {
                "model_score": model_score,
                "epoch": epoch,
                "dev_set_metrics": dev_set_metrics,
            }
            with (self.save_path / BEST_MODEL_INFO_FILE).open("w") as f:
                json.dump(model_info, f, indent=4)

    def load_best_model(
        self,
        model: torch.nn.Module,
        device_name: str | None = None,
    ):
        device_name = device_name or get_device()
        model.to(device_name)
        best_ckpt_path = self.save_path / BEST_CHECKPOINT_FILE
        if best_ckpt_path.exists():
            checkpoint_info = torch.load(
                best_ckpt_path,
                map_location=device_name,
                weights_only=False,
            )
            model.load_state_dict(checkpoint_info["model_state_dict"])
            logger.info(f"Loaded best model from {self.save_path / BEST_CHECKPOINT_FILE}")
        else:
            logger.warning(f"No best model checkpoint found in {best_ckpt_path}, using the current model.")
        return model

    def resume_from_checkpoint(
        self,
        resume_exp_path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device_name: str = "cuda",
    ) -> int:
        model.to(device_name)
        ckpt_dir = Path(resume_exp_path)
        checkpoint_path = ckpt_dir / LAST_CHECKPOINT_FILE

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device_name,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if lr_scheduler and checkpoint["lr_scheduler_state_dict"]:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self.best_model_score = checkpoint.get("best_model_score")
        self.best_model_epoch = checkpoint.get("best_model_epoch")
        epoch = checkpoint["epoch"]

        logger.info(f"Resumed from checkpoint {resume_exp_path} that was saved at the end of epoch {epoch}")
        if self.best_model_score is not None:
            logger.info(f"Best score: {self.best_model_score:.3f} was achieved at epoch {self.best_model_epoch}")
        else:
            logger.info("No best model score found, starting from scratch.")
        resume_epoch = epoch + 1
        logger.info(f"Resuming training from epoch {resume_epoch}...")
        return resume_epoch

    def save_final_summary(self):
        self.run_info["best_model_metrics"] = self.best_model_metrics
        self.run_info["best_model_score"] = self.best_model_score
        self.run_info["best_model_epoch"] = self.best_model_epoch
        logger.info(
            f"Best model score: {self.best_model_score:.3f} at epoch {self.best_model_epoch}",
        )
        logger.info(f"Saving run summary to {self.save_path}")
        time_now = get_local_time()
        self.run_info["finish_time"] = time_to_str(time_now)
        self.run_info["elapsed_time"] = get_elapsed_time_str(self.start_time, time_now)
        logger.info(f"Elapsed time: {self.run_info['elapsed_time']}")
        with (self.save_path / SUMMARY_INFO_FILE).open("w") as f:
            json.dump(self.run_info, f, indent=4)

    def get_elapsed_time_in_seconds(self):
        return (get_local_time() - self.start_time).total_seconds()


def has_checkpoint(path: Path | str):
    path = Path(path)
    return path.exists() and (path / LAST_CHECKPOINT_FILE).exists()
