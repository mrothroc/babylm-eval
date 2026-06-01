from __future__ import annotations

from tqdm import tqdm
from typing import TYPE_CHECKING, Any
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
import copy
import json
import warnings
import pathlib
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from torch.optim import AdamW

from evaluation_pipeline.finetune.classifier_model import ModelForSequenceClassification
from evaluation_pipeline.finetune.dataset import Dataset, PredictDataset
from evaluation_pipeline.finetune.utils import cosine_schedule_with_warmup

if TYPE_CHECKING:
    from argparse import Namespace
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import wandb


def _load_labeled_dataset(data_path: pathlib.Path, batch_size: int, tokenizer: PreTrainedTokenizerBase, shuffle: bool, drop_last: bool, args: Namespace) -> DataLoader:
    dataset = Dataset(data_path, args.task)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(dataset.collate_function, tokenizer, args.three_d_triangular_causal_mask, args.sequence_length), shuffle=shuffle, drop_last=drop_last)

    return dataloader


def _load_predict_dataset(data_path: pathlib.Path, batch_size: int, tokenizer: PreTrainedTokenizerBase, args: Namespace):
    dataset = PredictDataset(data_path, args.task)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(dataset.collate_function, tokenizer, args.three_d_triangular_causal_mask, args.sequence_length))

    return dataloader


class Trainer():

    def __init__(self: Trainer, args: Namespace, device: torch.device) -> None:
        """The Trainer class handles all the fine tuning,
        evaluation, and prediction of a given task for a
        given model.

        Args:
            args(Namespace): The config information such
                as hyperparameters, directories, verbose,
                model_name, optimizer_name, etc.
            device(torch.device): The device to use for
                finetuning.
        """
        self.args: Namespace = args
        self.device: torch.device = device
        self._init_model()
        self.load_data()
        self.global_step: int = 0
        self.phase_step: int = 0
        self.steps_per_epoch: int = len(self.train_dataloader) // self.args.gradient_accumulation
        self.total_steps = (len(self.train_dataloader) // self.args.gradient_accumulation) * self.args.num_epochs
        self.phase_total_steps: int = self.total_steps
        self.ema_active: bool = True
        self._init_opitmizer()
        self._init_scheduler()
        if args.wandb:
            self._init_wandb()

    def _init_wandb(self: Trainer) -> None:
        self.wandb_run = wandb.init(
            name=self.args.exp_name,
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            config=self.args
        )

    def _init_model(self: Trainer) -> None:
        self.model = ModelForSequenceClassification(self.args)
        self.ema_model: nn.Module = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        self.ema_model.to(self.device)
        self.tokenizer: PreTrainedTokenizerBase = AutoProcessor.from_pretrained(self.args.model_name_or_path, trust_remote_code=True, padding_side=self.args.padding_side)

    def load_data(self: Trainer) -> None:
        """This function loads the data and creates the
        dataloader for each split of the data.
        """
        assert self.args.batch_size % self.args.gradient_accumulation == 0, f"The gradient accumualtion {self.args.gradient_accumulation} should divide the batch size {self.args.batch_size}."

        self.train_dataloader: DataLoader = _load_labeled_dataset(self.args.train_data, self.args.batch_size // self.args.gradient_accumulation, self.tokenizer, True, True, self.args)

        self.valid_dataloader: DataLoader | None = None
        if self.args.valid_data is not None:
            self.valid_dataloader: DataLoader = _load_labeled_dataset(self.args.valid_data, self.args.valid_batch_size, self.tokenizer, False, False, self.args)

        self.predict_dataloader: DataLoader | None = None
        if self.args.predict_data is not None:
            self.predict_dataloader: DataLoader = _load_predict_dataset(self.args.predict_data, self.args.valid_batch_size, self.tokenizer, self.args)

    def _optimizer_kwargs(self: Trainer) -> dict[str, float | tuple[float, float] | bool]:
        return {
            "betas": (self.args.beta1, self.args.beta2),
            "eps": self.args.optimizer_eps,
            "weight_decay": self.args.weight_decay,
            "amsgrad": self.args.amsgrad
        }

    def _head_parameters(self: Trainer) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = list(self.model.classifier.parameters())
        if self.args.take_final and self.args.three_d_triangular_causal_mask:
            for module_name in ("final_position_projection", "final_projection", "position_projection"):
                module = getattr(self.model, module_name, None)
                if module is not None:
                    parameters.extend(list(module.parameters()))
        return parameters

    def _set_encoder_trainable(self: Trainer, trainable: bool) -> None:
        self.model.transformer.requires_grad_(trainable)
        for parameter in self._head_parameters():
            parameter.requires_grad = True

    def _init_opitmizer(self: Trainer) -> None:
        self._set_encoder_trainable(True)
        if self.args.optimizer in ["adamw", "adam"]:
            self.optimizer: Optimizer = AdamW(
                [
                    {"params": self.model.transformer.parameters(), "lr": self.args.encoder_lr},
                    {"params": self._head_parameters(), "lr": self.args.head_lr}
                ],
                **self._optimizer_kwargs()
            )
        else:
            raise NotImplementedError(f"The optimizer {self.args.optimizer} is not implemented!")

    def _init_head_only_optimizer(self: Trainer) -> None:
        self._set_encoder_trainable(False)
        if self.args.optimizer in ["adamw", "adam"]:
            self.optimizer = AdamW(self._head_parameters(), lr=self.args.head_lr, **self._optimizer_kwargs())
        else:
            raise NotImplementedError(f"The optimizer {self.args.optimizer} is not implemented!")

    def _init_scheduler(self: Trainer, total_steps: int | None = None) -> None:
        total_steps = self.total_steps if total_steps is None else total_steps
        if self.args.scheduler == "cosine":
            self.scheduler: LRScheduler | None = cosine_schedule_with_warmup(self.optimizer, int(self.args.warmup_proportion * total_steps), total_steps, self.args.min_factor)
        elif self.args.scheduler == "none":
            self.scheduler = None
        else:
            raise NotImplementedError(f"The scheduler {self.args.scheduler} is not implemented!")

    # TODO: Create getter and setter functions

    def reset_trainer(self: Trainer) -> None:
        """This function resets the Trainer. This means that it
        resets the global step back to zero, and re-initializes
        the optimizer and scheduler."""
        self.global_step = 0
        self._init_opitmizer()
        self._init_scheduler()

    def _prepare_phase(self: Trainer, phase: str, num_epochs: int) -> None:
        self.phase_step = 0
        self.phase_total_steps = self.steps_per_epoch * num_epochs
        if phase == "head_only":
            self.ema_active = False
            self._init_head_only_optimizer()
        elif phase == "joint":
            self.ema_active = True
            self._set_encoder_trainable(True)
            self._sync_ema_model()
            self._init_opitmizer()
        else:
            raise ValueError(f"Unknown training phase: {phase}")
        self._init_scheduler(self.phase_total_steps)

    def _sync_ema_model(self: Trainer) -> None:
        if self.ema_model is None:
            return
        self.ema_model.load_state_dict(self.model.state_dict())

    def _current_eval_model(self: Trainer) -> nn.Module:
        if self.ema_model is not None and self.ema_active:
            return self.ema_model
        return self.model

    def train_epoch(self: Trainer) -> float:
        """This function does a single epoch of the training.

        Args:
            total_steps(int): The total number of finetuning
                steps to do. Used for the progress bar and in
                case the finetuning needs to be stoped mid
                epoch.
            global_step(int): The current step the finetuning
                is on.

        Returns:
            int: The current step the model is on at the end of
                the epoch.
        """
        self.model.train()
        self.optimizer.zero_grad()

        progress_bar = tqdm(initial=self.phase_step, total=self.phase_total_steps)
        cummulator = 0
        train_loss_final = 0.0

        for input_data, attention_mask, labels in self.train_dataloader:
            input_data = input_data.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)
            labels = labels.to(device=self.device)

            logits = self.model(input_data, attention_mask)

            loss = F.cross_entropy(logits, labels)
            train_loss_final = float(loss.detach().cpu())
            loss.backward()
            cummulator += 1
            if cummulator < self.args.gradient_accumulation:
                continue
            cummulator = 0

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.ema_model is not None and self.ema_active:
                with torch.no_grad():
                    for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
                        param_k.data.mul_(self.args.ema_decay).add_((1.0 - self.args.ema_decay) * param_q.detach().data)

            metrics = self.calculate_metrics(logits, labels, self.args.metrics)

            if hasattr(self, "wandb_run"):
                self.wandb_run.log(
                    {f"train/{metric}": value for metric, value in metrics.items()},
                    step=self.global_step
                )

            metrics_string = ", ".join([f"{key}: {self._format_metric_value(value)}" for key, value in metrics.items()])

            progress_bar.update()

            if self.args.verbose:
                progress_bar.set_postfix_str(metrics_string)

            self.global_step += 1
            self.phase_step += 1
            self.optimizer.zero_grad()

        progress_bar.close()

        return train_loss_final

    @torch.no_grad()
    def evaluate(self: Trainer, evaluate_best_model: bool = False) -> dict[str, float]:
        """This function does an evaluation pass on the
        validation dataset.

        Returns:
            dict[str, float]: A dictionary of scores of the
                model on the validation dataset, based on
                the metrics to evaluate on.
        """
        assert self.valid_dataloader is not None, "No valid dataset to run evaluation on!"

        if getattr(self, "best_model", None) is not None and evaluate_best_model:
            model: nn.Module = self.best_model
        else:
            model = self._current_eval_model()
        model.eval()

        progress_bar = tqdm(total=len(self.valid_dataloader))

        labels = []
        logits = []

        for input_data, attention_mask, label in self.valid_dataloader:
            input_data = input_data.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)
            label = label.to(device=self.device)

            logit = model(input_data, attention_mask)

            logits.append(logit)
            labels.append(label)

            progress_bar.update()

        labels = torch.cat(labels, dim=0)
        logits = torch.cat(logits, dim=0)

        metrics = self.calculate_metrics(logits, labels, self.args.metrics)

        if hasattr(self, "wandb_run"):
            self.wandb_run.log(
                {f"evaluate/{metric}": value for metric, value in metrics.items()},
                step=self.global_step
            )

        progress_bar.close()

        if self.args.verbose:
            metrics_string = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
            print(metrics_string)

        return metrics

    def save_model(self: Trainer, model: nn.Module) -> None:
        """This function saves the passed model to a file. The
        directory is specified inside the arguments passed to
        the constructor of the class.

        Args:
            model(nn.Module): The model to save.
        """
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), self.args.save_path / "model.pt")

    def _compare_scores(self: Trainer, best: float, current: float, bigger_better: bool) -> bool:
        if best is None:
            return True
        else:
            if current > best and bigger_better:
                return True
            elif current < best and not bigger_better:
                return True
            return False

    @staticmethod
    def _format_metric_value(value: Any) -> str:
        if isinstance(value, (float, int, np.floating, np.integer)):
            return f"{float(value):.4f}"
        return str(value)

    @staticmethod
    def calculate_metrics(logits: torch.Tensor, labels: torch.Tensor, metrics_to_calculate: list[str]) -> dict[str, float | int | dict[int, int] | list[float]]:
        """This function calculates the metrics specified by
        the user. This is a static method and can be used
        without initializing a Trainer.

        Args:
            logits(torch.Tensor): A tensor of logits per class
                calculated by a model.
            labels(torch.Tensor): A tensor of correct labels
                for each element of the batch
            metrics_to_calculate(list[str]): A list of metrics
                to evaluate.

        Returns:
        dict[str, float]: a dictionary containing the scores of
            the model on the specified metrics.

        Shapes:
            - logits: :math:`(B, N)`
            - labels: :math:`(B)`, where each element is in
                :math:`[0, N-1]`
        """
        predictions = logits.argmax(dim=-1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        num_labels = logits.size(-1)
        metrics = dict()

        label_ids = list(range(num_labels))
        pred_counts = np.bincount(predictions, minlength=num_labels)
        label_counts = np.bincount(labels, minlength=num_labels)
        total_predictions = int(pred_counts.sum())
        majority_count = int(label_counts.max()) if label_counts.size > 0 else 0

        metrics["pred_dist"] = {label_id: int(pred_counts[label_id]) for label_id in label_ids}
        metrics["classes_seen"] = int(np.count_nonzero(pred_counts))
        metrics["max_pred_frac"] = float(pred_counts.max() / total_predictions) if total_predictions > 0 else 0.0
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions))
        metrics["macro_f1"] = float(f1_score(labels, predictions, labels=label_ids, average="macro", zero_division=0))
        metrics["per_class_f1"] = [float(value) for value in f1_score(labels, predictions, labels=label_ids, average=None, zero_division=0)]
        metrics["majority_baseline"] = float(majority_count / len(labels)) if len(labels) > 0 else 0.0
        metrics["mcc"] = float(matthews_corrcoef(labels, predictions))

        for metric in metrics_to_calculate:
            if metric == "f1":
                if num_labels == 2:
                    metrics["f1"] = float(f1_score(labels, predictions, zero_division=0))
                else:
                    metrics["f1"] = metrics["macro_f1"]
            elif metric == "accuracy":
                metrics["accuracy"] = float(accuracy_score(labels, predictions))
            elif metric == "mcc":
                metrics["mcc"] = float(metrics["mcc"])
            else:
                print(f"Metric {metric} is unknown / not implemented. It will be skipped!")

        return metrics

    def _collapse_gate_passed(self: Trainer, metrics: dict[str, Any]) -> bool:
        return metrics["classes_seen"] == self.args.num_labels and metrics["max_pred_frac"] < 0.95

    def _write_audit_entry(self: Trainer, audit_entry: dict[str, Any]) -> None:
        if not self.args.audit_log:
            return
        with (self.args.output_path / "audit.jsonl").open("a") as file:
            json.dump(audit_entry, file)
            file.write("\n")

    def _write_audit_summary(self: Trainer, audit_entry: dict[str, Any] | None, collapse_warning: bool) -> None:
        if not self.args.audit_log:
            return
        summary = copy.deepcopy(audit_entry) if audit_entry is not None else {}
        summary["collapse_warning"] = collapse_warning
        with (self.args.output_path / "audit_summary.json").open("w") as file:
            json.dump(summary, file, indent=2)

    def _init_audit_log(self: Trainer) -> None:
        if not self.args.audit_log:
            return
        (self.args.output_path / "audit.jsonl").unlink(missing_ok=True)
        (self.args.output_path / "audit_summary.json").unlink(missing_ok=True)

    def train(self: Trainer) -> None:
        """This function does the training based on the
        hyperparameters, model, optimizer, scheduler specified
        to the constructor of the class.
        """
        best_score: float | None = None
        fallback_best_score: float | None = None
        self.best_model: nn.Module | None = None
        fallback_best_model: nn.Module | None = None
        best_audit_entry: dict[str, Any] | None = None
        fallback_best_audit_entry: dict[str, Any] | None = None
        any_gate_passed = False
        epoch_index = 0

        self._init_audit_log()

        for phase, num_epochs in (("head_only", self.args.head_only_epochs), ("joint", self.args.joint_epochs)):
            if num_epochs <= 0:
                continue

            self._prepare_phase(phase, num_epochs)

            for _ in range(num_epochs):
                epoch_index += 1
                train_loss_final = self.train_epoch()
                valid_metrics: dict[str, Any] = {}
                collapse_gate_passed = False
                is_best = False
                update_best = False
                update_fallback = False

                if self.valid_dataloader is not None:
                    valid_metrics = self.evaluate()
                    collapse_gate_passed = self._collapse_gate_passed(valid_metrics)
                    score: float = valid_metrics[self.args.metric_for_valid]
                    candidate_model = self._current_eval_model()

                    if self._compare_scores(fallback_best_score, score, self.args.higher_is_better):
                        fallback_best_score = score
                        fallback_best_model = copy.deepcopy(candidate_model)
                        update_fallback = True

                    if collapse_gate_passed:
                        any_gate_passed = True
                        if self._compare_scores(best_score, score, self.args.higher_is_better):
                            best_score = score
                            is_best = True
                            if self.args.keep_best_model:
                                self.best_model = copy.deepcopy(candidate_model)
                                update_best = True

                audit_entry = {
                    "epoch": epoch_index,
                    "phase": phase,
                    "train_loss_final": train_loss_final,
                    "valid_metrics": valid_metrics,
                    "collapse_gate_passed": collapse_gate_passed,
                    "is_best": is_best
                }

                if update_fallback:
                    fallback_best_audit_entry = copy.deepcopy(audit_entry)
                if is_best:
                    best_audit_entry = copy.deepcopy(audit_entry)

                self._write_audit_entry(audit_entry)

                if self.args.save:
                    if self.args.keep_best_model and update_best:
                        self.save_model(self.best_model)
                    elif not self.args.keep_best_model:
                        self.save_model(self._current_eval_model())
                    elif self.ema_model is not None and self.ema_active:
                        self.save_model(self.ema_model)
                    elif self.best_model is None:
                        self.save_model(self.model)

        collapse_warning = self.valid_dataloader is not None and not any_gate_passed and fallback_best_model is not None
        if collapse_warning:
            best_audit_entry = copy.deepcopy(fallback_best_audit_entry)
            if best_audit_entry is not None:
                best_audit_entry["is_best"] = True
            warnings.warn("No validation epoch passed the collapse gate; falling back to the highest-metric checkpoint.", RuntimeWarning)
            if self.args.keep_best_model:
                self.best_model = fallback_best_model
                if self.args.save:
                    self.save_model(self.best_model)

        self._write_audit_summary(best_audit_entry, collapse_warning)

    @torch.no_grad()
    def predict_classification(self: Trainer) -> torch.Tensor:
        """This function creates predictions for the prediction
        dataset.

        Returns:
            dict[str, float]: A dictionary of scores of the
                model on the validation dataset, based on
                the metrics to evaluate on.
        """
        assert self.predict_dataloader is not None, "No predict dataset to predict on!"

        if getattr(self, "best_model", None) is not None:
            model: nn.Module = self.best_model
        else:
            model = self._current_eval_model()
        model.eval()

        progress_bar = tqdm(total=len(self.predict_dataloader))

        logits = []

        for input_data, attention_mask in self.predict_dataloader:
            input_data = input_data.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)

            logit = model(input_data, attention_mask)

            logits.append(logit)

            progress_bar.update()

        logits = torch.cat(logits, dim=0)
        preds = logits.argmax(dim=-1)

        progress_bar.close()

        return preds
