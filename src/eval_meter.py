# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Evaluation of model performance.
# pylint: disable= no-member, arguments-differ, invalid-name

# Modified: removed long documentation; preserve sparse metric positions.
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)

__all__ = ["Meter", "performance_summary"]


class Meter(object):
    def __init__(self, mean=None, std=None):
        self.mask = []
        self.y_pred = []
        self.y_true = []
        if mean is not None and std is not None:
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None

    def update(self, y_pred, y_true, mask=None):
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self.mask.append(torch.ones(self.y_pred[-1].shape))
        else:
            self.mask.append(mask.detach().cpu())

    def _finalize(self):
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        if self.mean is not None and self.std is not None:
            y_pred = y_pred * self.std + self.mean
        return (mask, y_pred, y_true)

    def _reduce_scores(self, scores, reduction="none"):
        if reduction == "none":
            return scores
        elif reduction == "mean":
            valid_scores = [s for s in scores if not np.isnan(s)]
            if len(valid_scores) == 0:
                return float("nan")
            return np.mean(valid_scores)
        elif reduction == "sum":
            return np.sum(scores)
        else:
            raise ValueError(f"Expected reduction to be 'none', 'mean' or 'sum', got {reduction}")

    def multilabel_score(self, score_func, reduction="none"):
        mask, y_pred, y_true = self._finalize()
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            if len(task_y_true) < 2 or len(task_y_pred) < 2:
                scores.append(float("nan"))
                continue
            task_score = score_func(task_y_true, task_y_pred)
            scores.append(float("nan") if task_score is None else task_score)
        return self._reduce_scores(scores, reduction)

    def pcc(self, reduction="none"):
        def score(y_true, y_pred):
            if len(y_true) < 2 or len(y_pred) < 2:
                return None
            return -pearsonr(y_true.numpy(), y_pred.numpy())[0]

        return self.multilabel_score(score, reduction)

    def pearson_r2(self, reduction="none"):
        def score(y_true, y_pred):
            if len(y_true) < 2 or len(y_pred) < 2:
                return None
            return pearsonr(y_true.numpy(), y_pred.numpy())[0] ** 2

        return self.multilabel_score(score, reduction)

    def mae(self, reduction="none"):
        def score(y_true, y_pred):
            return F.l1_loss(y_true, y_pred).data.item()

        return self.multilabel_score(score, reduction)

    def rmse(self, reduction="none"):
        def score(y_true, y_pred):
            return torch.sqrt(F.mse_loss(y_pred, y_true).cpu()).item()

        return self.multilabel_score(score, reduction)

    def roc_auc_score(self, reduction="none"):
        assert self.mean is None and self.std is None, (
            "Label normalization should not be performed for binary classification."
        )

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                return None
            else:
                return roc_auc_score(y_true.long().numpy(), torch.sigmoid(y_pred).numpy())

        return self.multilabel_score(score, reduction)

    def pr_auc_score(self, reduction="none"):
        assert self.mean is None and self.std is None, (
            "Label normalization should not be performed for binary classification."
        )

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                return None
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true.long().numpy(), torch.sigmoid(y_pred).numpy()
                )
                return auc(recall, precision)

        return self.multilabel_score(score, reduction)

    def accuracy_score(self, reduction="none"):
        assert self.mean is None and self.std is None, (
            "Label normalization should not be performed for binary classification."
        )

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                return None
            else:
                y_true = y_true.detach().cpu().long().numpy().tolist()
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                y_pred = np.array(pred_y)
                return accuracy_score(y_true, y_pred)

        return self.multilabel_score(score, reduction)

    def balanced_accuracy(self, reduction="none"):
        assert self.mean is None and self.std is None, (
            "Label normalization should not be performed for binary classification."
        )

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                return None
            else:
                y_true = y_true.detach().cpu().long().numpy().tolist()
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                y_pred = np.array(pred_y)
                return balanced_accuracy_score(y_true, y_pred)

        return self.multilabel_score(score, reduction)

    def precision_score(self, reduction="none"):
        assert self.mean is None and self.std is None, (
            "Label normalization should not be performed for binary classification."
        )

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                return None
            else:
                y_true = y_true.detach().cpu().long().numpy().tolist()
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                y_pred = np.array(pred_y)
                return precision_score(y_true, y_pred, zero_division=0)

        return self.multilabel_score(score, reduction)

    def recall_score(self, reduction="none"):
        assert self.mean is None and self.std is None, (
            "Label normalization should not be performed for binary classification."
        )

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                return None
            else:
                y_true = y_true.detach().cpu().long().numpy().tolist()
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                y_pred = np.array(pred_y)
                return recall_score(y_true, y_pred)

        return self.multilabel_score(score, reduction)

    def f1_score(self, reduction="none"):
        assert self.mean is None and self.std is None, (
            "Label normalization should not be performed for binary classification."
        )

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                return None
            else:
                y_true = y_true.detach().cpu().long().numpy().tolist()
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                y_pred = np.array(pred_y)
                return f1_score(y_true, y_pred)

        return self.multilabel_score(score, reduction)

    def specificity_score(self, reduction="none"):
        assert self.mean is None and self.std is None, (
            "Label normalization should not be performed for binary classification."
        )

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                return None
            else:
                y_true = y_true.detach().cpu().long().numpy().tolist()
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                y_pred = np.array(pred_y)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                return cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])

        return self.multilabel_score(score, reduction)

    def compute_metric(self, metric_name, reduction="none"):
        if metric_name == "r2":
            return self.pearson_r2(reduction)
        elif metric_name == "pcc":
            return self.pcc(reduction)
        elif metric_name == "mae":
            return self.mae(reduction)
        elif metric_name == "rmse":
            return self.rmse(reduction)
        elif metric_name == "roc_auc_score":
            return self.roc_auc_score(reduction)
        elif metric_name == "pr_auc_score":
            return self.pr_auc_score(reduction)
        elif metric_name == "precision":
            return self.precision_score(reduction)
        elif metric_name == "recall":
            return self.recall_score(reduction)
        elif metric_name == "accuracy":
            return self.accuracy_score(reduction)
        elif metric_name == "bacc":
            return self.balanced_accuracy(reduction)
        elif metric_name == "f1":
            return self.f1_score(reduction)
        elif metric_name == "specificity":
            return self.specificity_score(reduction)
        else:
            raise ValueError(
                'Expect metric_name to be "r2" or "mae" or "rmse" or "roc_auc_score" or "pr_auc", got {}'.format(
                    metric_name
                )
            )


def performance_summary(values, metric_name, task_names):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(task_names) != len(values):
        raise ValueError("Task names and metric values must have equal lengths")
    schema_path = Path(__file__).resolve().parents[1] / "configs/task_schema.json"
    with schema_path.open(encoding="utf-8") as handle:
        fixed_tiers = json.load(handle)["performance_tiers"]
    tiers = np.asarray([fixed_tiers[task] for task in task_names])
    summaries = []
    for name in ["Robust", "Intermediate", "Difficult", "Novel"]:
        scores = values[(tiers == name) & np.isfinite(values)]
        mean = np.mean(scores) if scores.size else float("nan")
        summaries.append(f"{name}_{metric_name}={mean:.4f}")
    return " ".join(summaries)
