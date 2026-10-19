from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
import torch
import torch.nn as nn
import numpy as np
from eval_meter import Meter, performance_summary
import pandas as pd
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)
from torch.optim.lr_scheduler import ReduceLROnPlateau

n_tasks = 135
n_ctasks = 108


class TrainLearner(nn.Module):
    def __init__(self, model, normalizer=None):
        super(TrainLearner, self).__init__()
        self.net = model
        self.device = next(model.parameters()).device
        self.task_difficulties = self.__init__task_difficulties()
        self.n_tasks = len(self.task_difficulties)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.0006, weight_decay=0.0001, eps=1e-08
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.7, patience=12, min_lr=1e-05
        )
        self.classification_criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.regression_criterion = nn.HuberLoss(reduction="none", delta=1.5)
        self.normalizer = normalizer
        self.device = next(model.parameters()).device
        n_tasks = self.n_tasks
        self.register_buffer(
            "task_boost", torch.ones(n_tasks, device=self.device, dtype=torch.float32)
        )
        self.task_boost[121] = 1.2
        self.task_boost[123] = 1.2
        self.task_boost[125] = 1.5
        self.task_boost[126] = 1.3
        self.task_boost[127] = 1.3
        self.task_boost[130] = 1.4

    def __init__task_difficulties(self):
        difficulties = []
        classification_difficulties = [
            2.6,
            1.5,
            4.0,
            4.0,
            3.2,
            2.7,
            4.0,
            2.3,
            2.6,
            1.8,
            2.5,
            1.8,
            2.1,
            2.9,
            2.6,
            2.1,
            2.3,
            2.2,
            2.8,
            2.2,
            2.5,
            3.0,
            2.4,
            2.9,
            1.8,
            1.6,
            4.0,
            1.7,
            1.7,
            2.2,
            2.2,
            2.3,
            2.6,
            1.8,
            2.3,
            1.8,
            2.6,
            2.9,
            1.9,
            2.6,
            2.0,
            2.5,
            1.6,
            2.3,
            2.8,
            2.8,
            2.8,
            1.9,
            2.4,
            1.9,
            2.9,
            2.0,
            2.5,
            2.0,
            1.9,
            3.8,
            2.9,
            1.8,
            2.9,
            2.9,
            2.1,
            2.9,
            3.1,
            4.0,
            3.2,
            3.1,
            3.0,
            2.5,
            2.3,
            2.1,
            2.1,
            2.1,
            2.3,
            2.6,
            2.0,
            1.9,
            2.4,
            2.2,
            2.8,
            2.0,
            2.4,
            2.5,
            2.5,
            1.9,
            2.8,
            1.8,
            2.1,
            2.2,
            2.0,
            2.1,
            2.1,
            3.0,
            2.4,
            2.3,
            3.1,
            3.0,
            2.6,
            3.0,
            2.6,
            2.9,
            2.6,
            2.0,
            2.9,
            2.2,
            2.1,
            3.0,
            2.5,
            1.8,
        ]
        regression_difficulties = [
            2.3,
            2.5,
            3.2,
            4.0,
            4.0,
            4.0,
            3.6,
            3.2,
            3.7,
            3.2,
            4.0,
            3.8,
            3.2,
            3.0,
            2.8,
            2.1,
            2.1,
            3.2,
            2.1,
            2.1,
            2.3,
            2.2,
            3.0,
            4.2,
            4.2,
            4.2,
            4.2,
        ]
        difficulties.extend(classification_difficulties)
        difficulties.extend(regression_difficulties)
        return difficulties

    def get_difficulty_based_weights(self, epoch: int):
        import math
        import torch

        if getattr(self, "_w_epoch", None) == epoch and hasattr(self, "_w_cached"):
            return self._w_cached
        if not hasattr(self, "_task_diffs_buf"):
            td = torch.tensor(self.task_difficulties, dtype=torch.float32, device=self.device)
            self._task_diffs_buf = td
            self._mask_simple = td < 2.0
            self._mask_medium = (td >= 2.0) & (td < 3.0)
            self._mask_hard = (td >= 3.0) & (td < 4.0)
            self._mask_extra_hard = td >= 4.0
        tau = getattr(self, "schedule_tau", 80.0)
        s = 1.0 - math.exp(-float(epoch) / max(1.0, tau))

        def cos_interp(start, end, s_):
            return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * s_))

        w_s = cos_interp(1.05, 0.97, s)
        w_m = cos_interp(1.0, 1.08, s)
        w_h = cos_interp(0.98, 1.12, s)
        w_x = cos_interp(0.97, 1.18, s)
        w = torch.empty_like(self._task_diffs_buf)
        w[self._mask_simple] = w_s
        w[self._mask_medium] = w_m
        w[self._mask_hard] = w_h
        w[self._mask_extra_hard] = w_x
        tb = getattr(self, "task_boost", None)
        if isinstance(tb, torch.Tensor) and tb.numel() == w.numel():
            w.mul_(tb)
        w.div_(w.mean().detach().clamp_min(1e-08))
        w.clamp_(0.3, 3.0)
        if not hasattr(self, "_weights_ema"):
            self._weights_ema = w.detach().clone()
        else:
            self._weights_ema.mul_(0.9).add_(w.detach(), alpha=0.1)
        w_out = self._weights_ema / self._weights_ema.mean().detach().clamp_min(1e-08)
        self._w_cached = w_out
        return w_out

    def compute_regression_loss_with_normalization(self, logits_r, labels_r, masks_r):
        if self.normalizer is not None:
            eps = 0.001
            stds = (
                torch.tensor(self.normalizer.stds, device=logits_r.device, dtype=torch.float32)
                + eps
            )
            std_min, std_max = (stds.min(), stds.max())
            scaled = (stds - std_min) / (std_max - std_min + 1e-08)
            weights = 1.5 - scaled
            weights = weights / weights.mean()
            weights = torch.clamp(weights, min=0.5, max=2.0)
            base = self.regression_criterion(logits_r, labels_r)
            m = (masks_r != 0).float()
            weighted = base * m * weights.unsqueeze(0)
            denom = (m * weights.unsqueeze(0)).sum(dim=0).clamp_min(1.0)
            loss_r_ind = weighted.sum(dim=0) / denom
            if torch.any(torch.isnan(loss_r_ind)):
                print("Warning: NaN loss detected; falling back to the original loss calculation.")
                loss_r_ind = torch.mean(base * m, dim=0)
        else:
            loss_r_ind = torch.mean(
                self.regression_criterion(logits_r, labels_r) * (masks_r != 0).float(), dim=0
            )
        return loss_r_ind

    def compute_difficulty_aware_loss(self, epoch, logits, labels, masks):
        difficulty_weights = self.get_difficulty_based_weights(epoch)
        logits_c = logits[:, :n_ctasks]
        labels_c = labels[:, :n_ctasks]
        masks_c = masks[:, :n_ctasks]
        c_mask = (masks_c != 0).float()
        c_loss = self.classification_criterion(logits_c, labels_c)
        loss_c_ind = (c_loss * c_mask).sum(dim=0) / c_mask.sum(dim=0).clamp_min(1.0)
        weighted_loss_c = loss_c_ind * difficulty_weights[:n_ctasks]
        loss_c = weighted_loss_c.mean()
        logits_r = logits[:, n_ctasks:]
        labels_r = labels[:, n_ctasks:]
        masks_r = masks[:, n_ctasks:]
        if self.normalizer:
            loss_r_ind = self.compute_regression_loss_with_normalization(
                logits_r, labels_r, masks_r
            )
        else:
            r_mask = (masks_r != 0).float()
            r_loss = self.regression_criterion(logits_r, labels_r)
            loss_r_ind = (r_loss * r_mask).sum(dim=0) / r_mask.sum(dim=0).clamp_min(1.0)
        weighted_loss_r = loss_r_ind * difficulty_weights[n_ctasks:]
        loss_r = weighted_loss_r.mean()
        return (loss_c, loss_r, loss_c_ind, loss_r_ind)

    def run_a_train_epoch_mr(self, epoch, data_loader, n_epochs):
        self.net.train()
        losses = []
        losses_r = []
        losses_ind = []
        train_meter_r = Meter()
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg1, bg2, labels, masks = batch_data
            bg1 = bg1.to(self.device)
            bg2 = bg2.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)
            n_feats1 = bg1.ndata.pop("hv").to(self.device)
            e_feats1 = bg1.edata.pop("he").to(self.device)
            n_feats2 = bg2.ndata.pop("hv").to(self.device)
            e_feats2 = bg2.edata.pop("he").to(self.device)
            logits = self.net([[bg1, n_feats1, e_feats1], [bg2, n_feats2, e_feats2]])
            n_ctasks = 108
            logits_r = logits[:, n_ctasks:]
            labels_r = labels[:, n_ctasks:]
            masks_r = masks[:, n_ctasks:]
            loss_c, loss_r, loss_c_ind, loss_r_ind = self.compute_difficulty_aware_loss(
                epoch, logits, labels, masks
            )
            loss = 0.25 * loss_c + loss_r
            loss_ind = torch.cat((loss_c_ind, loss_r_ind), dim=0)
            loss_ind = torch.flatten(loss_ind)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5)
            self.optimizer.step()
            train_meter_r.update(logits_r, labels_r, masks_r)
            losses.append(loss.data.item())
            losses_r.append(loss_r.data.item())
            losses_ind.append(loss_ind.tolist())
        total_r2 = np.nanmean(train_meter_r.compute_metric("r2"))
        total_loss_r = np.mean(losses_r)
        total_loss = np.mean(losses)
        total_loss_ind = np.mean(losses_ind, axis=0)
        return (total_r2, total_loss, total_loss_r, total_loss_ind.tolist())

    def run_a_valid_epoch_mr(self, epoch, data_loader, n_epochs):
        self.net.eval()
        val_losses = []
        val_losses_r = []
        val_losses_ind = []
        eval_meter_r = Meter()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):
                smiles, bg1, bg2, vali_labels, vali_masks = batch_data
                bg1 = bg1.to(self.device)
                bg2 = bg2.to(self.device)
                n_feats1 = bg1.ndata.pop("hv").to(self.device)
                e_feats1 = bg1.edata.pop("he").to(self.device)
                n_feats2 = bg2.ndata.pop("hv").to(self.device)
                e_feats2 = bg2.edata.pop("he").to(self.device)
                vali_labels = vali_labels.to(self.device)
                vali_masks = vali_masks.to(self.device)
                vali_logits = self.net([[bg1, n_feats1, e_feats1], [bg2, n_feats2, e_feats2]])
                vali_logits_r = vali_logits[:, n_ctasks:]
                vali_labels_r = vali_labels[:, n_ctasks:]
                vali_masks_r = vali_masks[:, n_ctasks:]
                vali_loss_c, vali_loss_r, vali_loss_c_ind, vali_loss_r_ind = (
                    self.compute_difficulty_aware_loss(epoch, vali_logits, vali_labels, vali_masks)
                )
                vali_loss_ind = torch.cat((vali_loss_c_ind, vali_loss_r_ind), dim=0)
                vali_loss_ind = torch.flatten(vali_loss_ind)
                vali_loss = vali_loss_r
                val_losses.append(vali_loss.data.item())
                val_losses_r.append(vali_loss_r.data.item())
                val_losses_ind.append(vali_loss_ind.tolist())
                eval_meter_r.update(vali_logits_r, vali_labels_r, vali_masks_r)
            total_r2 = np.nanmean(eval_meter_r.compute_metric("r2"))
            total_loss_r = np.mean(val_losses_r)
            total_loss = np.mean(val_losses)
            total_loss_ind = np.mean(val_losses_ind, axis=0)
            return (total_r2, total_loss, total_loss_r, total_loss_ind.tolist())

    def eval_epoch_mr(self, i, data_loader, label_name, n_ctasks, path, name):
        self.net.eval()
        meter = Meter()
        with torch.no_grad():
            for _, bg1, bg2, labels, masks in data_loader:
                bg1, bg2 = (bg1.to(self.device), bg2.to(self.device))
                labels, masks = (labels.to(self.device), masks.to(self.device))
                n_feats1 = bg1.ndata.pop("hv").to(self.device)
                e_feats1 = bg1.edata.pop("he").to(self.device)
                n_feats2 = bg2.ndata.pop("hv").to(self.device)
                e_feats2 = bg2.edata.pop("he").to(self.device)
                logits = self.net([[bg1, n_feats1, e_feats1], [bg2, n_feats2, e_feats2]])
                if self.normalizer is not None:
                    labels = self.normalizer.inverse_transform_labels(labels, masks, n_ctasks)
                    logits = self.normalizer.inverse_transform_labels(logits, masks, n_ctasks)
                meter.update(logits[:, n_ctasks:], labels[:, n_ctasks:], masks[:, n_ctasks:])
        metrics = {
            "PCC_denorm": meter.compute_metric("pcc"),
            "R2_denorm": meter.compute_metric("r2"),
            "MAE_denorm": meter.compute_metric("mae"),
            "RMSE_denorm": meter.compute_metric("rmse"),
        }
        pd.DataFrame({"Task": label_name[n_ctasks:], **metrics}).to_csv(
            Path(path) / f"{i}{name}_evalue_r_denorm.csv", index=False
        )
        print(
            f"Fold {i} | {name} | mean_R2={np.nanmean(metrics['R2_denorm']):.4f} MAE={np.nanmean(metrics['MAE_denorm']):.4f} RMSE={np.nanmean(metrics['RMSE_denorm']):.4f} | {performance_summary(metrics['R2_denorm'], 'R2', label_name[n_ctasks:])}"
        )
        return tuple(
            (np.nanmean(metrics[key]) for key in ["R2_denorm", "MAE_denorm", "RMSE_denorm"])
        )
