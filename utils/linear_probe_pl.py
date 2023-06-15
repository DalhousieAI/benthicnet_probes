from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from utils.constrained_ff import ConstrainedFFNNModel, get_constr_out
from utils.multilabel_ff import MultiLabelFFNNModel


# Epoch metrics
def weighted_mean(metric_dict, metric_key, batch_size_key):
    original_metric_array = np.array(metric_dict[metric_key])
    original_batch_size_array = np.array(metric_dict[batch_size_key])
    non_nan_indices = np.where(~np.isnan(original_metric_array))[0]
    if len(non_nan_indices) == 0:
        return np.nan

    nan_less_metric_array = original_metric_array[non_nan_indices]
    nan_less_batch_size_array = original_batch_size_array[non_nan_indices]

    norm_batch_size_weights = nan_less_batch_size_array / np.sum(
        nan_less_batch_size_array
    )
    return np.average(nan_less_metric_array, weights=norm_batch_size_weights)


# Return indices of samples which are valid for metrics in a batch
def find_all_non_false_indices(batch_tensor):
    # Check if all elements in each tensor of the batch are False
    all_false_mask = torch.all(batch_tensor == False, dim=1)
    # Find indices of tensors that are not all False
    indices = torch.nonzero(~all_false_mask).flatten()

    return indices


# Drop certain keys in dictionary
def drop_keys_with_string(dictionary, prefix):
    keys_to_drop = [key for key in dictionary.keys() if prefix in key]
    for key in keys_to_drop:
        del dictionary[key]


# Utility function for updating metric dictionaries
def update_epoch_metrics_dict(original_dict, new_dict):
    merged_dict = defaultdict(list)

    # Update merged_dict with original_dict
    for key, value in original_dict.items():
        merged_dict[key].extend(value)
    for key, value in new_dict.items():
        merged_dict[key].extend(value)

    merged_dict = dict(merged_dict)
    return merged_dict


# Calculate constrained loss for hierarchical multi-label classification
def mcloss(logits, targets, R, masks):
    # MCLoss
    constr_output = get_constr_out(logits, R)
    output = targets * logits.double()
    output = get_constr_out(output, R)
    output = (1 - targets) * constr_output.double() + targets * output
    loss = F.binary_cross_entropy_with_logits(output, targets, reduction="none")

    # Average across labels first
    loss = torch.mean(loss * masks, 1)
    return loss


# Calculate hierarhcical multilabel model performance metrics
def ml_metrics(targets, predicted, prefix):
    targets = targets.detach().cpu()
    predicted = predicted.detach().cpu()

    valid_sample_indices = find_all_non_false_indices(targets)

    targets = torch.index_select(targets, 0, valid_sample_indices)
    predicted = torch.index_select(predicted, 0, valid_sample_indices)

    # Total correct predictions - metrics
    if len(valid_sample_indices) > 0:
        acc = accuracy_score(targets, predicted)
        ap_score = average_precision_score(targets, predicted, average="micro")
        f1 = f1_score(targets, predicted, average="micro", zero_division=0)
    else:
        acc = np.nan
        ap_score = np.nan
        f1 = np.nan

    scores_dict = {
        prefix + "_acc": [acc],
        prefix + "_ap_score": [ap_score],
        prefix + "_f1_score": [f1],
    }
    return scores_dict


class LinearProbe(pl.LightningModule):
    def __init__(self, encoder, heads, optimizer, scheduler, Rs):
        super().__init__()
        self.encoder = encoder
        self.heads = heads
        self.optimizer = optimizer
        if scheduler:
            self.scheduler = scheduler
        self.Rs = Rs
        self.epoch_metrics_dict = {}

    def configure_optimizers(self):
        if self.scheduler:
            return [self.optimizer], [self.scheduler]
        else:
            return self.optimizer

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        logits = {
            head: head_network(outputs) for head, head_network in self.heads.items()
        }
        return logits

    # Parse targets for predictions
    def process_targets_for_nans_drop_labels(self, outputs, targets):
        non_nan_indices = (
            torch.logical_not(torch.isnan(targets).any(dim=1))
            .nonzero(as_tuple=False)
            .squeeze()
        )

        outputs = torch.index_select(outputs, 0, non_nan_indices)
        targets = torch.index_select(targets, 0, non_nan_indices)

        return outputs, targets

    # Parse targets for training
    def process_targets_for_nans_keep_labels(self, outputs, targets, masks=None):
        nan_indices = torch.isnan(targets).any(dim=1).nonzero(as_tuple=False).squeeze()

        targets[nan_indices] = torch.zeros_like(targets[nan_indices])

        nan_mask = torch.ones(len(targets), device=self.device)
        nan_mask[nan_indices] = 0

        if masks is not None:
            masks[nan_indices] = torch.zeros_like(
                masks[nan_indices], device=self.device
            )
            return outputs, targets, masks, nan_mask

        return outputs, targets, nan_mask

    def shared_step(self, batch, batch_idx, partition_prefix):
        (
            inputs,
            biota,
            substrate,
            relief,
            bedforms,
            colour,
            biota_mask,
            substrate_mask,
            relief_mask,
            bedforms_mask,
        ) = batch
        head_losses = []
        out = self(inputs)
        for head in self.heads:
            head_net = self.heads[head]
            original_outputs = out[head]
            original_targets = locals()[head].to(
                self.device
            )  # targets is in a batch, some samples of which are empty
            outputs_pred, targets_pred = self.process_targets_for_nans_drop_labels(
                original_outputs, original_targets
            )
            if isinstance(head_net, ConstrainedFFNNModel):
                local_R = self.Rs[head].to(self.device)
                masks = locals()[head + "_mask"].to(self.device)
                (
                    outputs_train,
                    targets_train,
                    masks_train,
                    nan_mask,
                ) = self.process_targets_for_nans_keep_labels(
                    original_outputs, original_targets, masks
                )
                head_loss_samples = mcloss(
                    outputs_train, targets_train, local_R, masks_train
                )

                outputs_pred = get_constr_out(outputs_pred, local_R)
            elif isinstance(head_net, MultiLabelFFNNModel):
                (
                    outputs_train,
                    targets_train,
                    nan_mask,
                ) = self.process_targets_for_nans_keep_labels(
                    original_outputs, original_targets
                )
                head_loss_raw = F.binary_cross_entropy_with_logits(
                    outputs_train, targets_train, reduction="none"
                )
                head_loss_samples = torch.mean(head_loss_raw, 1)
            else:
                raise ValueError("Head network type not recognized")

            head_loss = torch.mean(head_loss_samples * nan_mask)
            head_losses.append(head_loss)

            # We denote effective batch size as the number of samples that are not nan
            effective_head_batch_size = len(targets_pred)

            sigmoid = nn.Sigmoid()
            predicted = sigmoid(outputs_pred) > 0.5
            prefix = f"{partition_prefix}_{head}"

            batch_metrics_dict = ml_metrics(targets_pred, predicted, prefix)

            self.epoch_metrics_dict = update_epoch_metrics_dict(
                self.epoch_metrics_dict, batch_metrics_dict
            )
            self.epoch_metrics_dict = update_epoch_metrics_dict(
                self.epoch_metrics_dict, {f"{prefix}_loss": [head_loss]}
            )
            self.epoch_metrics_dict = update_epoch_metrics_dict(
                self.epoch_metrics_dict,
                {f"{prefix}_batch_size": [effective_head_batch_size]},
            )

        head_losses = torch.stack(head_losses)
        num_non_zero_head_losses = torch.nonzero(head_losses).size(0)
        batch_loss = torch.sum(head_losses) / num_non_zero_head_losses

        self.epoch_metrics_dict = update_epoch_metrics_dict(
            self.epoch_metrics_dict, {f"{partition_prefix}_loss": [batch_loss.item()]}
        )
        return batch_loss

    def shared_epoch_end(self, partition_prefix):
        epoch_loss = np.average(self.epoch_metrics_dict[f"{partition_prefix}_loss"])
        for head in self.heads:
            prefix = f"{partition_prefix}_{head}"
            epoch_head_acc = weighted_mean(
                self.epoch_metrics_dict, f"{prefix}_acc", f"{prefix}_batch_size"
            )
            epoch_head_ap_score = weighted_mean(
                self.epoch_metrics_dict, f"{prefix}_ap_score", f"{prefix}_batch_size"
            )
            epoch_head_f1_score = weighted_mean(
                self.epoch_metrics_dict, f"{prefix}_f1_score", f"{prefix}_batch_size"
            )

            head_log = {
                f"{prefix}_acc": epoch_head_acc,
                f"{prefix}_ap_score": epoch_head_ap_score,
                f"{prefix}_f1_score": epoch_head_f1_score,
            }
            self.log_dict(head_log, sync_dist=True)
        return epoch_loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "train")
        return loss

    def on_train_epoch_end(self):
        train_epoch_loss = self.shared_epoch_end("train")
        self.log("train_loss", train_epoch_loss, sync_dist=True)
        drop_keys_with_string(self.epoch_metrics_dict, "train")

    def validation_step(self, batch, batch_idx):
        val_loss = self.shared_step(batch, batch_idx, "val")
        return val_loss

    def on_validation_epoch_end(self):
        val_epoch_loss = self.shared_epoch_end("val")
        self.log("val_loss", val_epoch_loss, sync_dist=True)
        drop_keys_with_string(self.epoch_metrics_dict, "val")

    def test_step(self, batch, batch_idx):
        test_loss = self.shared_step(batch, batch_idx, "test")
        return test_loss

    def on_test_epoch_end(self):
        test_epoch_loss = self.shared_epoch_end("test")
        self.log("test_loss", test_epoch_loss, sync_dist=True)
        drop_keys_with_string(self.epoch_metrics_dict, "test")
