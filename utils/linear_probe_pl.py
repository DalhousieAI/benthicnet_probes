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

_SAMPLE_HEAD_DICT = {
    "biota": 0,
    "substrate": 1,
    "relief": 2,
    "bedforms": 3,
    "colour": 4,
    "biota_mask": 5,
    "substrate_mask": 6,
    "relief_mask": 7,
    "bedforms_mask": 8,
}


# Epoch metrics
def weighted_mean(metric_dict, metric_key, batch_size_key):
    original_metric_array = np.array(metric_dict[metric_key])
    original_batch_size_array = np.array(metric_dict[batch_size_key])
    existing_metric_indices = np.where(original_metric_array != -1)[0]

    existing_metric_array = original_metric_array[existing_metric_indices]
    existing_batch_size_array = original_batch_size_array[existing_metric_indices]

    total_batch_size = np.sum(existing_batch_size_array)
    if total_batch_size == 0:
        return -1.0

    norm_batch_size_weights = existing_batch_size_array / np.sum(
        existing_batch_size_array
    )
    return np.average(existing_metric_array, weights=norm_batch_size_weights)


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
    # MCLoss - why doubles?
    constr_output = get_constr_out(logits, R)
    output = targets * logits.double()
    output = get_constr_out(output, R)
    output = (1 - targets) * constr_output.double() + targets * output
    loss = F.binary_cross_entropy_with_logits(output, targets, reduction="none")

    valid_logit_count = torch.sum(masks == 1, dim=1)

    nonzero_mask = valid_logit_count != 0
    # Average across labels first
    loss = torch.sum(loss * masks, dim=1)
    loss[nonzero_mask] /= valid_logit_count[nonzero_mask]

    return loss


# Calculate hierarhcical multilabel model performance metrics
def ml_metrics(targets, probabilities, predicted, prefix):
    targets = targets.detach().cpu()
    probabilities = probabilities.detach().cpu()
    predicted = predicted.detach().cpu()

    # Total correct predictions - metrics
    if len(targets) > 0:
        acc = accuracy_score(targets, predicted)
        ap_score = average_precision_score(targets, probabilities, average="micro")
        f1 = f1_score(targets, predicted, average="micro", zero_division=0)
    else:
        acc = -1.0
        ap_score = -1.0
        f1 = -1.0

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
    def process_targets_for_empty_labels(self, outputs, targets):
        existing_indices = (
            torch.logical_not(torch.all(targets == 0, dim=1))
            .nonzero(as_tuple=False)
            .squeeze()
        )

        outputs = torch.index_select(outputs, 0, existing_indices)
        targets = torch.index_select(targets, 0, existing_indices)

        return outputs, targets

    # Produce mask for training (give locations of targets which are empty)
    def get_missing_mask(self, targets):
        missing_target_indices = (
            torch.all(targets == 0, dim=1).nonzero(as_tuple=False).squeeze()
        )

        missing_mask = torch.ones(len(targets), device=self.device)
        missing_mask[missing_target_indices] = 0

        return missing_mask

    def shared_step(self, batch, batch_idx, partition_prefix):
        inputs, data = batch
        head_losses = 0
        out = self(inputs)
        num_valid_head_losses = 0
        for head in self.heads:
            head_net = self.heads[head]
            original_outputs = out[head]
            original_targets = data[_SAMPLE_HEAD_DICT[head]].to(
                self.device
            )  # targets is in a batch, some samples of which are empty (full of zeros)

            # The missing mask is a sanity check for the empty samples for CFFNN heads
            # It is also necessary for colour heads, which do not have a mask
            # This mask is applied at the "head loss" level
            missing_mask = self.get_missing_mask(original_targets)
            outputs_pred, targets_pred = self.process_targets_for_empty_labels(
                original_outputs, original_targets
            )

            if isinstance(head_net, ConstrainedFFNNModel):
                local_R = self.Rs[head].to(self.device)
                # This is the sample mask, which is applied at the "sample loss" level
                masks = data[_SAMPLE_HEAD_DICT[head + "_mask"]].to(self.device)
                head_loss_samples = mcloss(
                    original_outputs, original_targets, local_R, masks
                )

                outputs_pred = get_constr_out(outputs_pred, local_R)
            elif isinstance(head_net, MultiLabelFFNNModel):
                head_loss_raw = F.binary_cross_entropy_with_logits(
                    original_outputs, original_targets, reduction="none"
                )
                head_loss_samples = torch.mean(head_loss_raw, 1)
            else:
                raise ValueError("Head network type not recognized")

            # Think of this part as applying mask at the "batch loss" level
            # We are "zeroing out" the loss for heads which received no samples this batch
            effective_head_batch_size = len(targets_pred)
            if effective_head_batch_size > 0:
                head_loss = (
                    torch.sum(head_loss_samples * missing_mask)
                    / effective_head_batch_size
                )
                num_valid_head_losses += 1
            else:
                # If all samples for this head are empty,
                # we do not include it in the average batch loss across heads
                head_loss = torch.sum(head_loss_samples) * 0
            head_losses += head_loss

            sigmoid = nn.Sigmoid()
            probabilities = sigmoid(outputs_pred)
            predicted = probabilities > 0.5
            prefix = f"{partition_prefix}_{head}"

            batch_metrics_dict = ml_metrics(
                targets_pred, probabilities, predicted, prefix
            )

            self.epoch_metrics_dict = update_epoch_metrics_dict(
                self.epoch_metrics_dict, batch_metrics_dict
            )
            self.epoch_metrics_dict = update_epoch_metrics_dict(
                self.epoch_metrics_dict, {f"{prefix}_loss": [head_loss.detach().cpu()]}
            )

            self.epoch_metrics_dict = update_epoch_metrics_dict(
                self.epoch_metrics_dict,
                {f"{prefix}_batch_size": [effective_head_batch_size]},
            )
        head_losses = head_losses / num_valid_head_losses

        self.epoch_metrics_dict = update_epoch_metrics_dict(
            self.epoch_metrics_dict,
            {f"{partition_prefix}_loss": [head_losses.detach().cpu()]},
        )
        return head_losses

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
            epoch_head_loss = weighted_mean(
                self.epoch_metrics_dict, f"{prefix}_loss", f"{prefix}_batch_size"
            )

            head_log = {
                f"{prefix}_acc": epoch_head_acc,
                f"{prefix}_ap_score": epoch_head_ap_score,
                f"{prefix}_f1_score": epoch_head_f1_score,
                f"{prefix}_loss": epoch_head_loss,
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
        _ = self.shared_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        val_epoch_loss = self.shared_epoch_end("val")
        self.log("val_loss", val_epoch_loss, sync_dist=True)
        drop_keys_with_string(self.epoch_metrics_dict, "val")

    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        test_epoch_loss = self.shared_epoch_end("test")
        self.log("test_loss", test_epoch_loss, sync_dist=True)
        drop_keys_with_string(self.epoch_metrics_dict, "test")

    def predict_head_step(self, batch, head="biota", random_out=False):
        inputs, data = batch

        # Data for relevant head
        head_data = data[_SAMPLE_HEAD_DICT[head]]

        # Predictions
        if random_out:
            head_outs = torch.rand(head_data.shape, device=self.device) - 0.5
        else:
            head_outs = self(inputs)[head]
        local_R = self.Rs[head].to(self.device)
        outputs_pred = get_constr_out(head_outs, local_R)
        sigmoid = nn.Sigmoid()
        predictions = sigmoid(outputs_pred) > 0.5

        # Masks
        masks = data[_SAMPLE_HEAD_DICT[head + "_mask"]]

        return predictions, head_data, masks
