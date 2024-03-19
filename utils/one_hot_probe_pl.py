import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from utils.linear_probe_pl import update_epoch_metrics_dict, weighted_mean


def one_hot_metrics(targets, predicted, prefix):
    targets = targets.detach().cpu()
    predicted = predicted.detach().cpu()

    # Total correct predictions - metrics
    acc = accuracy_score(targets, predicted)
    f1 = f1_score(targets, predicted, average="macro", zero_division=0)

    scores_dict = {
        prefix + "_acc": [acc],
        prefix + "_f1_score": [f1],
    }
    return scores_dict


class OneHotLinearProbe(pl.LightningModule):
    def __init__(self, encoder, classifier, optimizer, scheduler):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.optimizer = optimizer
        if scheduler:
            self.scheduler = scheduler
        self.epoch_metrics_dict = {}

    def configure_optimizers(self):
        if self.scheduler:
            return [self.optimizer], [self.scheduler]
        else:
            return self.optimizer

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        logits = self.classifier(outputs)
        return logits

    def shared_step(self, batch, batch_idx, mode):
        inputs, tgts = batch
        out = self(inputs)

        batch_size = len(tgts)

        loss = F.cross_entropy(out, tgts)
        preds = torch.argmax(out, dim=1)

        batch_metrics_dict = one_hot_metrics(tgts, preds, prefix=mode)

        self.epoch_metrics_dict = update_epoch_metrics_dict(
            self.epoch_metrics_dict, batch_metrics_dict
        )
        self.epoch_metrics_dict = update_epoch_metrics_dict(
            self.epoch_metrics_dict, {f"{mode}_loss": [loss.detach().cpu()]}
        )
        self.epoch_metrics_dict = update_epoch_metrics_dict(
            self.epoch_metrics_dict,
            {f"{mode}_batch_size": [batch_size]},
        )

        return loss

    def shared_epoch_end(self, mode):
        epoch_loss = np.average(self.epoch_metrics_dict[f"{mode}_loss"])

        epoch_acc = weighted_mean(
            self.epoch_metrics_dict, f"{mode}_acc", f"{mode}_batch_size"
        )
        epoch_f1 = weighted_mean(
            self.epoch_metrics_dict, f"{mode}_f1_score", f"{mode}_batch_size"
        )

        metrics_log = {
            f"{mode}_loss": epoch_loss,
            f"{mode}_acc": epoch_acc,
            f"{mode}_f1_score": epoch_f1,
        }

        self.log_dict(metrics_log, sync_dist=True)
        return epoch_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode="train")

    def on_train_epoch_end(self):
        train_epoch_loss = self.shared_epoch_end("train")
        self.log("train_loss", train_epoch_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode="val")

    def on_validation_epoch_end(self):
        val_epoch_loss = self.shared_epoch_end("val")
        self.log("val_loss", val_epoch_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode="test")

    def on_test_epoch_end(self):
        test_epoch_loss = self.shared_epoch_end("test")
        self.log("test_loss", test_epoch_loss, sync_dist=True)

    def predict_step(self, batch):
        self.encoder.eval()
        self.classifier.eval()

        inputs, tgts = batch

        # Predictions
        classifier_outs = self(inputs)
        predictions = torch.max(classifier_outs, dim=1)[1]

        return predictions, tgts
