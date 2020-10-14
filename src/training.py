import os
import sys
import argparse
from datetime import datetime as dt
import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.logconf import logging
from prep_data import LunaDataset
from models import LunaModel
from util.util import enumerateWithEstimate

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'log')
METRICS_LABEL_IDX = 0
METRICS_PREDICTION_IDX = 1
METRICS_LOSS_IDX = 2
NUM_METRICS = 3


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num-workers",
            "-n",
            help="Number of workers for background data-loading",
            default=6,
            type=int
        )
        parser.add_argument(
            "--batch-size",
            "-b",
            help="Size of training/validation batch (Default: 16)",
            default=16,
            type=int
        )
        parser.add_argument(
            "--epochs",
            "-e",
            help="Number of training epochs (Default: 5)",
            default=5,
            type=int
        )
        parser.add_argument(
            "--val-stride",
            "-v",
            help="Stride for Validation data (Default: 10)",
            default=10,
            type=int
        )
        parser.add_argument(
            "--balance",
            help="Balancing ratio (Default: 2 - indicating 2:1 negative to positive ratio)",
            default=2,
            type=int
        )
        parser.add_argument(
            "--max-samples",
            "-m",
            help="Maximum number of samples (Default: 150k)",
            default=150_000,
            type=int
        )
        self.args = parser.parse_args(sys_argv)
        self.start_time = dt.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.total_training_samples_count = 0

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.tb_writer_train, self.tb_writer_val = self.init_tb_writer()

    def init_model(self):
        model = LunaModel()
        model = model.to(self.device)
        return model

    def init_optimizer(self):
        optimizer = SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        return optimizer

    def init_tb_writer(self):
        iteration_logs_dir = os.path.join(logs_dir, self.start_time)
        # if not os.path.exists(iteration_logs_dir):
        #     os.makedirs(iteration_logs_dir)
        trn_writer = SummaryWriter(f"{iteration_logs_dir}-trn")
        val_writer = SummaryWriter(f"{iteration_logs_dir}-val")

        return trn_writer, val_writer

    def init_train_dl(self):
        train_dataset = LunaDataset(val_stride=self.args.val_stride,
                                    class_balance=self.args.balance,
                                    max_samples=self.args.max_samples)
        train_dl = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda
        )
        return train_dl

    def init_val_dl(self):
        val_dataset = LunaDataset(val_stride=self.args.val_stride,
                                  is_val_set=True,
                                  class_balance=self.args.balance,
                                  max_samples=self.args.max_samples)
        val_dl = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda
        )
        return val_dl

    def run_train_loop(self, train_dl, epoch_idx):
        self.model.train()
        epoch_train_metrics = torch.zeros(
            NUM_METRICS,
            len(train_dl.dataset),
            device=self.device
        )
        # batch_iterator = enumerateWithEstimate(train_dl, f"Train Epoch - {epoch_idx}", train_dl.num_workers)
        batch_iterator = enumerate(train_dl)
        for batch_idx, batch in batch_iterator:
            # log.info("Loaded 1 batch")
            self.optimizer.zero_grad()
            batch_loss = self.compute_batch_loss(batch_idx, train_dl.batch_size, batch, epoch_train_metrics)
            batch_loss.backward()
            self.optimizer.step()
            if (batch_idx + 1) % 16 == 0:
                log.info(f"Processed {batch_idx + 1} batch")

        self.total_training_samples_count += len(train_dl.dataset)
        return epoch_train_metrics.to('cpu')

    def run_val_loop(self, val_dl, epoch_idx):
        with torch.no_grad():
            self.model.eval()
            epoch_val_metrics = torch.zeros(
                NUM_METRICS,
                len(val_dl.dataset),
                device=self.device
            )
            batch_iterator = enumerateWithEstimate(val_dl, f"Val Epoch - {epoch_idx}", val_dl.num_workers)
            for batch_idx, batch in batch_iterator:
                batch_loss = self.compute_batch_loss(batch_idx, val_dl.batch_size, batch, epoch_val_metrics)

            return epoch_val_metrics.to('cpu')

    def compute_batch_loss(self, batch_idx, batch_size, batch, metrics):
        ct_tensor, label_tensor, _seriesuid_list, _center_irc_list = batch

        ct_tensor_device = ct_tensor.to(self.device, non_blocking=True)
        label_tensor_device = label_tensor.to(self.device, non_blocking=True)

        logit_pred, label_pred = self.model(ct_tensor_device)

        loss = nn.CrossEntropyLoss(reduction='none')(
            logit_pred,
            label_tensor_device[:, 1]
        )

        batch_start_idx = batch_idx * batch_size
        batch_end_idx = batch_start_idx + len(logit_pred)

        metrics[METRICS_LOSS_IDX, batch_start_idx:batch_end_idx] = loss.detach()
        metrics[METRICS_LABEL_IDX, batch_start_idx:batch_end_idx] = label_tensor_device[:, 1].detach()
        metrics[METRICS_PREDICTION_IDX, batch_start_idx:batch_end_idx] = label_pred[:, 1].detach()

        return loss.mean()

    def log_metrics(self, metrics_tensor, mode, classification_threshold=0.5):
        neg_label_mask = metrics_tensor[METRICS_LABEL_IDX] <= classification_threshold
        pos_label_mask = ~neg_label_mask

        neg_pred_mask = metrics_tensor[METRICS_PREDICTION_IDX] <= classification_threshold
        pos_pred_mask = ~neg_pred_mask

        total_negative = neg_label_mask.sum()
        correct_negative = (neg_label_mask & neg_pred_mask).sum()

        total_positive = pos_label_mask.sum()
        correct_positive = (pos_label_mask & pos_pred_mask).sum()

        correct_all = correct_positive + correct_negative
        total_all = total_positive + total_negative

        metrics_dict = dict()
        metrics_dict['loss/all'] = metrics_tensor[METRICS_LOSS_IDX].mean()
        metrics_dict['loss/pos'] = metrics_tensor[METRICS_LOSS_IDX, list(pos_label_mask)].mean()
        metrics_dict['loss/neg'] = metrics_tensor[METRICS_LOSS_IDX, list(neg_label_mask)].mean()

        metrics_dict['accuracy/all'] = 100 * correct_all / np.float32(total_all)
        metrics_dict['accuracy/pos'] = 100 * correct_positive / np.float32(total_positive)
        metrics_dict['accuracy/neg'] = 100 * correct_negative / np.float32(total_negative)

        log.info(f"Loss {mode}: {metrics_dict['loss/all']}\n"
                 f"Accuracy {mode}: {metrics_dict['accuracy/all']} ({correct_all} of {total_all})\n\n")
        log.info(f"Loss {mode}_pos: {metrics_dict['loss/pos']} \n"
                 f"Accuracy {mode}_pos: {metrics_dict['accuracy/pos']} ({correct_positive} of {total_positive})\n\n")
        log.info(f"Loss {mode}_neg: {metrics_dict['loss/neg']} \n"
                 f"Accuracy {mode}_neg: {metrics_dict['accuracy/neg']} ({correct_negative} of {total_negative})\n\n")

        writer = getattr(self, f"tb_writer_{mode}")
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.total_training_samples_count)

        writer.add_pr_curve(
            "PR",
            metrics_tensor[METRICS_LABEL_IDX],
            metrics_tensor[METRICS_PREDICTION_IDX],
            self.total_training_samples_count
        )

    def main(self):
        log.info(f"Start process at {self.start_time}")
        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()

        log.info(f"Training with {len(train_dl)} training batches "
                 f"and {len(val_dl)} validation batches")

        num_epochs = self.args.epochs
        for epoch_idx in range(1, num_epochs + 1):
            log.info(f"Epoch {epoch_idx} of {num_epochs}")
            train_metrics = self.run_train_loop(train_dl, epoch_idx)
            self.log_metrics(train_metrics, "train")

            val_metrics = self.run_val_loop(val_dl, epoch_idx)
            self.log_metrics(val_metrics, "val")


if __name__ == '__main__':
    LunaTrainingApp().main()
