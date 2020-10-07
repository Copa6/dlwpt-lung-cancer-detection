import sys
import argparse
from datetime import datetime as dt
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from util.logconf import logging
from prep_data import LunaDataset
from models import LunaModel


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


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
        self.args = parser.parse_args(sys_argv)
        self.start_time = dt.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        model = LunaModel()
        model = model.to(self.device)
        return model

    def init_optimizer(self):
        optimizer = SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        return optimizer

    def _init_train_dl(self):
        train_dataset = LunaDataset(val_stride=self.args.val_stride)
        train_dl = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda
        )
        return train_dl

    def _init_val_dl(self):
        val_dataset = LunaDataset(val_stride=self.args.val_stride, is_val_set=True)
        val_dl = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda
        )
        return val_dl

    def main(self):
        log.info(f"Start process at {self.start_time}")
        train_dl = self._init_train_dl()
        val_dl = self._init_val_dl()

        log.info(f"Training with {len(train_dl)} training batches "
                 f"and {len(val_dl)} validation batches")


if __name__ == '__main__':
    LunaTrainingApp().main()







