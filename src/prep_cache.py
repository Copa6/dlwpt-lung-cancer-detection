import sys
import os
import argparse
from datetime import datetime as dt
from torch.utils.data import DataLoader
from util.logconf import logging
from prep_data import LunaDataset
from util.util import enumerateWithEstimate


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class CacheLunaData:
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

    def main(self):
        log.info(f"Start caching at {self.start_time}")
        dl = DataLoader(
            LunaDataset(max_samples=10e6, class_balance=0, to_cache=1),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

        # batch_iterator = enumerateWithEstimate(dl,
        #                                        f"Fill cache ", dl.num_workers)
        for batch_idx, _ in enumerate(dl):
            if (batch_idx+1)%16 == 0:
                log.info(f"Processed {batch_idx + 1} batches")


if __name__ == '__main__':
    CacheLunaData().main()
