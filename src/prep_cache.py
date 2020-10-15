import sys
import os
import argparse
from tqdm import tqdm
import joblib
from datetime import datetime as dt
import pandas as pd
from torch.utils.data import DataLoader
from util.logconf import logging
from prep_data import load_ct_from_cache, slice_ct_cached, candidates_with_diameters_file, cached_data_dir


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class CacheLunaData:
    def __init__(self):
        self.start_time = dt.now().strftime("%Y-%m-%d_%H.%M.%S")
        self.ct_slice_size = tuple([32, 48, 48])

        df = pd.read_csv(candidates_with_diameters_file)
        self.data = df.loc[df["on_disk"]].reset_index(drop=True)
        self.data.sort_values("seriesuid", inplace=True)

    def main(self):
        log.info(f"Start caching at {self.start_time}")
        total = self.data.shape[0]
        for idx, row in tqdm(enumerate(self.data.iterrows()), total=total):
            seriesuid, coord_x, coord_y, coord_z, nodule_class, _, _ = row[1]
            center_xyz = tuple([coord_x, coord_y, coord_z])
            ct_slice, center_irc = slice_ct_cached(
                seriesuid,
                self.ct_slice_size,
                center_xyz
            )

            cache_file = os.path.join(cached_data_dir, f"{seriesuid}_{coord_x}_{coord_y}_{coord_z}_{nodule_class}.pkl")
            joblib.dump([ct_slice, center_irc], cache_file)
            


if __name__ == '__main__':
    CacheLunaData().main()
