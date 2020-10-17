import sys
import os
import glob
import functools
import pandas as pd
import joblib
import SimpleITK as sitk
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime as dt
from torch.utils.data import DataLoader
from util.logconf import logging
from util.util import XyzTuple, xyz2irc
from prep_data import candidates_with_diameters_file, cached_data_dir, data_dir


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class Ct:
    def __init__(self, series_uid):
        self.series_uid = series_uid
        mhd_path = glob.glob(f"{data_dir}/subset*/{series_uid}.mhd")[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_array = np.asarray(sitk.GetArrayFromImage(ct_mhd), np.float32)

        self.ct = np.clip(ct_array, -1000, 1000)

        self.origin = XyzTuple(*ct_mhd.GetOrigin())
        self.voxel_size = XyzTuple(*ct_mhd.GetSpacing())
        self.direction = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_ct_slice(self, slice_dimensions, center_xyz):
        center_irc = xyz2irc(
            center_xyz,
            self.origin,
            self.voxel_size,
            self.direction
        )

        ct_shape = self.ct.shape
        slice_list = []
        for i, dim in enumerate(slice_dimensions):
            start_idx = center_irc[i] + dim//2
            end_idx = start_idx + dim

            if start_idx < 0:
                start_idx = 0
                end_idx = dim

            if end_idx > ct_shape[i]:
                start_idx = ct_shape[i] - dim
                end_idx = ct_shape[i]

            slice_list.append(slice(start_idx, end_idx))

        ct_slice = self.ct[tuple(slice_list)]
        return ct_slice, center_irc


@functools.lru_cache(1, typed=True)
def load_ct_from_cache(series_uid):
    ct = Ct(series_uid)
    return ct

@functools.lru_cache(2)
def slice_ct_cached(series_uid, slice_dimensions, center_xyz):
    ct = load_ct_from_cache(series_uid)
    ct_slice, center_irc = ct.get_ct_slice(slice_dimensions, center_xyz)
    return ct_slice, center_irc


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
            cache_file = os.path.join(cached_data_dir, f"{seriesuid}_{coord_x}_{coord_y}_{coord_z}_{nodule_class}.pkl")
            if not os.path.exists(cache_file):
                center_xyz = tuple([coord_x, coord_y, coord_z])
                ct_slice, center_irc = slice_ct_cached(
                    seriesuid,
                    self.ct_slice_size,
                    center_xyz
                )

                
                joblib.dump([ct_slice, center_irc], cache_file)
            


if __name__ == '__main__':
    CacheLunaData().main()
