import os
import glob
import functools
import pandas as pd
import joblib
import math
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from util.util import XyzTuple, xyz2irc, enumerateWithEstimate
from util.logconf import logging
from util.disk import getCache

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(base_dir, "data")
annotations_file = os.path.join(data_dir, "annotations.csv")
candidates_file = os.path.join(data_dir, "candidates.csv")
candidates_with_diameters_file = os.path.join(data_dir, "candidates_with_dia.csv")
cached_data_dir = os.path.join(data_dir, "cache")

# raw_cache = getCache(cached_data_dir)


def add_diameter_to_candidates(override=False):
    if not(os.path.exists(candidates_with_diameters_file)) or override:
        annotations_df = pd.read_csv(annotations_file)
        candidates_df = pd.read_csv(candidates_file)
        mhd_list_files = glob.glob(f"{data_dir}/subset*/*.mhd")
        present_on_disk_mhd = {os.path.split(p)[-1][:-4] for p in mhd_list_files}

        diameters = [0] * len(candidates_df)
        is_present = [True] * len(candidates_df)
        for idx, row in candidates_df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx} rows")
            s_uid, x, y, z, label = row
            if s_uid not in present_on_disk_mhd:
                print(f"{s_uid} not found on disk")
                is_present[idx] = False
                continue
            diameter_table = annotations_df.loc[annotations_df["seriesuid"] == s_uid]
            for _, a_row in diameter_table.iterrows():
                _a, a_x, a_y, a_z, dia = a_row
                if all([abs(i - j) <= (dia/4) for i, j in zip([a_x, a_y, a_z], [x, y, z])]):
                    diameters[idx] = dia
                    break
            candidates_df["diameters"] = diameters
            candidates_df["on_disk"] = is_present
            candidates_df.sort_values("diameters", inplace=True)
            candidates_df.to_csv(candidates_with_diameters_file, index=False)

        else:
            print("Found existing file and no override flag.")


class LunaDataset(Dataset):
    def __init__(self, is_val_set=False, val_stride=10, class_balance=None, max_samples=None):
        self.class_balance = class_balance
        self.ct_slice_size = tuple([32, 48, 48])

        df = pd.read_csv(candidates_with_diameters_file)
        df_on_disk = df.loc[df["on_disk"]].reset_index(drop=True)
        
        if is_val_set:
            data = df_on_disk.iloc[::val_stride, :].reset_index(drop=True)
        else:
            data = df_on_disk.drop(df_on_disk.index[::val_stride]).reset_index(drop=True)

        self.max_samples = max_samples if max_samples else data.shape[0]
        self.pos_samples = data[data["class"] == 1].reset_index(drop=True)
        self.num_pos_samples = len(self.pos_samples)

        self.neg_samples = data[data["class"] == 0].reset_index(drop=True)
        self.num_neg_samples = len(self.neg_samples)

        if self.class_balance:
            log.info(f"Prepare balanced data")
            self.data = self.prepare_balanced_data()
        else:
            self.data = data
        # print(self.data.head())

        log.info(f"Load {'val' if is_val_set else 'training'} data with "
                    f"{max_samples} maximum samples\n"
                    f"Balancing ratio - {class_balance}, with \n"
                    f"{self.num_pos_samples} positive samples\n"
                    f"{self.num_neg_samples} negative samples")

    def prepare_balanced_data(self):
        num_positives = self.max_samples//(self.class_balance + 1)
        num_negatives = self.max_samples - num_positives

        sampled_positives = self.pos_samples.sample(n=num_positives, replace=True)
        sampled_negatives = self.neg_samples.sample(n=num_negatives, replace=True)
        balanced_data = sampled_positives.append(sampled_negatives).reset_index(drop=True)
        balanced_data = balanced_data.sample(frac=1)
        balanced_data.sort_values("diameters", inplace=True)
        data = balanced_data.reset_index(drop=True)
        return data

    def __len__(self):
        num_samples = len(self.data)
        data_len = min(self.max_samples, num_samples)
        return data_len

    def __getitem__(self, idx):
        # log.debug(f"Load {idx}")
        seriesuid, coord_x, coord_y, coord_z, nodule_class, _, _ = self.data.iloc[idx, :]
        cache_file = os.path.join(cached_data_dir, f"{seriesuid}_{coord_x}_{coord_y}_{coord_z}_{nodule_class}.pkl")
        ct_slice, center_irc = joblib.load(cache_file)

        ct_tensor = torch.tensor(ct_slice, dtype=torch.float32)
        ct_tensor = ct_tensor.unsqueeze(0)

        label_tensor = torch.tensor(
            [not nodule_class, nodule_class],
            dtype=torch.long)

        return (
            ct_tensor,
            label_tensor,
            seriesuid,
            center_irc
        )


if __name__ == '__main__':
    train_dl = DataLoader(LunaDataset(class_balance=1, to_cache=0),
                          batch_size=256,
                          num_workers=6)

    for idx, batch in enumerateWithEstimate(train_dl, "Training DL"):
        if idx > 2:
            break
        ct_t, label_t, series_uid, center_irc = batch
        print(f"{len(series_uid)} elements in batch")
        print(ct_t[0].shape)
        print(label_t.shape)

